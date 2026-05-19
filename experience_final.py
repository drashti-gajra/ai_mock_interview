"""
Production DeBERTa BIOES NER Testing Script
WITH INTEGRATED COMPREHENSIVE DATE EXTRACTION AND POSITION MATCHING
Supports batch processing, detailed debugging, and sliding window for long sequences
WITH BIOES SEQUENCE CORRECTION
WITH MONGODB COMPANY AND POSITION SEARCH
"""

from dateutil import parser
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import json
import pickle
import warnings
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
from pymongo import MongoClient
from urllib.parse import quote_plus
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import spacy

# Load spaCy for auxiliary feature extraction
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Installing spaCy model...")
    os.system("python -m spacy download en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

# ==================== GLOBAL TESTER CACHE ====================
# Add this section RIGHT AFTER your imports, BEFORE any class definitions

import threading

# Global cached tester instance for reuse
_global_tester_instance = None
_tester_initialization_lock = threading.Lock()

def get_cached_tester(
    model_path: str,
    mongodb_connection: str = None,
    mongodb_password: str = None,
    use_regex_dates: bool = True,
    use_position_matching: bool = True
) -> 'NERTester':
    """
    Get or create a cached NERTester instance.
    Thread-safe singleton pattern for MongoDB caching optimization.
    
    Args:
        model_path: Path to NER model
        mongodb_connection: MongoDB connection string
        mongodb_password: MongoDB password
        use_regex_dates: Enable regex date extraction
        use_position_matching: Enable position matching
    
    Returns:
        Cached NERTester instance
    """
    global _global_tester_instance
    
    # Fast path: if already initialized, return immediately
    if _global_tester_instance is not None:
        print("✅ Using cached NERTester instance (MongoDB already loaded)")
        return _global_tester_instance
    
    # Slow path: need to initialize
    with _tester_initialization_lock:
        # Double-check after acquiring lock (another thread might have initialized)
        if _global_tester_instance is not None:
            print("✅ Using cached NERTester instance (MongoDB already loaded)")
            return _global_tester_instance
        
        print("\n" + "="*80)
        print("🔧 INITIALIZING NERTester WITH MONGODB CACHE (ONE-TIME OPERATION)")
        print("="*80)
        
        import time
        start_time = time.time()
        
        # Create new NERTester instance (MongoDB cached here)
        _global_tester_instance = NERTester(
            model_path=model_path,
            use_regex_dates=use_regex_dates,
            use_position_matching=use_position_matching,
            mongodb_connection=mongodb_connection,
            mongodb_password=mongodb_password
        )
        
        end_time = time.time()
        
        print("\n" + "="*80)
        print(f"✅ NERTester CACHED IN MEMORY")
        print(f"   Initialization time: {end_time - start_time:.2f}s")
        print(f"   MongoDB data: CACHED")
        print(f"   Subsequent calls will be 2.5s faster")
        print("="*80 + "\n")
        
        return _global_tester_instance


def clear_tester_cache():
    """
    Clear the cached tester instance.
    Use this only if you need to reload MongoDB data or reinitialize.
    """
    global _global_tester_instance
    
    with _tester_initialization_lock:
        if _global_tester_instance is not None:
            print("🧹 Clearing cached NERTester instance...")
            _global_tester_instance = None
            print("✅ Cache cleared")
        else:
            print("ℹ️ No cached instance to clear")


def normalize_text_newlines(text: str) -> str:
    """
    Normalize newlines in text - handles both actual and escaped newlines.
    This fixes issues where MongoDB/API serialization escapes newlines.
    """
    if not text:
        return text
    
    # Count different newline types
    actual_newline_count = text.count('\n')
    escaped_newline_count = text.count('\\n')
    
    # If we have escaped newlines and few/no actual newlines, convert them
    if escaped_newline_count > 5 and actual_newline_count < 5:
        text = text.replace('\\n', '\n')
        print(f"DEBUG: Converted {escaped_newline_count} escaped newlines to actual newlines")
    
    return text



# ==================== MONGODB CONNECTION ====================

def load_companies_from_mongodb(connection_string: str, db_password: str) -> Tuple[Set[str], List[str]]:
    """
    Load company names from MongoDB Company collection.
    Returns:
        - Set of lowercase company names for quick lookup
        - List of original company names for display
    """
    try:
        # Replace password in connection string
        conn_str = connection_string.replace('<db_password>', db_password)
        
        # Connect to MongoDB
        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        db = client['Experience']
        collection = db['Company']
        
        # Fetch all companies - CORRECTED: use "Company" field, not "name"
        company_set = set()
        company_list = []
        cursor = collection.find({}, {'Company': 1, '_id': 0})
        
        for doc in cursor:
            if 'Company' in doc and doc['Company']:
                company_name = str(doc['Company']).strip()
                if company_name:
                    company_set.add(company_name.lower())
                    company_list.append(company_name)
        
        client.close()
        print(f"Loaded {len(company_set)} companies from MongoDB Company collection")
        return company_set, company_list
        
    except Exception as e:
        print(f"Error loading companies from MongoDB: {e}")
        return set(), []

def clean_line_numbers_from_entity(entity: str) -> str:
    """
    Remove line numbers from entity text while preserving quantifiable metrics.
    Also removes common prefixes like "Position:" or "Company:".
    
    Handles patterns like:
    - "61. Security Manager" -> "Security Manager"
    - "Senior Business Systems 46. Analyst 47. Sr. Business" -> "Senior Business Systems Analyst Sr. Business"
    - "Position: Managing Architect" -> "Managing Architect"
    - "Company: IBM" -> "IBM"
    But preserves "increased revenue by 91.46%" or "25.5 million"
    """
    if not entity:
        return entity
    
    # Step 0: Remove common prefixes like "Position:" or "Company:"
    cleaned = re.sub(r'^(?:Position|Company)\s*:\s*', '', entity.strip(), flags=re.IGNORECASE)
    
    # Step 1: Remove line numbers at the very beginning: "123. Text" -> "Text"
    cleaned = re.sub(r'^\d+\.\s+', '', cleaned.strip())
    
    # Step 2: Remove line numbers in the middle that are clearly line markers
    # Pattern: space + digits + period + space (but NOT followed by % or "million" or "billion")
    # Use negative lookahead to preserve percentages and financial numbers
    cleaned = re.sub(r'\s+\d+\.\s+(?![\d%]|million|billion|thousand|percent)', ' ', cleaned)
    
    # Step 3: Clean up resulting multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()


def merge_bracket_entities(entities: List[str]) -> List[str]:
    """
    Merge sequential entities where one has an opening bracket and the next has a closing bracket.
    
    Example:
        ['TATA COMMUNICATIONS LTD. (fka', 'VSNL)'] -> ['TATA COMMUNICATIONS LTD. (fka VSNL)']
    
    Args:
        entities: List of entity strings
    
    Returns:
        List of entities with bracket pairs merged
    """
    if not entities or len(entities) < 2:
        return entities
    
    merged = []
    i = 0
    
    while i < len(entities):
        current = entities[i]
        
        # Check if current entity has unmatched opening bracket
        open_count = current.count('(')
        close_count = current.count(')')
        
        if open_count > close_count and i + 1 < len(entities):
            # Check if next entity has closing bracket
            next_entity = entities[i + 1]
            next_close_count = next_entity.count(')')
            next_open_count = next_entity.count('(')
            
            if next_close_count > next_open_count:
                # Merge these two entities
                merged_entity = current + ' ' + next_entity
                merged.append(merged_entity.strip())
                print(f"   🔗 Merged entities: '{current}' + '{next_entity}' → '{merged_entity.strip()}'")
                i += 2  # Skip both entities
                continue
        
        merged.append(current)
        i += 1
    
    return merged

def load_positions_from_mongodb(connection_string: str, db_password: str) -> List[str]:
    """
    Load position names from MongoDB Position collection.
    Returns a list of position strings for exact matching.
    """
    try:
        # Replace password in connection string
        conn_str = connection_string.replace('<db_password>', db_password)
        
        # Connect to MongoDB
        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        db = client['Experience']
        collection = db['Position']
        
        # Fetch all positions - CORRECTED: use "Position" field, not "name"
        positions = []
        cursor = collection.find({}, {'Position': 1, '_id': 0})
        
        for doc in cursor:
            if 'Position' in doc and doc['Position']:
                position_name = str(doc['Position']).strip()
                if position_name:
                    positions.append(position_name)
        
        client.close()
        print(f"Loaded {len(positions)} positions from MongoDB Position collection")
        return positions
        
    except Exception as e:
        print(f"Error loading positions from MongoDB: {e}")
        return []
    

    
def load_locations_from_mongodb(connection_string: str, db_password: str) -> Set[str]:
    """
    Load location entities from MongoDB Location collection.
    Returns a set of lowercase location entities for exact matching.
    """
    try:
        # Replace password in connection string
        conn_str = connection_string.replace('<db_password>', db_password)
        
        # Connect to MongoDB
        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        db = client['Experience']
        collection = db['Location']
        
        # Fetch all locations and collect entities from all fields
        location_entities = set()
        
        # Fields to extract from Location collection
        location_fields = ['name', 'state_code', 'state_name', 'country_code', 'country_name']
        
        cursor = collection.find({})
        
        for doc in cursor:
            for field in location_fields:
                if field in doc and doc[field]:
                    value = str(doc[field]).strip()
                    if value:
                        location_entities.add(value.lower())
        
        client.close()
        print(f"Loaded {len(location_entities)} location entities from MongoDB Location collection")
        return location_entities
        
    except Exception as e:
        print(f"Error loading locations from MongoDB: {e}")
        return set()
    
def is_ongoing_date(date_string: str) -> bool:
    """
    Check if a date string contains ongoing/present indicators.
    Uses the same regex pattern as extract_all_dates() for consistency.
    """
    ongoing = (
        r"(?:Present|Now|Current|Ongoing|Till[\s]*Date|To[\s]*Date|Continuing|Today|"
        r"Nowadays|Currently|Presently|Still|Ongoing|Active|In[\s]+Progress|"
        r"In[\s]+Process|Underway|In[\s]+Operation)"
    )
    return bool(re.search(ongoing, date_string, re.IGNORECASE))


def find_location_near_date(
    date_string: str,
    raw_text: str,
    location_list: List[str],
    window_lines: int = 3
) -> Optional[str]:
    """
    Find a location string near the date in the raw text.
    Only used for ongoing/present dates.
    Searches within window_lines above and below the date line.

    Args:
        date_string: The date string to locate in text
        raw_text: The full original resume text
        location_list: List of matched location strings (e.g. ["Jacksonville, FL", "Mumbai, India"])
        window_lines: Number of lines above/below date to search

    Returns:
        The closest location string found, or None
    """
    if not date_string or not location_list:
        return None

    lines = raw_text.split('\n')

    # Find the line number where the date appears
    date_line_idx = None
    for i, line in enumerate(lines):
        if date_string.lower() in line.lower():
            date_line_idx = i
            break

    if date_line_idx is None:
        return None

    # Define search window
    search_start = max(0, date_line_idx - window_lines)
    search_end = min(len(lines), date_line_idx + window_lines + 1)
    window_text = '\n'.join(lines[search_start:search_end])
    window_text_lower = window_text.lower()

    # Find the closest location within the window
    best_location = None
    best_distance = float('inf')

    for location in location_list:
        loc_lower = location.lower()
        if loc_lower in window_text_lower:
            # Find which line it appears on and compute distance to date line
            for i in range(search_start, search_end):
                if loc_lower in lines[i].lower():
                    distance = abs(i - date_line_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_location = location
                        break

    return best_location
    
    
def contains_stopword(position: str, stopwords: Set[str]) -> bool:
    """
    Check if a position contains any stopword as an individual word.
    
    Args:
        position: The position string to check
        stopwords: Set of stopwords to check against
    
    Returns:
        True if position contains a stopword as an individual word, False otherwise
    """
    if not position:
        return False
    
    # Normalize: lowercase and split on whitespace and common punctuation
    # This handles cases like "on experience" or "Testing and Deployment"
    position_lower = position.lower()
    
    # Split by whitespace and punctuation, keeping only alphanumeric words
    words = re.findall(r'\b[a-z]+\b', position_lower)
    
    # Check if any word is a stopword
    for word in words:
        if word in stopwords:
            return True
    
    return False


def filter_triplets_by_stopwords(triplets: List[Dict], stopwords: Set[str]) -> List[Dict]:
    """
    Remove triplets where the position contains any stopword.
    
    Args:
        triplets: List of triplet dictionaries
        stopwords: Set of stopwords to filter against
    
    Returns:
        Filtered list of triplets
    """
    if not stopwords:
        return triplets
    
    filtered = []
    removed_count = 0
    
    for triplet in triplets:
        position = triplet.get('Position', '') or triplet.get('position', '')
        
        if contains_stopword(position, stopwords):
            removed_count += 1
            company = triplet.get('Company', '') or triplet.get('company', 'N/A')
            print(f"   ✗ Removed: '{position}' at {company}")
        else:
            filtered.append(triplet)
    
    if removed_count > 0:
        print(f"\n ✅ Filtered out {removed_count} entries containing stopwords")
    else:
        print(f"\n ✅ No stopwords found in positions")
    
    return filtered

def get_education_keywords() -> Set[str]:
    education_keywords = {
        # Degree levels
        'bachelor', 'bachelors', "bachelor's",
        'master', 'masters', "master's",
        'doctorate', 'doctoral', 'post-doctoral', 'postdoctoral',
        'undergraduate', 'postgraduate',
        'honours', 'honors',

        # Degree abbreviations
        'bsc', 'b.sc', 'bs', 'b.s', 'ba', 'b.a',
        'b.tech', 'btech', 'b.e', 'be',
        'b.com', 'bcom',
        'msc', 'm.sc', 'ms', 'm.s', 'ma', 'm.a',
        'm.tech', 'mtech', 'm.e', 'me',
        'm.com', 'mcom',
        'mba',
        'phd', 'ph.d', 'ph.d.',
        'llb', 'll.b', 'llm', 'll.m',
        'mbbs', 'md', 'm.d',

        # Institution type words
        'university', 'universities',
        'college', 'colleges',
        'institute', 'institution', 'institutions',
        'school',
        'academy',
        'polytechnic',
        'conservatory',

        # Enrollment/completion status words
        'graduated', 'pursuing', 'certified',
        'diploma',
        'certificate',
        'program',
    }
    return education_keywords


def contains_education_entity(text: str, education_keywords: Set[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower().strip()
    # Check each keyword as a whole word match
    for keyword in education_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def filter_education_triplets(
    triplets: List[Dict],
    education_keywords: Set[str] = None
) -> Tuple[List[Dict], List[Dict]]:

    if education_keywords is None:
        education_keywords = get_education_keywords()

    valid_triplets = []
    filtered_triplets = []

    print(f"\n{'='*80}")
    print("FILTERING EDUCATION-RELATED TRIPLETS")
    print(f"{'='*80}")

    for triplet in triplets:
        position = triplet.get('Position', '') or triplet.get('position', '')
        company  = triplet.get('Company',  '') or triplet.get('company',  'N/A')
        year     = triplet.get('Year',     '') or triplet.get('year',     'N/A')

        position_is_education = contains_education_entity(position, education_keywords)
        company_is_education  = contains_education_entity(company,  education_keywords)

        # OR logic: either entity being education is enough to filter
        if position_is_education and company_is_education:
            filtered_triplets.append(triplet)
            print(f"   ✗ FILTERED: '{position}' at {company} ({year})")
            if position_is_education:
                print(f"     Reason: Position contains education keyword")
            if company_is_education:
                print(f"     Reason: Company contains education keyword")
        else:
            valid_triplets.append(triplet)

    print(f"\n{'='*80}")
    print(f"EDUCATION FILTERING RESULTS:")
    print(f"  Total triplets:              {len(triplets)}")
    print(f"  Valid (work experience):     {len(valid_triplets)}")
    print(f"  Filtered (education):        {len(filtered_triplets)}")
    print(f"{'='*80}\n")

    return valid_triplets, filtered_triplets


def filter_education_from_combined_triplets(
    triplets: List[Dict],
    duos: List[Dict],
    education_keywords: Set[str] = None
) -> Tuple[List[Dict], List[Dict], int, int]:

    if education_keywords is None:
        education_keywords = get_education_keywords()

    print(f"\n{'='*80}")
    print("FILTERING EDUCATION-RELATED TRIPLETS")
    print(f"{'='*80}")

    valid_triplets    = []
    filtered_triplets = []

    for triplet in triplets:
        position = triplet.get('Position', '') or triplet.get('position', '')
        company  = triplet.get('Company',  '') or triplet.get('company',  'N/A')
        year     = triplet.get('Year',     '') or triplet.get('year',     'N/A')

        position_is_education = contains_education_entity(position, education_keywords)
        company_is_education  = contains_education_entity(company,  education_keywords)

        if position_is_education and company_is_education:
            filtered_triplets.append(triplet)
            print(f"   ✗ FILTERED: '{position}' at {company} ({year})")
            if position_is_education:
                print(f"     Reason: Position contains education keyword")
            if company_is_education:
                print(f"     Reason: Company contains education keyword")
        else:
            valid_triplets.append(triplet)

    print(f"\n{'='*80}")
    print("FILTERING EDUCATION-RELATED DUOs")
    print(f"{'='*80}")

    valid_duos    = []
    removed_duos  = []

    for duo in duos:
        position = duo.get('Position', '') or duo.get('position', '')
        company  = duo.get('Company',  '') or duo.get('company',  'N/A')
        year     = duo.get('Year',     '') or duo.get('year',     'N/A')

        position_is_education = contains_education_entity(position, education_keywords)
        company_is_education  = contains_education_entity(company,  education_keywords)

        if position_is_education and company_is_education:
            removed_duos.append(duo)
            print(f"   ✗ FILTERED DUO: '{position}' at {company} ({year})")
            if position_is_education:
                print(f"     Reason: Position contains education keyword")
            if company_is_education:
                print(f"     Reason: Company contains education keyword")
        else:
            valid_duos.append(duo)

    triplets_removed = len(filtered_triplets)
    duos_removed     = len(removed_duos)

    print(f"\n{'='*80}")
    print(f"DUO FILTERING RESULTS:")
    print(f"  Total DUOs:   {len(duos)}")
    print(f"  Valid:        {len(valid_duos)}")
    print(f"  Filtered:     {len(removed_duos)}")
    print(f"{'='*80}\n")

    print(f"📊 Education Filtering Summary:")
    print(f"   Triplets removed: {triplets_removed}")
    print(f"   DUOs removed:     {duos_removed}")
    print(f"   Total removed:    {triplets_removed + duos_removed}")

    return valid_triplets, valid_duos, triplets_removed, duos_removed


def search_companies_around_dates(
    text: str,
    date_pivots: List[str],
    company_set: Set[str],
    company_list: List[str],
    ner_companies: List[str],
    window_size: int = 7
) -> List[str]:
    """
    Search for company names around date pivots using exact matching.
    
    Args:
        text: Original text
        date_pivots: List of extracted dates to use as pivots
        company_set: Set of lowercase company names for quick checks
        company_list: List of original company names
        ner_companies: Companies already extracted by NER (to avoid duplicates)
        window_size: Number of words to check on each side of pivot
    
    Returns:
        List of newly found company names
    """
    if not company_set or not date_pivots:
        return []
    
    # Create a mapping from lowercase to original names
    company_map = {}
    for company in company_list:
        company_map[company.lower()] = company
    
    # Create a set of normalized NER companies to avoid duplicates
    ner_companies_normalized = {comp.lower().strip() for comp in ner_companies}
    
    # Split text into words
    words = text.split()
    
    # Create a mapping of date positions in the text
    date_positions = []
    for date in date_pivots:
        # Find all occurrences of this date in the text
        date_pattern = re.escape(date)
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            start_pos = match.start()
            # Find word index at this position
            current_pos = 0
            for word_idx, word in enumerate(words):
                word_start = text.find(word, current_pos)
                word_end = word_start + len(word)
                if start_pos >= word_start and start_pos < word_end:
                    date_positions.append(word_idx)
                    break
                current_pos = word_end
    
    # Remove duplicate positions
    date_positions = list(set(date_positions))
    
    found_companies = []
    found_companies_set = set()
    
    # For each date position, search surrounding words
    for date_pos in date_positions:
        # Define window boundaries
        start_idx = max(0, date_pos - window_size)
        end_idx = min(len(words), date_pos + window_size + 1)
        
        # Check each word in the window
        for idx in range(start_idx, end_idx):
            if idx == date_pos:
                continue  # Skip the date itself
            
            word = words[idx].strip()
            # Remove common punctuation for matching
            clean_word = re.sub(r'[.,;:!?\'"()\[\]{}<>]', '', word)
            
            if not clean_word:
                continue
            
            # Exact match search in company set
            word_lower = clean_word.lower()
            if word_lower in company_set:
                # Get the original casing
                original_company = company_map.get(word_lower, clean_word)
                
                # Check if not already in NER companies or found companies
                if (original_company.lower() not in ner_companies_normalized and 
                    original_company.lower() not in found_companies_set):
                    found_companies.append(original_company)
                    found_companies_set.add(original_company.lower())
    
    return found_companies


# ==================== DATE EXTRACTION FUNCTIONS ====================

import re
from typing import List

def extract_all_dates(text: str) -> List[str]:
    """
    Extract all possible date-like patterns from raw resume/job text.
    Returns only exact, non-overlapping matches with preference for longer matches.
    Includes duration expressions like "X Years", "X Months".
    
    UPDATED: Added patterns for "Oct ’22 – Present", "Oct ’22 – July '24", etc.
    FIXED: Uses whitespace-based tokenization to prevent extracting half dates.
    FIXED: Added proper boundaries for month/year formats like "01/2015 - 04/2018"
    FIXED: Filters out standalone duration expressions but keeps durations in date ranges
    """
    
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|" \
             r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|" \
             r"January|February|March|April|June|July|August|September|October|November|December|" \
             r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC|" \
             r"Janv|Févr|Mars|Avr|Mai|Juin|Juill|Août|Sept|Oct|Nov|Déc|" \
             r"Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)"
    
    seasons = r"(?:Spring|Summer|Fall|Autumn|Winter|" \
              r"SPRING|SUMMER|FALL|AUTUMN|WINTER|" \
              r"Spr|Sum|Fall|Aut|Win|" \
              r"Vernal|Summer|Autumnal|Winter|" \
              r"Dry[\s]+Season|Wet[\s]+Season|Rainy[\s]+Season|" \
              r"Harvest[\s]+Season|Holiday[\s]+Season)"
    
    academic_terms = r"(?:Fall[\s]+Semester|Spring[\s]+Semester|Summer[\s]+Semester|Winter[\s]+Semester|" \
                     r"Fall[\s]+Quarter|Spring[\s]+Quarter|Summer[\s]+Quarter|Winter[\s]+Quarter|" \
                     r"Fall[\s]+Term|Spring[\s]+Term|Summer[\s]+Term|Winter[\s]+Term|" \
                     r"First[\s]+Semester|Second[\s]+Semester|Third[\s]+Semester|Fourth[\s]+Semester|" \
                     r"Semester[\s]+1|Semester[\s]+2|Semester[\s]+3|Semester[\s]+4|" \
                     r"Trimester[\s]+1|Trimester[\s]+2|Trimester[\s]+3)"
    
    year_full = r"\d{4}"
    year_short = r"\d{2}"         # bare two-digit year, no apostrophe
    year_short_apos = r"['''\u2018\u2019]?\d{2}"  # with optional apostrophe (for backward compat)
    year_any = rf"(?:{year_full}|{year_short})"
    sep = r"[\s]*(?:-|–|—|−|‐|‑|‒|―|⁻|₋|﹣|－|to|till|until|through|thru|throughout|and|&)[\s]*"
    
    ongoing = r"(?:Present|Now|Current|Ongoing|Till[\s]*Date|To[\s]*Date|Continuing|Today|" \
              r"Nowadays|Currently|Presently|Still|Ongoing|Active|In[\s]+Progress|" \
              r"In[\s]+Process|Underway|In[\s]+Operation)"
    
    patterns = [
        # ========== NEW: Date ranges with duration in parentheses (HIGHEST PRIORITY) ==========
        (rf"{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}[\s]*\(\s*\d+[\s]*(?:Years?|Yrs?|Months?|Mos?)\s*\)", 20),  # June 2025 - June 2035 (10 years)
        (rf"{year_full}[\s]*{sep}[\s]*{year_full}[\s]*\(\s*\d+[\s]*(?:Years?|Yrs?|Months?|Mos?)\s*\)", 20),  # 2025-2035 (10 years)
        (rf"\d{{1,2}}/\d{{4}}[\s]*{sep}[\s]*\d{{1,2}}/\d{{4}}[\s]*\(\s*\d+[\s]*(?:Years?|Yrs?|Months?|Mos?)\s*\)", 20),  # 06/2025 - 06/2035 (10 years)
        
        # ========== NEW: PATTERNS FOR NUMERIC MONTH/YEAR FORMATS (HIGH PRIORITY) ==========
        # Fixed boundaries for "01/2015 - 04/2018" type patterns
        (rf"\b\d{{1,2}}/\d{{4}}\s*{sep}\s*\d{{1,2}}/\d{{4}}\b", 15),  # 01/2015 - 04/2018
        (rf"\b\d{{1,2}}/\d{{4}}\s*{sep}\s*{ongoing}\b", 15),  # 05/2018 - Present
        (rf"\b\d{{1,2}}\.\d{{4}}\s*{sep}\s*\d{{1,2}}\.\d{{4}}\b", 15),  # 01.2015 - 04.2018
        (rf"\b\d{{1,2}}-\d{{4}}\s*{sep}\s*\d{{1,2}}-\d{{4}}\b", 15),  # 01-2015 - 04-2018
         # ========== MM/YY – MM/YY (two-digit year, numeric month) ==========
        # Handles: 06/22 – 10/23, 6/22 - 10/23, 06/22 – Present
        (rf"\b\d{{1,2}}/\d{{2}}\s*{sep}\s*\d{{1,2}}/\d{{2}}\b", 14),   # 06/22 – 10/23
        (rf"\b\d{{1,2}}/\d{{2}}\s*{sep}\s*{ongoing}\b", 14),             # 06/22 – Present
        (rf"\b\d{{1,2}}\.\d{{2}}\s*{sep}\s*\d{{1,2}}\.\d{{2}}\b", 14),  # 06.22 – 10.23
        (rf"\b\d{{1,2}}\.\d{{2}}\s*{sep}\s*{ongoing}\b", 14),            # 06.22 – Present
        (rf"\b\d{{1,2}}-\d{{2}}\s*{sep}\s*\d{{1,2}}-\d{{2}}\b", 14),   # 06-22 – 10-23
        (rf"\b\d{{1,2}}-\d{{2}}\s*{sep}\s*{ongoing}\b", 14),             # 06-22 – Present
        
        
        # ========== YOUR SPECIFIC PATTERNS ==========
        # Oct ’22 – Present (smart quotes with dash)
        (rf"\b{months}\s*[''']{year_short}\s*–\s*{ongoing}\b", 15),
        # Oct '22 – Present (regular apostrophe with dash)
        (rf"\b{months}\s*[']{year_short}\s*–\s*{ongoing}\b", 15),
        # Oct ’22 - Present (smart quotes with hyphen)
        (rf"\b{months}\s*[''']{year_short}\s*-\s*{ongoing}\b", 15),
        # Oct '22 - Present (regular apostrophe with hyphen)
        (rf"\b{months}\s*[']{year_short}\s*-\s*{ongoing}\b", 15),
        
        # Oct ’22 – July '24 (smart to smart with different quotes)
        (rf"\b{months}\s*[''']{year_short}\s*–\s*{months}\s*[']{year_short}\b", 15),
        (rf"\b{months}\s*[']{year_short}\s*–\s*{months}\s*[''']{year_short}\b", 15),
        # Oct '22 – July '24 (regular to regular)
        (rf"\b{months}\s*[']{year_short}\s*–\s*{months}\s*[']{year_short}\b", 15),
        # Oct ’22 – July ’24 (smart to smart same)
        (rf"\b{months}\s*[''']{year_short}\s*–\s*{months}\s*[''']{year_short}\b", 15),
        
        # Oct ’22 - July '24 (with hyphen instead of dash)
        (rf"\b{months}\s*[''']{year_short}\s*-\s*{months}\s*[']{year_short}\b", 15),
        (rf"\b{months}\s*[']{year_short}\s*-\s*{months}\s*[''']{year_short}\b", 15),
        (rf"\b{months}\s*[']{year_short}\s*-\s*{months}\s*[']{year_short}\b", 15),
        (rf"\b{months}\s*[''']{year_short}\s*-\s*{months}\s*[''']{year_short}\b", 15),
        
        # ========== HIGHEST PRIORITY: Month-based patterns with ongoing ==========
        # Using \b for proper whitespace tokenization
        (rf"\b{months}[\s,]+{year_any}{sep}{ongoing}\b", 14),  # Sep 2023 - Present
        (rf"\b{months}[\s]*\n[\s]*{year_any}[\s]*{sep}[\s]*{ongoing}\b", 14),  # Multi-line with ongoing
        
        # ========== NEW: Patterns for dates split across lines ==========
        # Handle: "June\n2013 - June 2014" (month on one line, rest on next)
        (rf"\b{months}[\s]*\n[\s]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\b", 14),  # Month\nYear - Month Year
        (rf"\b{months}[\s]*\n[\s]*{year_any}[\s]*{sep}[\s]*{months}[\s]*\n[\s]*{year_any}\b", 14),  # Month\nYear - Month\nYear
        # Handle: "June 2013\n- June 2014" (first date complete, separator on next line)
        (rf"\b{months}[\s,]+{year_any}[\s]*\n[\s]*{sep}[\s]*{months}[\s,]+{year_any}\b", 14),
        # Handle: "June 2013 -\nJune 2014" (separator at end, second date on next line)  
        (rf"\b{months}[\s,]+{year_any}[\s]*{sep}[\s]*\n[\s]*{months}[\s,]+{year_any}\b", 14),
        
        # ========== NEW: Patterns for dates split across lines WITH LINE NUMBERS ==========
        # Handle: "June\n52. 2013 - June 2014" (month on one line, line number + rest on next)
        (rf"\b{months}[\s]*\n[\s]*\d+\.[\s]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\b", 14),
        # Handle: "June\n52. 2013\n53. - June 2014"
        (rf"\b{months}[\s]*\n[\s]*\d+\.[\s]*{year_any}[\s]*\n[\s]*\d*\.?[\s]*{sep}[\s]*{months}[\s,]+{year_any}\b", 14),
        # Handle line number after separator: "June 2013 -\n52. June 2014"
        (rf"\b{months}[\s,]+{year_any}[\s]*{sep}[\s]*\n[\s]*\d+\.[\s]*{months}[\s,]+{year_any}\b", 14),
        
        # ========== HIGH PRIORITY: Month-to-month patterns ==========
        (rf"\b{months}['']?{year_any}[\s]*{sep}[\s]*{months}['']?{year_any}\b", 13),  # Aug'2019 to Nov'2019
        (rf"\b{months}['']?[\s]*{year_any}[\s]*{sep}[\s]*{months}['']?[\s]*{year_any}\b", 13),  # Aug '2019 to Nov '2019
        (rf"\b{months}[\s,]+{year_any}{sep}{months}[\s,]+{year_any}\b", 13),  # May 2022 - December 2022
        (rf"\b{months}\s*[-–]\s*{year_any}{sep}{months}\s*[-–]\s*{year_any}\b", 13),  # Month-year to month-year
        # Add these patterns to handle smart quotes and apostrophes
        (rf"\b{months}[\s]*[''']{year_short}[\s]*{sep}[\s]*{months}[\s]*[''']{year_short}\b", 13),  # March '22 – April '24
        (rf"\b{months}[\s]*[''']{year_short}{sep}{months}[\s]*[''']{year_short}\b", 13),  # March'22–April'24
        (rf"\b{months}[\s]*[''']{year_short}[\s]*{sep}[\s]*{months}[\s]*[''']{year_short}[,.;]?\b", 13),  # With trailing punctuation
        # Add these patterns for month-year with smart quotes to ongoing
        (rf"\b{months}[\s]*[''']{year_short}[\s]*{sep}[\s]*{ongoing}\b", 14),  # July '21 – Present
        (rf"\b{months}[\s]*[''']{year_short}{sep}{ongoing}\b", 14),  # July'21–Present
        (rf"\b{months}[\s]*[''']{year_short}[\s]*{sep}[\s]*{ongoing}[,.;]?\b", 14),  # With trailing punctuation

        # Also add patterns for mixed quote styles with ongoing
        (rf"\b{months}[\s]*['\"`]{year_short}[\s]*{sep}[\s]*{ongoing}\b", 14),  # Mixed quote styles with ongoing
        # ========== NEW: "Month 'YY – Month 'YY" — explicit SPACE before apostrophe ==========
# Covers: March '22 – April '24, Mar '22 - Apr '24 (all apostrophe encodings)
        (rf"\b{months}\s+'\d{{2}}\s*{sep}\s*{months}\s+'\d{{2}}\b", 16),
        (rf"\b{months}\s+\u2019\d{{2}}\s*{sep}\s*{months}\s+\u2019\d{{2}}\b", 16),
        (rf"\b{months}\s+\u2018\d{{2}}\s*{sep}\s*{months}\s+\u2018\d{{2}}\b", 16),
        # Covers: March '22 – Present, Mar '22 - Present (all apostrophe encodings)
        (rf"\b{months}\s+'\d{{2}}\s*{sep}\s*{ongoing}\b", 16),
        (rf"\b{months}\s+\u2019\d{{2}}\s*{sep}\s*{ongoing}\b", 16),
        (rf"\b{months}\s+\u2018\d{{2}}\s*{sep}\s*{ongoing}\b", 16),

        # Also add patterns for mixed quote styles
        (rf"\b{months}[\s]*[''']{year_short}[\s]*{sep}[\s]*{months}[\s]*['\"`]{year_short}\b", 12),  # Mixed quote styles
        # Add these patterns to the HIGH PRIORITY: Month-to-month patterns section
        (rf"\b{months}\.[\s]*{year_any}[\s]*{sep}[\s]*{months}\.[\s]*{year_any}\b", 13),  # Nov. 2021 - Sep. 2023
        (rf"\b{months}\.[\s]*{year_any}{sep}{months}\.[\s]*{year_any}\b", 13),  # Nov.2021-Sep.2023
        (rf"\b{months}\.[\s]*{year_any}[\s]*{sep}[\s]*{months}\.[\s]*{year_any}\b", 13),  # Nov. 2021 - Sep. 2023 (with spaces)

        # Also add these to handle mixed formats
        (rf"\b{months}\.[\s]*{year_any}[\s]*{sep}[\s]*{months}[\s]*{year_any}\b", 12),  # Nov. 2021 - Sep 2023
        (rf"\b{months}[\s]*{year_any}[\s]*{sep}[\s]*{months}\.[\s]*{year_any}\b", 12),  # Nov 2021 - Sep. 2023
        
        # ========== MEDIUM-HIGH PRIORITY: Full date patterns ==========
        (rf"\b{year_full}[\s,]+{months}{sep}{year_full}[\s,]+{months}\b", 12),
        (rf"\b{year_full}[\s,]+{months}{sep}{ongoing}\b", 12),
        (rf"\b{months}[\s]+\d{{1,2}}(?:st|nd|rd|th)?,?[\s]+{year_full}{sep}{months}[\s]+\d{{1,2}}(?:st|nd|rd|th)?,?[\s]+{year_full}\b", 12),
        (rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s]+{months}[\s,]+{year_full}{sep}\d{{1,2}}(?:st|nd|rd|th)?[\s]+{months}[\s,]+{year_full}\b", 12),
        
        # ========== NEW: PARENTHESES AND BRACKETS PATTERNS (HIGH PRIORITY) ==========
        (rf"\(\s*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\s*\)", 13),  # (Sep 2023 - Dec 2023)
        (rf"\(\s*{year_full}[\s]*{sep}[\s]*{year_full}\s*\)", 12),  # (2023-2024)
        (rf"\(\s*{year_full}[\s]*{sep}[\s]*{ongoing}\s*\)", 12),  # (2023-Present)
        (rf"\(\s*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{ongoing}\s*\)", 13),  # (Sep 2023 - Present)
        (rf"\[\s*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\s*\]", 13),  # [Sep 2023 - Dec 2023]
        (rf"\[\s*{year_full}[\s]*{sep}[\s]*{year_full}\s*\]", 12),  # [2023-2024]
        (rf"\{{s*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\s*\}}", 13),  # {Sep 2023 - Dec 2023}
        (rf"\{{s*{year_full}[\s]*{sep}[\s]*{year_full}\s*\}}", 12),  # {2023-2024}
        
        # ========== NEW: VARIOUS BRACKET AND PUNCTUATION COMBINATIONS ==========
        (rf"«\s*{year_full}[\s]*{sep}[\s]*{year_full}\s*»", 12),  # «2023-2024»
        (rf"‹\s*{year_full}[\s]*{sep}[\s]*{year_full}\s*›", 12),  # ‹2023-2024›
        (rf"\"{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\"", 13),  # "Sep 2023 - Dec 2023"
        (rf"'{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}'", 13),  # 'Sep 2023 - Dec 2023'
        (rf"`{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}`", 13),  # `Sep 2023 - Dec 2023`
        
        # ========== NEW: PREFIXED DATE RANGES ==========
        (rf"(?:Duration|Period|Time|Span|Interval|Term)[\s]*:[\s]*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}", 13),
        (rf"(?:From|Since)[\s]+{months}[\s,]*{year_any}[\s]*(?:to|until|till|through)[\s]+{months}[\s,]*{year_any}", 13),
        (rf"(?:Between)[\s]+{months}[\s,]*{year_any}[\s]+(?:and)[\s]+{months}[\s,]*{year_any}", 13),
        (rf"(?:Date|Dates)[\s]*:[\s]*{year_full}[\s]*{sep}[\s]*{year_full}", 12),
        (rf"(?:Year|Years)[\s]*:[\s]*{year_full}[\s]*{sep}[\s]*{year_full}", 12),
        
        # ========== SEASON AND ACADEMIC TERM PATTERNS ==========
        (rf"\b{seasons}[\s,]*{year_any}{sep}{seasons}[\s,]*{year_any}\b", 11),
        (rf"\b{seasons}[\s,]*{year_any}{sep}{ongoing}\b", 11),
        (rf"\b{academic_terms}[\s,]*{year_any}{sep}{academic_terms}[\s,]*{year_any}\b", 11),
        (rf"\b{academic_terms}[\s,]*{year_any}{sep}{ongoing}\b", 11),
        
        # ========== MEDIUM PRIORITY: Year-only patterns with various separators ==========
        # FIXED: Added \b to prevent partial matches like "2010 - 20" from "2010 - 2015"
        (rf"\b{year_full}\s*[-–]\s*{ongoing}\b", 9),  # 2021 - Present
        (rf"\b{year_full}\s*[-–]\s*{year_full}\b", 9),  # 2012-2015 (FIXED: was matching "2012-20")
        (rf"\b{year_full}\s*(?:–|—)\s*{ongoing}\b", 9),  # Em dash with ongoing
        (rf"\b{year_full}\s+to\s+{ongoing}\b", 9),  # to ongoing
        (rf"\b{year_full}\s+to\s+{year_full}\b", 9),  # to full year
        (rf"\b{year_full}[\s]*{sep}[\s]*{ongoing}[,.;]?\b", 9),  # with punctuation
        
        # ========== NEW: YEAR-ONLY PATTERNS WITH PUNCTUATION ==========
        (rf"\b{year_full}\s*/\s*{year_full}\b", 10),  # 2023/2024
        (rf"\b{year_full}\s*\\\s*{year_full}\b", 10),  # 2023\2024
        (rf"\b{year_full}\s*&\s*{year_full}\b", 10),  # 2023 & 2024
        (rf"\b{year_full}\s*,\s*{year_full}\b", 10),  # 2023, 2024
        (rf"\b{year_full}\s*;\s*{year_full}\b", 10),  # 2023; 2024
        (rf"\b{year_full}\s*\.\s*{year_full}\b", 10),  # 2023.2024
        
        # Multi-line patterns
        (rf"\b{months}[\s]*\n[\s]*{year_any}[\s]*{sep}[\s]*{months}[\s]*\n[\s]*{year_any}\b", 12),
        (rf"\b{months}[\s]+{year_any}[\s]*\n[\s]*{sep}[\s]*\n[\s]*{months}[\s]+{year_any}\b", 12),
        (rf"\b{year_full}[\s]*\n[\s]*{sep}[\s]*\n[\s]*{year_full}\b", 9),
        
        # Patterns with various apostrophe and quote styles
        (rf"\b{months}[`'\"''']{year_any}[\s]*{sep}[\s]*{months}[`'\"''']{year_any}\b", 12),
        (rf"\b{months}[`'\"'''][\s]*{year_any}[\s]*{sep}[\s]*{months}[`'\"'''][\s]*{year_any}\b", 12),
        
        # Patterns with missing or extra spaces (printing errors)
        (rf"\b{months}{year_any}[\s]*{sep}[\s]*{months}{year_any}\b", 11),
        (rf"\b{months}[\s]+{year_any}[\s]*{sep}[\s]*{months}{year_any}\b", 11),
        (rf"\b{months}{year_any}[\s]*{sep}[\s]*{months}[\s]+{year_any}\b", 11),
        
        # Patterns with various separator styles
        (rf"\b{months}[\s,]*{year_any}[\s]*(?:–|—)[\s]*{months}[\s,]*{year_any}\b", 12),
        
        # Patterns with "to" as separator (more explicit)
        (rf"\b{months}[\s,]*{year_any}[\s]+to[\s]+{months}[\s,]*{year_any}\b", 12),
        (rf"\b{year_full}[\s]+to\s+{year_full}\b", 10),
        
        # Patterns with parentheses and brackets (existing)
        (rf"\({months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\)", 11),
        (rf"\[{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\]", 11),
        (rf"\({year_full}[\s]*{sep}[\s]*{year_full}\)", 8),
        
        # Patterns with trailing/leading punctuation
        (rf"\b{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}[,.;]?\b", 11),
        (rf"\b{year_full}[\s]*{sep}[\s]*{year_full}[,.;]?\b", 8),
        
        # ========== NEW: COMPOUND DATE PATTERNS ==========
        (rf"\b{months}[\s,]*{year_any}\s*\(\s*{year_full}[\s]*{sep}[\s]*{year_full}\s*\)", 12),  # Sep 2023 (2023-2024)
        (rf"\b{year_full}[\s]*{sep}[\s]*{year_full}[\s]*[/\\][\s]*{year_full}[\s]*{sep}[\s]*{year_full}\b", 10),  # 2020-2021/2022-2023
        (rf"\b{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}[\s]*(?:and|&)[\s]*{months}[\s,]*{year_any}[\s]*{sep}[\s]*{months}[\s,]*{year_any}\b", 11),
        
        # ========== NEW: NUMERIC DATE RANGES WITH VARIOUS FORMATS ==========
        (rf"\b\d{{1,2}}\.\d{{1,2}}\.\d{{4}}[\s]*{sep}[\s]*\d{{1,2}}\.\d{{1,2}}\.\d{{4}}\b", 10),
        (rf"\b\d{{1,2}}-\d{{1,2}}-\d{{4}}[\s]*{sep}[\s]*\d{{1,2}}-\d{{1,2}}-\d{{4}}\b", 10),
        (rf"\b\d{{4}}/\d{{1,2}}/\d{{1,2}}[\s]*{sep}[\s]*\d{{4}}/\d{{1,2}}/\d{{1,2}}\b", 10),
        (rf"\b\d{{1,2}}\s+{months}\s+\d{{4}}[\s]*{sep}[\s]*\d{{1,2}}\s+{months}\s+\d{{4}}\b", 12),
        
        # ========== LOWER PRIORITY: Generic patterns ==========
        # FIXED: Added proper boundaries to prevent partial matches
        (rf"\b{year_full}[\s]{sep}[\s]{year_full}\b", 7),
        (rf"\b{year_full}/{year_full}\b", 7),
        (rf"\b{year_full}\s*[-–]\s*{year_full}\b", 7),
        (rf"\bQ[1-4]{sep}Q[1-4][\s,]+{year_full}\b", 7),
        (rf"\bQ[1-4][\s,]+{year_full}{sep}Q[1-4][\s,]+{year_full}\b", 7),
        (rf"\b{year_full}[\s]+(?:Spring|Summer|Fall|Autumn|Winter){sep}{year_full}[\s]+(?:Spring|Summer|Fall|Autumn|Winter)\b", 7),
        (rf"\b(?:Spring|Summer|Fall|Autumn|Winter)[\s,]+{year_full}{sep}(?:Spring|Summer|Fall|Autumn|Winter)[\s,]+{year_full}\b", 7),
        (rf"\bH[1-2]{sep}H[1-2][\s,]+{year_full}\b", 7),
        (rf"\b{months}{sep}{months}[\s,]+{year_full}\b", 6),
        (rf"\b{seasons}{sep}{seasons}[\s,]+{year_full}\b", 6),
        (rf"\b(?:between|from)[\s]+{year_full}[\s]+(?:and|to)[\s]+{year_full}\b", 6),
        (rf"\b(?:between|from)[\s]+{months}[\s,]+{year_any}[\s]+(?:and|to)[\s]+{months}[\s,]+{year_any}\b", 6),
        (rf"\b(?:between|from)[\s]+{seasons}[\s,]*{year_any}[\s]+(?:and|to)[\s]+{seasons}[\s,]*{year_any}\b", 6),
        
        # ========== LOWEST PRIORITY: Duration patterns (will be filtered later) ==========
        # These are kept with priority 0 but will be filtered out if standalone
        (r"\b\d+[\s](?:\+[\s])?(?:Years?|Yrs?)(?:[\s]+and[\s]+\d+[\s]*(?:Months?|Mos?))?\b", 0),
        (r"\b\d+[\s]*(?:Months?|Mos?)\b", 0),
        (r"\b\d+[\s]*(?:Weeks?|Wks?)\b", 0),
        (r"\b\d+[\s]*(?:Days?)\b", 0),
        (r"\b\d+[\s]+(?:years?|months?|weeks?|days?)[\s]+ago\b", 0),
        (r"(?:for)[\s]+\d+[\s]+(?:years?|months?)[\s]+(?:from|starting)[\s]+(?:\d{{4}})", 0),
        (r"\b(?:Last|Past|Previous)[\s]+(?:\d+[\s]+)?(?:years?|months?|weeks?)\b", 0),
        (r"\b(?:Next|Upcoming|Following)[\s]+(?:\d+[\s]+)?(?:years?|months?|weeks?)\b", 0),
        (r"\b\d+[\s]*[-–][\s]*(?:Year|Month|Week|Day)s?\b", 0),
        (r"\b\d+[\s]*(?:\.|,)[\s]*\d+[\s]*(?:years?|months?)\b", 0),
        (r"\b\d+[\s]*[/\\][\s]*\d+[\s]*(?:years?|months?)\b", 0),
    ]
    
    all_matches = []
    for pattern, priority in patterns:
        try:
            compiled = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
            for match in compiled.finditer(text):
                match_text = re.sub(r'\s+', ' ', match.group(0).strip())
                if len(match_text) < 2:
                    continue
                all_matches.append({
                    'text': match_text,
                    'start': match.start(),
                    'end': match.end(),
                    'priority': priority,
                    'length': len(match_text)
                })
        except re.error:
            continue
    
    all_matches.sort(key=lambda x: (x['priority'], x['length'], -x['start']), reverse=True)
    
    final_matches = []
    used_ranges = []
    for match in all_matches:
        start, end = match['start'], match['end']
        overlaps = any(not (end <= u_start or start >= u_end) for u_start, u_end in used_ranges)
        if not overlaps:
            final_matches.append(match['text'])
            used_ranges.append((start, end))
    
    # Clean extracted dates: remove embedded line numbers (e.g., "June 52. 2013" -> "June 2013")
    def clean_date_line_numbers(date_text):
        """Remove line numbers embedded in dates from line-numbered text."""
        # Pattern to remove line numbers like "52. " or "123. " in the middle of dates
        cleaned = re.sub(r'\s+\d+\.\s+', ' ', date_text)
        # Also handle at the start
        cleaned = re.sub(r'^\d+\.\s+', '', cleaned)
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    cleaned_matches = [clean_date_line_numbers(match) for match in final_matches]
    
    # NEW: Function to check if we should keep a date match
    def should_keep_date(date_text):
        """Determine if a date should be kept or filtered out."""
        # Convert to lowercase for easier matching
        date_lower = date_text.lower()
        
        # Check if it contains time units (year, month, week, day)
        time_units = ['year', 'month', 'week', 'day', 'yr', 'mo', 'wk', 'yrs', 'mos', 'wks']
        is_duration = any(unit in date_lower for unit in time_units)
        
        # Check if it's part of a date range (has separator)
        separators = ['-', '–', '—', 'to', 'till', 'until', 'through']
        has_separator = any(sep in date_text for sep in separators)
        
        # Check if it's in parentheses (could be additional info like "(10 years)")
        has_parentheses = date_text.strip().startswith('(') and date_text.strip().endswith(')')
        
        # Check if it contains actual date components (months, years)
        month_pattern = re.compile(months, re.IGNORECASE)
        has_month = bool(month_pattern.search(date_text))
        has_year = bool(re.search(r'\d{4}', date_text))
        
        # Check if it's a date range with duration in parentheses
        # e.g., "June 2025 - June 2035 (10 years)"
        has_date_range_with_duration = (has_separator and 
                                        '(' in date_text and 
                                        ')' in date_text and
                                        is_duration)
        
        # Keep if:
        # 1. It's a date range (has separator) - always keep
        # 2. It has month or year components - likely a real date
        # 3. It's a date range with duration in parentheses
        # Remove if:
        # 1. It's a duration without any date separator
        # 2. It doesn't contain month or year components
        # 3. It's not in parentheses (standalone duration)
        
        if is_duration and not has_separator and not has_parentheses:
            # Standalone duration like "10 years" - remove
            return False
        
        if not has_month and not has_year and not has_separator:
            # Doesn't look like a date at all - remove
            return False
            
        return True
    
    # Apply filtering
    filtered_matches = [match for match in cleaned_matches if should_keep_date(match)]
    
    seen, unique_matches = set(), []
    for match in filtered_matches:
        norm = match.lower().strip()
        if norm not in seen:
            seen.add(norm)
            unique_matches.append(match)
    
    return unique_matches


# ==================== POSITION MATCHING FUNCTIONS ====================

def build_position_matcher(position_list: List[str]) -> Dict:
    """
    Build an efficient position matcher using case-insensitive exact matching.
    Returns a dictionary with normalized positions as keys.
    """
    position_map = {}
    for pos in position_list:
        # Normalize: lowercase and collapse whitespace
        normalized = ' '.join(pos.lower().split())
        if normalized:
            position_map[normalized] = pos  # Keep original form
    
    return position_map


def find_unmatched_positions(
    text: str,
    ner_companies: List[str],
    ner_positions: List[str],
    ner_dates: List[str],
    position_matcher: Dict
) -> List[str]:
    """
    Find position abbreviations in text that weren't caught by NER model.
    Uses exact word/phrase matching on non-entity text.
    
    Args:
        text: Original raw text
        ner_companies: Companies extracted by NER
        ner_positions: Positions extracted by NER
        ner_dates: Dates extracted by regex/NER
        position_matcher: Dictionary of normalized position -> original position
    
    Returns:
        List of newly found positions
    """
    # Create a working copy of text
    masked_text = text
    
    # Mask all NER entities to avoid re-matching
    all_entities = ner_companies + ner_positions + ner_dates
    
    # Sort by length (longest first) to avoid partial masking
    all_entities.sort(key=len, reverse=True)
    
    for entity in all_entities:
        if entity and len(entity.strip()) > 0:
            # Case-insensitive replacement with placeholder
            pattern = re.escape(entity)
            masked_text = re.sub(pattern, ' [MASKED] ', masked_text, flags=re.IGNORECASE)
    
    # Normalize the masked text
    masked_text = ' '.join(masked_text.split())
    
    # Find position matches in the masked text
    found_positions = []
    found_positions_set = set()
    
    # Check each position in the matcher
    for normalized_pos, original_pos in position_matcher.items():
        # Create pattern for whole word matching
        # Use word boundaries for simple words, flexible for complex ones
        pattern = r'\b' + re.escape(normalized_pos) + r'\b'
        
        if re.search(pattern, masked_text.lower()):
            # Avoid duplicates (case-insensitive)
            if normalized_pos not in found_positions_set:
                found_positions.append(original_pos)
                found_positions_set.add(normalized_pos)
    
    return found_positions

def match_locations_from_mongodb(
    text: str,
    location_entities: Set[str],
    mongodb_connection: str,
    mongodb_password: str
) -> List[str]:
    if not location_entities:
        return []
    
    try:
        conn_str = mongodb_connection.replace('<db_password>', mongodb_password)
        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        db = client['Experience']
        collection = db['Location']
        
        # All punctuation/bracket characters to strip from tokens
        STRIP_CHARS = ",.()[]{}\"'\u2018\u2019\u201c\u201d"
        
        text_tokens = text.split()
        matched_locations = []
        
        i = 0
        while i < len(text_tokens) - 1:
            
            # TOKEN 1: try bigram first, then unigram
            token1_bigram = None
            if i + 1 < len(text_tokens):
                token1_bigram = (
                    text_tokens[i].strip(STRIP_CHARS) + " " +
                    text_tokens[i+1].strip(STRIP_CHARS)
                ).lower()

            if token1_bigram and token1_bigram in location_entities:
                token1_match    = token1_bigram
                token1_original = (
                    text_tokens[i].strip(STRIP_CHARS) + " " +
                    text_tokens[i+1].strip(STRIP_CHARS)
                )
                token1_advance  = 2
            else:
                token1_match    = text_tokens[i].strip(STRIP_CHARS).lower()
                token1_original = text_tokens[i].strip(STRIP_CHARS)
                token1_advance  = 1

            if token1_match not in location_entities:
                i += 1
                continue

            # TOKEN 2: starts right after token1's tokens
            j = i + token1_advance

            if j >= len(text_tokens):
                i += 1
                continue

            token2_bigram = None
            if j + 1 < len(text_tokens):
                token2_bigram = (
                    text_tokens[j].strip(STRIP_CHARS) + " " +
                    text_tokens[j+1].strip(STRIP_CHARS)
                ).lower()

            if token2_bigram and token2_bigram in location_entities:
                token2_match    = token2_bigram
                token2_original = (
                    text_tokens[j].strip(STRIP_CHARS) + " " +
                    text_tokens[j+1].strip(STRIP_CHARS)
                )
                token2_advance  = 2
            elif text_tokens[j].strip(STRIP_CHARS).lower() in location_entities:
                token2_match    = text_tokens[j].strip(STRIP_CHARS).lower()
                token2_original = text_tokens[j].strip(STRIP_CHARS)
                token2_advance  = 1
            else:
                i += 1
                continue

            # MongoDB row check
            query = {
                '$and': [
                    {
                        '$or': [
                            {'name':         {'$regex': f'^{re.escape(token1_match)}$', '$options': 'i'}},
                            {'state_code':   {'$regex': f'^{re.escape(token1_match)}$', '$options': 'i'}},
                            {'state_name':   {'$regex': f'^{re.escape(token1_match)}$', '$options': 'i'}},
                            {'country_code': {'$regex': f'^{re.escape(token1_match)}$', '$options': 'i'}},
                            {'country_name': {'$regex': f'^{re.escape(token1_match)}$', '$options': 'i'}}
                        ]
                    },
                    {
                        '$or': [
                            {'name':         {'$regex': f'^{re.escape(token2_match)}$', '$options': 'i'}},
                            {'state_code':   {'$regex': f'^{re.escape(token2_match)}$', '$options': 'i'}},
                            {'state_name':   {'$regex': f'^{re.escape(token2_match)}$', '$options': 'i'}},
                            {'country_code': {'$regex': f'^{re.escape(token2_match)}$', '$options': 'i'}},
                            {'country_name': {'$regex': f'^{re.escape(token2_match)}$', '$options': 'i'}}
                        ]
                    }
                ]
            }

            common_rows = list(collection.find(query))

            if common_rows:
                matched_locations.append(
                    token1_original + ", " + token2_original
                )
                i += token1_advance + token2_advance
                continue

            i += 1

        client.close()
        return matched_locations

    except Exception as e:
        print(f"Error matching locations from MongoDB: {e}")
        return []
# ==================== LSTM CLASSIFIER FOR DESC/NOT DESC ====================

class AttentionPooling(nn.Module):
    """Self-attention pooling layer"""
    
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output, attention_mask):
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e4)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(attention_weights * lstm_output, dim=1)
        return pooled


class BERTBiLSTMClassifier(nn.Module):
    """BERT + BiLSTM + Attention classifier with auxiliary features"""
    
    def __init__(self, config):
        super(BERTBiLSTMClassifier, self).__init__()
        self.config = config
        
        # BERT encoder
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(config["bert_model"])
        
        # Freeze early BERT layers
        if config.get("freeze_bert_layers", 0) > 0:
            freeze_layers = config["freeze_bert_layers"]
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in list(self.bert.encoder.layer)[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config["bert_hidden_size"],
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_num_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=config["lstm_dropout"] if config["lstm_num_layers"] > 1 else 0
        )
        
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(config["lstm_hidden_size"] * 2)
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(config["lstm_hidden_size"] * 2)
        
        # Classifier with batch normalization (input: lstm_output + 6 aux features)
        self.classifier = nn.Sequential(
            nn.Linear(config["lstm_hidden_size"] * 2 + 6, config["classifier_hidden_size"]),
            nn.BatchNorm1d(config["classifier_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config["classifier_dropout"]),
            nn.Linear(config["classifier_hidden_size"], config["num_classes"])
        )
    
    def forward(self, input_ids, attention_mask, aux_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.layer_norm(lstm_output)
        pooled = self.attention_pooling(lstm_output, attention_mask)
        combined = torch.cat([pooled, aux_features], dim=1)
        logits = self.classifier(combined)
        return logits

def extract_auxiliary_features(text: str) -> dict:
    """
    Extract auxiliary features from text using spaCy.
    These 6 features are required by the LSTM model.
    """
    doc = nlp(text)
    
    # Feature 1: has_verb - Check if any token is a verb
    has_verb = any(token.pos_ == "VERB" for token in doc)
    
    # Feature 2: proper_noun_ratio
    proper_nouns = sum(1 for token in doc if token.pos_ == "PROPN")
    proper_noun_ratio = proper_nouns / len(doc) if len(doc) > 0 else 0
    
    # Feature 3: regex_degree - Check for degree patterns
    degree_patterns = [
        r'\b(bachelor|master|phd|ph\.d|doctorate|associate|diploma|certificate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?tech|m\.?tech|mba|bba|b\.?com|m\.?com|b\.?e\.?|m\.?e\.?)\b',
    ]
    regex_degree = any(re.search(p, text.lower()) for p in degree_patterns)
    
    # Feature 4: has_date - Check for date patterns
    has_date = any(ent.label_ == "DATE" for ent in doc.ents)
    if not has_date:
        date_patterns = [
            r'\b(19|20)\d{2}\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
        ]
        has_date = any(re.search(p, text.lower()) for p in date_patterns)
    
    # Feature 5: has_company - Check for ORG entities
    has_company = any(ent.label_ == "ORG" for ent in doc.ents)
    
    # Feature 6: has_university - Check for university keywords
    university_keywords = ['university', 'college', 'institute', 'school', 'academy', 'icai']
    has_university = any(kw in text.lower() for kw in university_keywords)
    
    return {
        'has_verb': int(has_verb),
        'proper_noun_ratio': proper_noun_ratio,
        'regex_degree': int(regex_degree),
        'has_date': int(has_date),
        'has_company': int(has_company),
        'has_university': int(has_university),
    }

class Config:
    """Dummy Config class to allow checkpoint loading"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_lstm_classifier(model_path: str, config_path: str, device: torch.device):
    """Load the trained LSTM classifier for DESC/NOT DESC verification"""
    try:
        if not os.path.exists(model_path):
            print(f"Warning: LSTM model not found at {model_path}")
            return None, None, None
        
        if not os.path.exists(config_path):
            print(f"Warning: LSTM config file not found at {config_path}")
            return None, None, None
        
        # Config matching the model architecture
        config = {
            "bert_model": "jjzha/jobbert-base-cased",
            "bert_hidden_size": 768,
            "lstm_hidden_size": 256,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.3,
            "classifier_hidden_size": 128,
            "classifier_dropout": 0.4,
            "num_classes": 2,
            "max_length": 64,
            "freeze_bert_layers": 10
        }
        
        print(f"📦 Loading LSTM DESC classifier...")
        print(f"   Model: {config['bert_model']}")
        print(f"   Architecture: BERT + BiLSTM + Attention + Auxiliary Features")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])

        # Initialize model
        model = BERTBiLSTMClassifier(config)

        # Fix for Gunicorn: register Config class so pickle can find it
        import sys
        import types
        main_module = sys.modules.get('__main__', types.ModuleType('__main__'))
        setattr(main_module, 'Config', Config)
        sys.modules['__main__'] = main_module

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        
        print(f"✅ LSTM DESC classifier loaded successfully")
        return model, tokenizer, config
        
    except Exception as e:
        print(f"Error loading LSTM classifier: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    

# ==================== OPTIMIZED BATCH PROCESSING FUNCTIONS ====================

import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
import threading

class LineDataset(Dataset):
    """Dataset for batching lines for LSTM processing"""
    def __init__(self, lines, tokenizer, max_length):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        # Pre-tokenize all lines in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.encodings = list(executor.map(self._encode_line, lines))
    
    def _encode_line(self, line):
        if not line.strip():
            return None
        return self.tokenizer(
            line.strip(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        return idx, self.encodings[idx]


def collate_lstm_batch(batch):
    """Custom collate function for LSTM batching"""
    indices = []
    input_ids_list = []
    attention_mask_list = []
    valid_mask = []
    
    for idx, encoding in batch:
        indices.append(idx)
        if encoding is not None:
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    if not any(valid_mask):
        return indices, None, None, valid_mask
    
    # Only stack valid encodings
    valid_input_ids = [input_ids_list[i] for i, v in enumerate(valid_mask) if v]
    valid_attention_masks = [attention_mask_list[i] for i, v in enumerate(valid_mask) if v]
    
    if valid_input_ids:
        input_ids = torch.stack(valid_input_ids)
        attention_mask = torch.stack(valid_attention_masks)
    else:
        input_ids = None
        attention_mask = None
    
    return indices, input_ids, attention_mask, valid_mask    


# ==================== MODEL ARCHITECTURE (Keep as is) ====================

class DisentangledSelfAttention(nn.Module):
    def __init__(self, config, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = config['attention_head_size']
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query_proj = nn.Linear(hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size)
        self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size)
        self.pos_proj = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.output_proj = nn.Linear(self.all_head_size, hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query_proj(hidden_states))
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))
        pos_query_layer = self.transpose_for_scores(self.pos_query_proj(hidden_states))
        pos_key_layer = self.transpose_for_scores(self.pos_key_proj(position_embeddings))
        
        c2c_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        c2p_scores = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
        p2c_scores = torch.matmul(pos_query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = c2c_scores + c2p_scores + p2c_scores
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + (attention_mask * -10000.0)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        output = self.output_proj(context_layer)
        return output


class EnhancedDeBERTaEncoder(nn.Module):
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        self.disentangled_attention = DisentangledSelfAttention(config, self.hidden_size)
        self.max_position_embeddings = config['max_length']
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(config['dropout_rate'])
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, input_ids, attention_mask):
        base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = base_outputs.last_hidden_state
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        attention_output = self.disentangled_attention(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_output)
        hidden_states = hidden_states + position_embeddings
        
        return hidden_states


class TokenSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(hidden_states + output)
        return output


class BIOESTaggingHead(nn.Module):
    def __init__(self, config, hidden_size: int, num_labels: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(config['dropout_rate'])
        )
        
        self.use_bilstm_context = config.get('use_bilstm_context', True)
        if self.use_bilstm_context:
            context_lstm_hidden = config.get('context_lstm_hidden', 256)
            self.bilstm_context = nn.LSTM(
                input_size=hidden_size,
                hidden_size=context_lstm_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=True
            )
            self.bilstm_norm = nn.LayerNorm(context_lstm_hidden * 2)
            self.bilstm_dropout = nn.Dropout(config.get('context_dropout', 0.15))
            context_output_size = context_lstm_hidden * 2
        else:
            context_output_size = hidden_size
        
        self.use_token_attention = config.get('use_token_attention', True)
        if self.use_token_attention:
            context_attention_heads = config.get('context_attention_heads', 8)
            self.token_attention = TokenSelfAttention(
                hidden_size=context_output_size,
                num_heads=context_attention_heads,
                dropout=config.get('context_dropout', 0.15)
            )
            self.fusion = nn.Sequential(
                nn.Linear(context_output_size * 2, context_output_size),
                nn.LayerNorm(context_output_size),
                nn.GELU(),
                nn.Dropout(config.get('context_dropout', 0.15))
            )
            final_context_size = context_output_size
        else:
            final_context_size = context_output_size
        
        if self.use_bilstm_context or self.use_token_attention:
            self.context_projection = nn.Linear(final_context_size, hidden_size)
        
        self.use_lstm = config.get('use_lstm', True)
        if self.use_lstm:
            lstm_hidden_size = config.get('lstm_hidden_size', 512)
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=config.get('lstm_num_layers', 2),
                batch_first=True,
                dropout=config.get('lstm_dropout', 0.1) if config.get('lstm_num_layers', 2) > 1 else 0,
                bidirectional=False
            )
            self.lstm_layer_norm = nn.LayerNorm(lstm_hidden_size)
            classifier_input_size = lstm_hidden_size
        else:
            classifier_input_size = hidden_size
        
        self.output_projection = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.LayerNorm(classifier_input_size // 2),
            nn.GELU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(classifier_input_size // 2, num_labels)
        )
        self.dropout = nn.Dropout(config['dropout_rate'])
        
    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.pre_classifier(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if self.use_bilstm_context:
            bilstm_out, _ = self.bilstm_context(hidden_states)
            bilstm_out = self.bilstm_norm(bilstm_out)
            bilstm_out = self.bilstm_dropout(bilstm_out)
            
            if self.use_token_attention:
                attention_out = self.token_attention(bilstm_out)
                fused = torch.cat([bilstm_out, attention_out], dim=-1)
                hidden_states = self.fusion(fused)
            else:
                hidden_states = bilstm_out
            hidden_states = self.context_projection(hidden_states)
        elif self.use_token_attention:
            attention_out = self.token_attention(hidden_states)
            fused = torch.cat([hidden_states, attention_out], dim=-1)
            hidden_states = self.fusion(fused)
            hidden_states = self.context_projection(hidden_states)
        
        if self.use_lstm:
            lstm_output, _ = self.lstm(hidden_states)
            hidden_states = self.lstm_layer_norm(lstm_output)
        
        logits = self.output_projection(hidden_states)
        return logits


class ProductionDeBERTaNER(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.base_config = AutoConfig.from_pretrained(config['model_name'])
        self.base_model = AutoModel.from_pretrained(config['model_name'], config=self.base_config)
        self.encoder = EnhancedDeBERTaEncoder(config, self.base_model)
        self.tagging_head = BIOESTaggingHead(config, self.base_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.tagging_head(hidden_states)
        return logits


# ==================== TESTING CLASS ====================

class NERTester:
    def __init__(self, model_path: str, device: str = 'auto', use_regex_dates: bool = True, 
             use_position_matching: bool = True, mongodb_connection: str = None,
             mongodb_password: str = None):
        """
        Initialize the NER tester
        """
        self.model_path = Path(model_path)
        self.use_regex_dates = use_regex_dates
        self.use_position_matching = use_position_matching
        self.mongodb_connection = mongodb_connection
        self.mongodb_password = mongodb_password

        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Regex-based date extraction: {'ENABLED' if use_regex_dates else 'DISABLED'}")
        print(f"Position abbreviation matching: {'ENABLED' if use_position_matching else 'DISABLED'}")

        # Load LSTM classifier
        self.lstm_model = None
        self.lstm_tokenizer = None
        self.lstm_config = None

        lstm_base_path = Path(r"C:\Users\bhanu\OneDrive\Desktop\ai mock interview\desc_neev")
        lstm_model_path = lstm_base_path / "best_model.pt"
        lstm_final_model_path = lstm_base_path / "final_model.pt"

        if lstm_model_path.exists() and lstm_final_model_path.exists():
            print(f"✅ Both model files found, attempting load...")
            try:
                self.lstm_model, self.lstm_tokenizer, self.lstm_config = load_lstm_classifier(
                    str(lstm_model_path), 
                    str(lstm_final_model_path), 
                    self.device
                )
            except Exception as e:
                print(f"❌ LSTM load exception: {e}")
                import traceback
                traceback.print_exc()
            
            if self.lstm_model:
                print("✅ LSTM DESC verification: ENABLED")
            else:
                print("❌ LSTM loaded None — check load_lstm_classifier internals")
        else:
            print(f"❌ File check failed:")
            print(f"   best_model.pt exists: {lstm_model_path.exists()}")
            print(f"   final_model.pt exists: {lstm_final_model_path.exists()}")

        # ✅ NEW: Cache MongoDB data ONCE at initialization
        print("\n📦 Loading MongoDB data (one-time cache)...")
        self.company_set = set()
        self.company_list = []
        self.location_entities = set()
        self.position_matcher = {}

        if mongodb_connection and mongodb_password:
            import time
            start_time = time.time()

            # Load companies
            self.company_set, self.company_list = load_companies_from_mongodb(
                mongodb_connection, 
                mongodb_password
            )
            if self.company_set:
                print(f"✅ MongoDB company search: ENABLED")

            # Load locations
            self.location_entities = load_locations_from_mongodb(
                mongodb_connection, 
                mongodb_password
            )
            if self.location_entities:
                print(f"✅ MongoDB location search: ENABLED")

            # Load positions
            if self.use_position_matching:
                position_list = load_positions_from_mongodb(mongodb_connection, mongodb_password)
                self.position_matcher = build_position_matcher(position_list)
                print(f"✅ Position matcher loaded with {len(self.position_matcher)} patterns from MongoDB")

            end_time = time.time()
            print(f"✅ MongoDB cache loaded in {end_time - start_time:.2f}s")
        else:
            print("⚠️  MongoDB connection not provided - external search disabled")

        # Updated sliding window parameters
        self.max_seq_length = 512
        self.window_size = 300
        self.overlap_size = 50

        # Load configuration and label mappings
        self._load_config()

        # Load tokenizer
        print(f"Loading tokenizer from: {self.config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            use_fast=True
        )

        # Load model
        self._load_model()

        print(f"Model loaded successfully from {model_path}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Sliding window: size={self.window_size}, overlap={self.overlap_size}")
        
    def _verify_and_tag_lines_with_lstm_batched(self, text: str, batch_size: int = 64) -> Dict:
        text = normalize_text_newlines(text)

        if '\n' in text and text.count('\n') > 1:
            text_lines = text.split('\n')
            newline_char = '\n'
        elif '\\n' in text:
            text_lines = text.split('\\n')
            newline_char = '\n'
        else:
            text_lines = [text]
            newline_char = '\n'

        num_lines = len(text_lines)

        print(f"\n{'='*80}")
        print("LSTM DESC VERIFICATION - SENTENCE-LEVEL CLASSIFICATION")
        print(f"{'='*80}")
        print(f"Total raw lines: {num_lines}")

        if self.lstm_model is None or self.lstm_tokenizer is None:
            print("\n⚠ LSTM model not loaded - treating all lines as NOT DESC")
            return {
                'tagged_lines': [
                    {'line_num': i+1, 'text': line, 'tag': 'NOT DESC', 'probability': 0.0}
                    for i, line in enumerate(text_lines)
                ],
                'desc_lines': [],
                'not_desc_lines': text_lines,
                'filtered_text': text,
                'line_tags': {i+1: 'NOT DESC' for i in range(num_lines)}
            }

        def is_non_classifiable(content: str) -> bool:
            s = content.strip()
            if not s:
                return True
            if re.match(r'^\[.+\]$', s):
                return True
            if re.match(r'^[-=*_|]{2,}$', s):
                return True
            if re.match(r'^\d+\.?$', s):
                return True
            if re.match(r'^page\s+\d+\s+of\s+\d+$', s, re.IGNORECASE):
                return True
            if re.match(r'^\d+\s+of\s+\d+$', s, re.IGNORECASE):
                return True
            if re.match(r'^total\s+items\s*:', s, re.IGNORECASE):
                return True
            if re.match(r'^unassigned\s*:', s, re.IGNORECASE):
                return True
            if re.match(r'^(https?://|www\.)\S+$', s, re.IGNORECASE):
                return True
            if re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', s):
                return True
            if re.match(r'^\([A-Za-z0-9\s]{1,20}\)$', s):
                return True
            if re.match(r'^-{3,}$', s):
                return True
            if re.match(r'^www\.linkedin\.com', s, re.IGNORECASE):
                return True
            if re.match(r'^[\d\s\+\-\(\)\.]+$', s) and len(s) < 20:
                return True
            if re.match(r'^\d+$', s):
                return True
            if re.match(r'^[A-Za-z\s]+:\s*[\d\.]+%?$', s) and len(s.split()) <= 4:
                return True
            return False

        # =========================================================
        # STEP 1: Group consecutive classifiable lines into blocks
        # =========================================================
        sentence_units = []
        auto_not_desc = set()

        non_classifiable_indices = set()
        for idx, line in enumerate(text_lines):
            if not line.strip() or is_non_classifiable(line.strip()):
                non_classifiable_indices.add(idx)
                auto_not_desc.add(idx)

        blocks = []
        current_block_indices = []
        current_block_text = []

        for idx, line in enumerate(text_lines):
            if idx in non_classifiable_indices:
                if current_block_indices:
                    blocks.append((current_block_indices, '\t'.join(current_block_text)))
                    current_block_indices = []
                    current_block_text = []
            else:
                current_block_indices.append(idx)
                current_block_text.append(line.strip())

        if current_block_indices:
            blocks.append((current_block_indices, '\t'.join(current_block_text)))

        for block_indices, block_text in blocks:
            sentences = re.split(r'(?<=[.!?])\s+|\t|(?<=\S)[ ]{8,}(?=\S)', block_text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                for idx in block_indices:
                    auto_not_desc.add(idx)
                continue

            for sent in sentences:
                best_idx = block_indices[0]
                best_overlap = -1
                sent_words = set(sent.lower().split())

                for idx in block_indices:
                    line_words = set(text_lines[idx].lower().split())
                    overlap = len(sent_words & line_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = idx

                sentence_units.append((best_idx, sent))

        print(f"Classifiable blocks: {len(blocks)}")
        print(f"Total sentence units to classify: {len(sentence_units)}")
        print(f"Auto NOT DESC (structural/empty): {len(auto_not_desc)}")

        if not sentence_units:
            return {
                'tagged_lines': [
                    {'line_num': i+1, 'text': line, 'tag': 'NOT DESC', 'probability': 1.0}
                    for i, line in enumerate(text_lines)
                ],
                'desc_lines': [],
                'not_desc_lines': text_lines,
                'filtered_text': newline_char.join(text_lines),
                'line_tags': {i+1: 'NOT DESC' for i in range(num_lines)}
            }

        # =========================================================
        # STEP 2: Extract aux features
        # =========================================================
        print(f"  Extracting auxiliary features for {len(sentence_units)} sentence units...")
        aux_features_list = []
        for orig_idx, sent in sentence_units:
            aux_dict = extract_auxiliary_features(sent)
            aux_tensor = torch.FloatTensor([
                aux_dict['has_verb'],
                aux_dict['proper_noun_ratio'],
                aux_dict['regex_degree'],
                aux_dict['has_date'],
                aux_dict['has_company'],
                aux_dict['has_university'],
            ])
            aux_features_list.append(aux_tensor)

        # =========================================================
        # STEP 3: Tokenize
        # =========================================================
        print(f"  Tokenizing {len(sentence_units)} sentence units...")
        encodings = []
        for orig_idx, sent in sentence_units:
            try:
                enc = self.lstm_tokenizer(
                    sent,
                    truncation=True,
                    padding='max_length',
                    max_length=self.lstm_config["max_length"],
                    return_tensors='pt'
                )
                encodings.append(enc)
            except Exception:
                encodings.append(None)

        # =========================================================
        # STEP 4: Batch inference
        # =========================================================
        num_items = len(sentence_units)
        num_batches = (num_items + batch_size - 1) // batch_size
        print(f"  Running {num_batches} batch(es) of size {batch_size}...")

        raw_predictions = [None] * num_items
        self.lstm_model.eval()

        with torch.inference_mode():
            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, num_items)

                b_ids, b_masks, b_aux, b_local_idx = [], [], [], []
                for local_idx in range(b_start, b_end):
                    enc = encodings[local_idx]
                    if enc is not None:
                        b_ids.append(enc['input_ids'].squeeze(0))
                        b_masks.append(enc['attention_mask'].squeeze(0))
                        b_aux.append(aux_features_list[local_idx])
                        b_local_idx.append(local_idx)

                if not b_ids:
                    continue

                ids_t   = torch.stack(b_ids).to(self.device)
                masks_t = torch.stack(b_masks).to(self.device)
                aux_t   = torch.stack(b_aux).to(self.device)

                logits = self.lstm_model(ids_t, masks_t, aux_t)
                probs  = torch.softmax(logits, dim=-1)
                preds  = torch.argmax(probs, dim=-1)

                for i, local_idx in enumerate(b_local_idx):
                    raw_predictions[local_idx] = {
                        'prediction':    preds[i].item(),
                        'prob_desc':     probs[i][0].item(),
                        'prob_not_desc': probs[i][1].item()
                    }

        # =========================================================
        # STEP 5: Map sentence predictions back to original lines.
        # A line is DESC if ANY of its sentences is DESC.
        # =========================================================
        line_tag_map = {idx: ('NOT DESC', 1.0) for idx in range(num_lines)}

        for i, (orig_idx, sent) in enumerate(sentence_units):
            pred = raw_predictions[i]
            if pred is None:
                continue

            current_tag, current_prob = line_tag_map[orig_idx]

            if pred['prediction'] == 0:  # DESC wins
                if current_tag == 'NOT DESC' or pred['prob_desc'] > current_prob:
                    line_tag_map[orig_idx] = ('DESC', pred['prob_desc'])
            else:
                if current_tag == 'NOT DESC' and pred['prob_not_desc'] > current_prob:
                    line_tag_map[orig_idx] = ('NOT DESC', pred['prob_not_desc'])

        # =========================================================
        # STEP 5b: Tag orphan fragment lines.
        # Multi-line sentences get attributed to one line via word
        # overlap, leaving overflow lines (e.g. "trends.") as NOT DESC.
        # Fix: if a short line's text is a substring of any DESC
        # sentence, tag it DESC too.
        # =========================================================
        desc_sentences_combined = ' '.join(
            sent.lower().strip()
            for i, (orig_idx, sent) in enumerate(sentence_units)
            if raw_predictions[i] is not None and raw_predictions[i]['prediction'] == 0
        )

        for idx, line in enumerate(text_lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue
            if line_tag_map[idx][0] == 'DESC':
                continue  # Already tagged DESC, skip

            line_lower = line_stripped.lower()
            word_count = len(line_lower.split())

            # Only consider short lines (fragments) — full entity lines won't match
            if word_count <= 5 and line_lower in desc_sentences_combined:
                line_tag_map[idx] = ('DESC', 0.95)
                print(f"   🔍 Orphan line {idx+1} re-tagged DESC (fragment): '{line_stripped}'")

        # =========================================================
        # STEP 6: Build output — tagged_lines at SENTENCE level
        # =========================================================
        desc_lines     = []
        not_desc_lines = []
        filtered_lines = []
        line_tags      = {}
        desc_count     = 0
        not_desc_count = 0

        # Build sentence-level tag map
        sentence_tag_list = []
        for i, (orig_idx, sent) in enumerate(sentence_units):
            pred = raw_predictions[i]
            if pred is None:
                sentence_tag_list.append(('NOT DESC', 1.0, sent))
            elif pred['prediction'] == 0:
                sentence_tag_list.append(('DESC', pred['prob_desc'], sent))
            else:
                sentence_tag_list.append(('NOT DESC', pred['prob_not_desc'], sent))

        for idx, line in enumerate(text_lines):
            tag, prob = line_tag_map[idx]
            line_tags[idx + 1] = tag

            if tag == 'DESC':
                desc_lines.append(line)
                filtered_lines.append('- - - -')
                desc_count += 1
            else:
                not_desc_lines.append(line)
                filtered_lines.append(line)
                not_desc_count += 1

        tagged_lines = []
        for i, (tag, prob, sent) in enumerate(sentence_tag_list):
            tagged_lines.append({
                'line_num':    i + 1,
                'text':        sent,
                'tag':         tag,
                'probability': prob
            })

        filtered_text = newline_char.join(filtered_lines)

        print(f"\n{'='*80}")
        print(f"DESC TAGGING SUMMARY:")
        print(f"  Total lines:     {num_lines}")
        print(f"  Sentence units:  {len(sentence_units)}")
        print(f"  DESC (blanked):  {desc_count}")
        print(f"  NOT DESC (kept): {not_desc_count}")
        print(f"{'='*80}\n")

        return {
            'tagged_lines':   tagged_lines,
            'desc_lines':     desc_lines,
            'not_desc_lines': not_desc_lines,
            'filtered_text':  filtered_text,
            'line_tags':      line_tags
        }
    
    def _load_config(self):
        """Load model configuration and label mappings"""
        with open(self.model_path / "label_mappings.pkl", 'rb') as f:
            label_mappings = pickle.load(f)
        
        self.label_to_id = label_mappings['label_to_id']
        self.id_to_label = label_mappings['id_to_label']
        self.num_labels = label_mappings['num_labels']
        
        config_path = self.model_path / "model_config.json"
        if not config_path.exists():
            config_path = self.model_path / "config.json"
        
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        if 'config' in loaded_config:
            self.config = loaded_config['config']
        else:
            self.config = loaded_config
    
    def _load_model(self):
        """Load the trained model"""
        self.model = ProductionDeBERTaNER(self.config, self.num_labels)
        
        state_dict = torch.load(
            self.model_path / "model_state.pt",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict,strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _verify_line_with_lstm(self, line_text: str) -> bool:
        """
        Verify if a line is DESC or NOT DESC using LSTM classifier.
        Returns True if NOT DESC (valid extraction), False if DESC (reject).
        """
        if self.lstm_model is None or self.lstm_tokenizer is None:
            return True  # Accept by default if LSTM not loaded

        if not line_text.strip():
            return True  # Accept empty lines

        try:
            # Tokenize
            encoding = self.lstm_tokenizer(
                line_text.strip(),
                truncation=True,
                padding='max_length',
                max_length=self.lstm_config["max_length"],
                return_tensors='pt'
            )

            # Extract auxiliary features (now returns all zeros)
            aux_dict = extract_auxiliary_features(line_text)
            aux_features = torch.FloatTensor([[
                aux_dict['has_verb'],
                aux_dict['proper_noun_ratio'],
                aux_dict['regex_degree'],
                aux_dict['has_date'],
                aux_dict['has_company'],
                aux_dict['has_university'],
            ]])

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            aux_features = aux_features.to(self.device)

            with torch.no_grad():
                logits = self.lstm_model(input_ids, attention_mask, aux_features)
                probabilities = torch.softmax(logits, dim=-1)

                # Model has 2 classes
                # Class 0: Description (reject)
                # Class 1: Non-Description (accept)
                prediction = torch.argmax(probabilities, dim=-1).item()

                # Return True if Non-Description (class 1)
                is_valid = (prediction == 1)

            return is_valid

        except Exception as e:
            print(f"Warning: LSTM verification failed for line: {e}")
            return True  # Accept by default on error
    
    def _create_fixed_windows(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Create fixed-size windows of 300 tokens with 50 token overlap.
        Ensures complete coverage of the text.
        """
        window_size = 300
        overlap_size = 50
        stride = window_size - overlap_size  # 250 tokens stride
        
        windows = []
        start = 0
        
        while start < len(tokens):
            end = min(start + window_size, len(tokens))
            windows.append((start, end))
            
            # If we've reached the end, stop
            if end >= len(tokens):
                break
            
            # Move forward by stride (250 tokens)
            start += stride
            
            # Ensure last window captures remaining tokens if needed
            if start < len(tokens) and (len(tokens) - start) < window_size:
                # Create a final window that includes all remaining tokens
                # This window might have more overlap with the previous one
                final_start = max(0, len(tokens) - window_size)
                if final_start != windows[-1][0]:  # Avoid duplicate windows
                    windows.append((final_start, len(tokens)))
                break
        
        return windows
    
    def _debug_entity_distances(self, tokens, companies, positions, dates):
        """
        Debug function to show actual token distances between entities.
        Updated to show date-centric proximity analysis.
        """
        def find_entity_position(entity_text):
            """Find token index of entity"""
            ent_tokens = entity_text.split()
            for i in range(len(tokens) - len(ent_tokens) + 1):
                match = True
                for j, ent_tok in enumerate(ent_tokens):
                    if tokens[i+j].lower() != ent_tok.lower():
                        match = False
                        break
                if match:
                    return i
            return None

        def calculate_distance(pos1, pos2):
            """Calculate absolute distance between two positions"""
            if pos1 is None or pos2 is None:
                return float('inf')
            return abs(pos1 - pos2)

        print(f"\n{'='*80}")
        print("ENTITY DISTANCE ANALYSIS (DATE-CENTRIC)")
        print(f"{'='*80}")

        # Analyze from each date's perspective
        for date in dates:
            date_pos = find_entity_position(date)
            if date_pos is None:
                print(f"\n’ Date '{date}' - NOT FOUND IN TOKENS")
                continue
            
            print(f"\n Date: '{date}' at token {date_pos}")

            # Find all companies with their distances
            company_distances = []
            for company in companies:
                comp_pos = find_entity_position(company)
                if comp_pos is not None:
                    dist = calculate_distance(date_pos, comp_pos)
                    company_distances.append((company, comp_pos, dist))

            if company_distances:
                # Sort by distance
                company_distances.sort(key=lambda x: x[2])
                print(f"\n   Companies (sorted by distance):")
                for comp, pos, dist in company_distances[:5]:  # Show top 5
                    status = " WITHIN WINDOW" if dist <= 15 else "¡ ¯¸  OUTSIDE WINDOW"
                    print(f"      [{dist:3d} tokens] {comp} {status}")

            # Find all positions with their distances
            position_distances = []
            for position in positions:
                pos_pos = find_entity_position(position)
                if pos_pos is not None:
                    dist = calculate_distance(date_pos, pos_pos)
                    position_distances.append((position, pos_pos, dist))

            if position_distances:
                # Sort by distance
                position_distances.sort(key=lambda x: x[2])
                print(f"\n   Positions (sorted by distance):")
                for pos, pos_idx, dist in position_distances[:5]:  # Show top 5
                    status = " WITHIN WINDOW" if dist <= 15 else "¡ ¯¸  OUTSIDE WINDOW"
                    print(f"      [{dist:3d} tokens] {pos} {status}")

        print(f"\n{'='*80}\n")
    
    
    def _form_triplets_by_date_pivot(self, tokens, text, companies, positions, dates, window_size=15):
        """
        Form triplets using DATE as primary pivot with strict proximity logic.
        FIXED: Finds closest position to date first, then closest company to position.
        Each date is used only ONCE (first occurrence only).
        """
        triplets = []
        triplet_id = 1
        used_triplet_keys = set()
        used_dates = set()

        def is_valid_entity(entity):
            if not entity or not entity.strip():
                return False
            clean_entity = re.sub(r'[^\w\s]', '', entity).strip()
            if len(clean_entity) < 2:
                return False
            if re.fullmatch(r'[\|\-\+\=\*\/\\\<\>\.\,\;\:\'\"\!@#$%^&\[\]]+', entity.strip()):
                return False
            junk_words = {'and', 'of', 'to', 'in','for' , 'with', 'on', 'at', 'by', 'as', 'the', '|', '-', '+', '='}
            if entity.lower().strip() in junk_words:
                return False
            return True

        def deduplicate_entities(entity_list):
            seen_normalized = set()
            result = []
            for entity in entity_list:
                if not is_valid_entity(entity):
                    continue
                normalized = re.sub(r'[^\w\s]', '', entity.lower().strip())
                # Only deduplicate if exact same string, keep duplicates with same meaning
                # but track by (normalized, original) so same text at different positions is kept
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    result.append(entity)
                else:
                    # Keep the cleaner version (no trailing punctuation)
                    existing_idx = next(i for i, e in enumerate(result) 
                                       if re.sub(r'[^\w\s]', '', e.lower().strip()) == normalized)
                    existing = result[existing_idx]
                    if len(entity) > len(existing):
                        result[existing_idx] = entity
            return result

        print(f"\n🔍 Deduplicating entities before triplet formation...")

        # Companies: deduplicate fully (same company shouldn't appear twice)
        companies = deduplicate_entities(companies)

        # Positions: only remove junk, keep ALL valid positions including duplicates
        # because same job title (e.g. "Salesforce Administrator") can appear multiple times
        positions_seen_text = set()
        positions_deduped = []
        for p in positions:
            if not is_valid_entity(p):
                continue
            # Keep every unique text — don't remove duplicates
            positions_deduped.append(p)
        positions = positions_deduped

        dates = [d for d in dates if is_valid_entity(d)]

        print(f"   After deduplication:")
        print(f"   Companies: {len(companies)}")
        print(f"   Positions: {len(positions)}")
        print(f"   Dates: {len(dates)}")

        if not dates:
            print(f"\n⚠️ No valid dates found after filtering - cannot form triplets")
            return [], set()

        def normalize_token(token):
            normalized = re.sub(r'[\|\-–—−\,;:\'\"`\'\'\/\\\[\]\(\)\{\}]', '', token.lower().strip())
            return normalized

        def create_collapsed_tokens(tokens, entity_list):
            collapsed = []
            entity_map = {}
        
            entity_spans = []
            for ent in entity_list:
                if not is_valid_entity(ent):
                    continue
                ent_tokens = ent.split()
                ent_normalized = [normalize_token(e) for e in ent_tokens]
                ent_normalized = [e for e in ent_normalized if e]
        
                if not ent_normalized:
                    continue
                
                # Find ALL occurrences of this entity in tokens
                for j in range(len(tokens) - len(ent_tokens) + 1):
                    text_normalized = [normalize_token(t) for t in tokens[j:j          +len(ent_tokens)]]
        
                    matched = False
                    if text_normalized == ent_normalized:
                        matched = True
                    else:
                        text_core = [t for t in text_normalized if t]
                        ent_core = [e for e in ent_normalized if e]
                        if len(text_core) >= 2 and len(ent_core) >= 2:
                            if text_core == ent_core:
                                matched = True
        
                    if matched:
                        entity_spans.append((j, j+len(ent_tokens)-1, ent))
        
            # Sort by position, then by length (longer match wins at same          position)
            entity_spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        
            # Remove TRULY overlapping spans only
            # Two spans overlap if they share token indices
            # Same entity at different positions should BOTH be kept
            non_overlapping = []
            occupied_indices = set()  # Track which token indices are already          consumed
        
            for start, end, ent in entity_spans:
                span_indices = set(range(start, end + 1))
                if not span_indices.intersection(occupied_indices):
                    # No overlap with any already-accepted span
                    non_overlapping.append((start, end, ent))
                    occupied_indices.update(span_indices)
                else:
                    print(f"   ⚠️  Skipping overlapping span: '{ent}' at           [{start}-{end}] "
                          f"(conflicts with already accepted span)")
        
            entity_spans = non_overlapping
        
            # Debug: show all accepted spans
            print(f"\n   📋 Entity spans accepted for collapse ({len           (entity_spans)}):")
            for start, end, ent in entity_spans[:20]:  # Show first 20
                print(f"      [{start}-{end}] '{ent}'")
            if len(entity_spans) > 20:
                print(f"      ... and {len(entity_spans) - 20} more")
        
            collapsed_idx = 0
            original_idx = 0
        
            while original_idx < len(tokens):
                matching_entity = None
                for start, end, entity_text in entity_spans:
                    if start == original_idx:
                        matching_entity = (start, end, entity_text)
                        break
                    
                if matching_entity:
                    start, end, entity_text = matching_entity
                    collapsed.append(f"[ENTITY:{entity_text}]")
                    entity_map[collapsed_idx] = (start, end, entity_text)
                    collapsed_idx += 1
                    original_idx = end + 1
                else:
                    collapsed.append(tokens[original_idx])
                    collapsed_idx += 1
                    original_idx += 1
        
            return collapsed, entity_map

        print(f"\n🔧 Creating collapsed token representation...")
        all_entities = companies + positions
        collapsed_tokens, entity_map = create_collapsed_tokens(tokens, all_entities)

        print(f"   Original tokens: {len(tokens)}")
        print(f"   Collapsed tokens: {len(collapsed_tokens)}")
        print(f"   Entity collapses: {len(entity_map)}")

        def find_spans_in_collapsed(entity_list, collapsed_tokens, entity_map):
            spans = []
            seen_keys = set()  # Track (collapsed_idx, entity_text) to avoid        true duplicates
        
            for collapsed_idx, (orig_start, orig_end, entity_text) in       entity_map.items():
                for target_entity in entity_list:
                    if not is_valid_entity(target_entity):
                        continue
                    
                    target_normalized = normalize_token(target_entity)
                    entity_normalized = normalize_token(entity_text)
        
                    matched = False
                    if target_normalized == entity_normalized:
                        matched = True
                    else:
                        # Also try multi-word normalization for compound        entities
                        target_words = [normalize_token(w) for w in         target_entity.split() if normalize_token(w)]
                        entity_words = [normalize_token(w) for w in entity_text.        split() if normalize_token(w)]
                        if target_words and entity_words and target_words ==        entity_words:
                            matched = True
        
                    if matched:
                        # Use (collapsed_idx, normalized_text) as key
                        # This allows same entity at DIFFERENT positions
                        # but prevents same entity at SAME position being added         twice
                        key = (collapsed_idx, entity_normalized)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            spans.append((collapsed_idx, collapsed_idx,         target_entity))
                            # DO NOT break — continue checking other target         entities
                            # in case multiple entities normalize to same text
                        break  # One target entity matched this collapsed       token, move on
                    
            # Debug: show all found spans
            print(f"\n   📋 Entity spans found in collapsed space:")
            for start, end, ent in spans:
                print(f"      [{start}-{end}] '{ent}'")
        
            return spans

        def calculate_distance_collapsed(start1, end1, start2, end2):
            if end1 < start2:
                return start2 - end1 - 1
            elif end2 < start1:
                return start1 - end2 - 1
            else:
                return 0

        def find_closest_entity_collapsed(target_start, target_end, entity_spans, max_distance):
            """Find closest entity in collapsed token space"""
            closest = None
            min_distance = float('inf')

            for e_start, e_end, entity in entity_spans:
                distance = calculate_distance_collapsed(target_start, target_end, e_start, e_end)
                if distance <= max_distance and distance < min_distance:
                    min_distance = distance
                    closest = (entity, distance, e_start, e_end)

            return closest

        def find_closest_company_to_position_collapsed(pos_start, pos_end, company_spans, max_distance):
            """Find company closest to a given position in collapsed space"""
            closest = None
            min_distance = float('inf')
            candidates = []

            for c_start, c_end, comp in company_spans:
                distance = calculate_distance_collapsed(pos_start, pos_end, c_start, c_end)
                if distance <= max_distance:
                    # Prefer longer company names (more specific)
                    length_bonus = len(comp.split()) * 0.1
                    adjusted_score = distance - length_bonus
                    candidates.append((comp, distance, c_start, c_end, adjusted_score))

            if candidates:
                candidates.sort(key=lambda x: x[4])
                best = candidates[0]
                closest = (best[0], best[1], best[2], best[3])

            return closest

        # Find entity spans in COLLAPSED space
        company_spans_collapsed = find_spans_in_collapsed(companies, collapsed_tokens, entity_map)
        position_spans_collapsed = find_spans_in_collapsed(positions, collapsed_tokens, entity_map)

        # Find ALL occurrences of each date in collapsed space
        print(f"\n🔍 Finding all date occurrences in collapsed space...")

        all_date_occurrences = []

        for date in dates:
            date_tokens = date.split()
            date_normalized = [normalize_token(dt) for dt in date_tokens]
            date_normalized = [d for d in date_normalized if d]

            if not date_normalized:
                continue
            
            for i in range(len(collapsed_tokens) - len(date_tokens) + 1):
                if collapsed_tokens[i].startswith('[ENTITY:'):
                    continue
                
                match_found = True
                matched_positions = []

                j = 0
                k = i

                while j < len(date_normalized) and k < len(collapsed_tokens):
                    if collapsed_tokens[k].startswith('[ENTITY:'):
                        k += 1
                        continue
                    
                    token_norm = normalize_token(collapsed_tokens[k])
                    if not token_norm:
                        k += 1
                        continue
                    
                    if token_norm == date_normalized[j]:
                        matched_positions.append(k)
                        j += 1
                        k += 1
                    else:
                        match_found = False
                        break
                    
                if match_found and j == len(date_normalized):
                    start_pos = matched_positions[0] if matched_positions else i
                    end_pos = matched_positions[-1] if matched_positions else i
                    all_date_occurrences.append((start_pos, end_pos, date))

        # ✅ FIXED TRIPLET LOGIC: Position-first pivot
        # Strategy: date → closest position → closest company to that position
        print(f"\n🔍 Processing dates (position-first pivot strategy)...")

        for d_start, d_end, date_text in all_date_occurrences:
        
            if date_text in used_dates:
                print(f"\n⏭️  Skipping duplicate date '{date_text}' at collapsed tokens [{d_start}-{d_end}]")
                continue
            
            print(f"\n🔍 Processing date '{date_text}' at collapsed tokens [{d_start}-{d_end}]")

            # ✅ STEP 1: Find closest POSITION to date
            closest_position = find_closest_entity_collapsed(d_start, d_end, position_spans_collapsed, window_size)

            if not closest_position:
                print(f"   ❌ No position found within {window_size} collapsed tokens of date")
                continue
            
            pos_text, pos_dist_to_date, pos_start, pos_end = closest_position
            print(f"   ✓ Closest position: '{pos_text}' at distance {pos_dist_to_date} collapsed tokens from date")

            # ✅ STEP 2: Find closest COMPANY to that position (within window_size)
            closest_company = find_closest_company_to_position_collapsed(
                pos_start, pos_end, company_spans_collapsed, window_size
            )

            if not closest_company:
                print(f"   ❌ No company found within {window_size} collapsed tokens of position")
                continue
            
            comp_text, comp_dist_to_pos, comp_start, comp_end = closest_company

            # Also verify company is reasonably close to date (2x window as soft check)
            comp_dist_to_date = calculate_distance_collapsed(d_start, d_end, comp_start, comp_end)
            print(f"   ✓ Closest company to position: '{comp_text}' at distance {comp_dist_to_pos} tokens from position")
            print(f"      Company distance to date: {comp_dist_to_date} collapsed tokens")

            # Check for duplicate triplet
            triplet_key = (pos_text.lower().strip(), comp_text.lower().strip(), 
                          pos_start, comp_start)

            if triplet_key in used_triplet_keys:
                print(f"   ⚠️ Skipping duplicate triplet (same entities at same positions)")
                continue
            
            used_triplet_keys.add(triplet_key)
            used_dates.add(date_text)

            print(f"\n   ✅ Creating triplet:")
            print(f"      Position: '{pos_text}'")
            print(f"      Company:  '{comp_text}'")
            print(f"      Date:     '{date_text}'")

            triplets.append({
                "id": triplet_id,
                "Position": pos_text,
                "Company": comp_text,
                "Year": date_text,
                "Status": "valid",
                "distances": {
                    "position_to_date": pos_dist_to_date,
                    "company_to_position": comp_dist_to_pos,
                    "company_to_date": comp_dist_to_date,
                }
            })
            triplet_id += 1

        print(f"\n✅ Triplet Formation Complete (Position-First Pivot):")
        print(f"   Total unique triplets formed: {len(triplets)}")
        print(f"   Total date occurrences found: {len(all_date_occurrences)}")
        print(f"   Dates used: {len(used_dates)}")
        print(f"   Dates skipped (duplicates): {len(all_date_occurrences) - len(used_dates)}")

        return triplets, used_dates


    def _form_duos_from_unused_dates(
    self,
    tokens: List[str],
    text: str,
    positions: List[str],
    dates: List[str],
    used_dates: Set[str],
    triplets: List[Dict],
    window_size: int = 10  # ✅ INCREASED from 10 to 30 tokens (~150 chars)
) -> List[Dict]:
        """
        Form DUOs from dates that weren't used in triplet formation.
        FIXED: Searches in ORIGINAL text to find dates embedded in descriptions.
        FIXED: Uses larger window size for better position detection.
        """

        used_duo_keys = set()

        def normalize_token(token):
            normalized = re.sub(r'[\|\-–—−\.,;:\'\"`\'\'\/\\\[\]\(\)\{\}]', '', token.lower().strip())
            return normalized

        def is_valid_entity(entity):
            if not entity or not entity.strip():
                return False
            clean_entity = re.sub(r'[^\w\s]', '', entity).strip()
            if len(clean_entity) < 2:
                return False
            return True

        def find_date_in_text(date_str: str, text: str) -> Optional[int]:
            """Find the character position of a date in the original text."""
            if not date_str:
                return None

            date_normalized = date_str.strip()

            # Try exact match first
            pos = text.lower().find(date_normalized.lower())
            if pos != -1:
                return pos

            # Try with flexible spacing
            date_pattern = re.escape(date_normalized)
            date_pattern = date_pattern.replace(r'\ ', r'[\s\-–—]*')

            match = re.search(date_pattern, text, re.IGNORECASE)
            if match:
                return match.start()

            return None

        def find_position_near_date(
            date_position: int, 
            positions: List[str], 
            text: str, 
            max_gap: int = 50  # Gap between entity boundary and date boundary
        ) -> Optional[Tuple[str, int, int]]:
            """
            Find a position entity near a date in the original text.
            UPDATED: Treats entire position as one unit.
            Measures gap from END of position to START of date (position before date)
            OR from END of date to START of position (date before position).
            Returns (position_text, position_char_index, gap) or None.
            """
            best_position = None
            best_gap = float('inf')
            best_pos_index = None

            # Find date end position for reverse direction check
            date_text_match = re.search(re.escape(text[date_position:date_position+20]), text)
            date_end_position = date_position + len(text[date_position:].split('\n')[0].strip())

            for pos in positions:
                if not is_valid_entity(pos):
                    continue
                
                pos_pattern = r'\b' + re.escape(pos.strip()) + r'\b'

                for match in re.finditer(pos_pattern, text, re.IGNORECASE):
                    pos_start = match.start()
                    pos_end = match.end()

                    # ✅ Direction 1: Position ends before date starts
                    # Gap = from END of position to START of date
                    gap_before = date_position - pos_end
                    if 0 < gap_before <= max_gap:
                        if gap_before < best_gap:
                            best_gap = gap_before
                            best_position = pos
                            best_pos_index = pos_start

                    # ✅ Direction 2: Date ends before position starts
                    # Gap = from END of date to START of position
                    gap_after = pos_start - date_end_position
                    if 0 < gap_after <= max_gap:
                        if gap_after < best_gap:
                            best_gap = gap_after
                            best_position = pos
                            best_pos_index = pos_start

            if best_position:
                return (best_position, best_pos_index, best_gap)
            return None

        def find_previous_triplet_company(date_char_position: int, all_entries: List[Dict], text: str) -> Optional[str]:
            """Find the company from the nearest PREVIOUS triplet."""
            previous_entries = []

            for entry in all_entries:
                entry_date = entry.get('Year', '')
                if not entry_date:
                    continue
                
                entry_date_pos = find_date_in_text(entry_date, text)

                if entry_date_pos is not None and entry_date_pos < date_char_position:
                    distance = date_char_position - entry_date_pos
                    previous_entries.append((distance, entry))

            if previous_entries:
                previous_entries.sort(key=lambda x: x[0])
                closest_entry = previous_entries[0][1]

                entry_type = "DUO" if str(closest_entry.get('id', '')).startswith('DUO_') else "TRIPLET"
                print(f"      🔍 Found previous {entry_type}: '{closest_entry.get('Company', 'N/A')}'")

                return closest_entry.get('Company')

            print(f"      ⚠️ No previous entry found before character position {date_char_position}")
            return None

        # Get unused dates
        unused_dates = [d for d in dates if d not in used_dates]

        if not unused_dates:
            print("\n📋 DUO Formation: No unused dates found")
            return []

        print(f"\n📋 DUO Formation Starting (Text-Based Search):")
        print(f"   Total dates: {len(dates)}")
        print(f"   Used in triplets: {len(used_dates)}")
        print(f"   Unused dates available: {len(unused_dates)}")
        print(f"   Window size: {window_size} tokens (gap after entity: 50 chars)")
        # ✅ DEBUG: Show which dates are unused
        print(f"\n   🔍 Unused dates:")
        for ud in unused_dates:
            print(f"      - '{ud}'")

        all_entries = triplets.copy()
        duos = []
        duo_id = 1
        processed_dates = set()

        for date in unused_dates:
            if date in processed_dates:
                continue
            
            # Find date in original text
            date_char_pos = find_date_in_text(date, text)

            if date_char_pos is None:
                print(f"\n   ⚠️ Date '{date}' not found in text, skipping")
                continue
            
            print(f"\n   📅 Processing unused date: '{date}' at character position {date_char_pos}")

            # Find closest position BEFORE this date
            position_result = find_position_near_date(
                date_position=date_char_pos,
                positions=positions,
                text=text,
                max_gap=50
            )

            if not position_result:
                print(f"      ❌ No position found within 50 characters before date")
                continue
            
            pos_text, pos_char_index, distance = position_result
            print(f"      ✅ Found position: '{pos_text}' at {distance} chars before date")

            # Create DUO key for duplicate detection
            duo_key = (pos_text.lower().strip(), date.lower().strip(), pos_char_index, date_char_pos)

            if duo_key in used_duo_keys:
                print(f"      ⚠️ Duplicate DUO detected, skipping")
                continue
            
            used_duo_keys.add(duo_key)

            # Find company from previous triplet
            previous_company = find_previous_triplet_company(date_char_pos, all_entries, text)

            if previous_company:
                company_text = previous_company
                company_source = "COPIED_FROM_PREVIOUS"
                print(f"   ✅ DUO {duo_id}: Position '{pos_text}' + Date '{date}'")
                print(f"      Company copied from previous entry: '{company_text}'")
            else:
                company_text = "N/A"
                company_source = "NOT_FOUND"
                print(f"   ⚠️ DUO {duo_id}: Position '{pos_text}' + Date '{date}'")
                print(f"      No previous entry found, Company set to: N/A")

            new_duo = {
                "id": f"DUO_{duo_id}",
                "Position": pos_text,  # ✅ strip "52." prefix
                "Company": company_text,
                "Year": date,
                "Status": "duo",
                "company_source": company_source,
                "distances": {
                    "position_to_date": distance
                }
            }

            duos.append(new_duo)
            all_entries.append(new_duo)
            duo_id += 1
            processed_dates.add(date)

        print(f"\n📋 DUO Formation Complete:")
        print(f"   Total DUOs formed: {len(duos)}")

        return duos

    
    def _filter_by_proximity(self, tokens, entities, window_size=15):
        """
        Keep Company, Position, and Date entities only if they co-occur 
        within +/- window_size tokens. Return triplets for display.
        """
        company_spans = []
        position_spans = []

        # Build spans for companies and positions
        for comp in entities['Company']:
            comp_tokens = comp.split()
            for i in range(len(tokens) - len(comp_tokens) + 1):
                if tokens[i:i+len(comp_tokens)] == comp_tokens:
                    company_spans.append((i, i+len(comp_tokens)-1, comp))
                    break

        for pos in entities['Position']:
            pos_tokens = pos.split()
            for i in range(len(tokens) - len(pos_tokens) + 1):
                if tokens[i:i+len(pos_tokens)] == pos_tokens:
                    position_spans.append((i, i+len(pos_tokens)-1, pos))
                    break

        triplets = []
        triplet_id = 1

        # Cross-check within window
        for c_start, c_end, comp in company_spans:
            for p_start, p_end, pos in position_spans:
                if abs(c_start - p_end) <= window_size or abs(p_start - c_end) <= window_size:
                    # Find a year (Date) near either comp or pos
                    matched_years = []
                    for date in entities['Date']:
                        date_tokens = date.split()
                        for i in range(len(tokens) - len(date_tokens) + 1):
                            if tokens[i:i+len(date_tokens)] == date_tokens:
                                if min(abs(i - c_start), abs(i - p_start)) <= window_size:
                                    matched_years.append(date)
                                    break
                    year_str = ', '.join(matched_years)
                    triplets.append({
                        'id': triplet_id,
                        'Position': pos,
                        'Company': comp,
                        'Year': year_str,
                        'Status': 'valid'
                    })
                    triplet_id += 1

        return triplets


    
    def _is_coherent_transition(self, prev_label: str, curr_label: str) -> bool:
        """
        Check if the transition from prev_label to curr_label is coherent
        in the BIOES tagging scheme.
        """
        if prev_label == 'O':
            return curr_label in ['O', '[SEP]'] or curr_label.startswith(('B-', 'S-'))
        
        if prev_label == '[SEP]':
            return True
        
        if '-' not in prev_label or '-' not in curr_label:
            return curr_label == 'O' or curr_label == '[SEP]'
        
        prev_prefix, prev_type = prev_label.split('-', 1)
        curr_prefix, curr_type = curr_label.split('-', 1)
        
        transitions = {
            'B': ['I', 'E'],  # B can be followed by I or E of same type
            'I': ['I', 'E'],  # I can be followed by I or E of same type
            'E': ['B', 'S', 'O', '[SEP]'],  # E ends entity
            'S': ['B', 'S', 'O', '[SEP]']   # S is standalone
        }
        
        if prev_prefix in transitions:
            if prev_prefix in ['B', 'I']:
                # Must continue with same entity type
                return (curr_prefix in transitions[prev_prefix] and 
                        prev_type == curr_type)
            else:
                # E and S can start new entity
                return curr_prefix in transitions[prev_prefix] or curr_label == 'O'
        
        return False
    
    def _merge_with_weighted_voting(
        self,
        all_labels: List[List[str]],
        all_probs: List[List[np.ndarray]],
        window_ranges: List[Tuple[int, int]],
        tokens: List[str]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Merge predictions using weighted voting, giving more weight to predictions
        from the middle of windows vs. edges (overlap regions).
        """
        total_length = len(tokens)
        
        # Initialize voting structure
        vote_accumulator = {}
        
        for window_idx, (labels, probs, (start, end)) in enumerate(
            zip(all_labels, all_probs, window_ranges)
        ):
            window_size = end - start
            
            for local_idx, (label, prob) in enumerate(zip(labels, probs)):
                global_idx = start + local_idx
                
                if global_idx >= total_length:
                    continue
                
                # Calculate weight based on position in window
                # Higher weight for middle tokens, lower for edge tokens
                distance_from_edge = min(local_idx, window_size - local_idx - 1)
                weight = 1.0
                
                if distance_from_edge < 25:  # Within 25 tokens of edge
                    weight = 0.5 + (distance_from_edge / 50.0)
                
                # Store weighted vote
                if global_idx not in vote_accumulator:
                    vote_accumulator[global_idx] = {}
                
                if label not in vote_accumulator[global_idx]:
                    vote_accumulator[global_idx][label] = {
                        'weight_sum': 0.0,
                        'conf_sum': 0.0,
                        'prob_sum': np.zeros_like(prob),
                        'count': 0
                    }
                
                conf = float(np.max(prob))
                vote_accumulator[global_idx][label]['weight_sum'] += weight
                vote_accumulator[global_idx][label]['conf_sum'] += conf * weight
                vote_accumulator[global_idx][label]['prob_sum'] += prob * weight
                vote_accumulator[global_idx][label]['count'] += 1
        
        # Resolve votes with entity coherence
        merged_labels = []
        merged_probs = []
        
        for idx in range(total_length):
            if idx in vote_accumulator:
                best_label = None
                best_score = -1
                best_prob = None
                
                for label, stats in vote_accumulator[idx].items():
                    # Calculate weighted score
                    score = stats['conf_sum'] / max(stats['weight_sum'], 0.001)
                    
                    # Boost score for entity continuation
                    if idx > 0 and merged_labels:
                        prev_label = merged_labels[-1]
                        if self._is_coherent_transition(prev_label, label):
                            score *= 1.3
                    
                    if score > best_score:
                        best_score = score
                        best_label = label
                        # Normalize probability distribution
                        best_prob = stats['prob_sum'] / max(stats['weight_sum'], 0.001)
                
                merged_labels.append(best_label)
                merged_probs.append(best_prob)
            else:
                merged_labels.append('O')
                o_probs = np.zeros(self.num_labels)
                o_id = self.label_to_id.get('O', 0)
                o_probs[o_id] = 1.0
                merged_probs.append(o_probs)
        
        return merged_labels, merged_probs
    
    def correct_bioes_sequences(
        self, 
        tokens: List[str], 
        labels: List[str], 
        confidences: List[float]
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """Correct invalid BIOES sequences"""
        corrected_labels = []
        corrected_confidences = []
        corrections = []
        
        current_entity_type = None
        
        for i, (token, label, conf) in enumerate(zip(tokens, labels, confidences)):
            
            if label == '[SEP]' or label == 'O':
                current_entity_type = None
                corrected_labels.append(label)
                corrected_confidences.append(conf)
                
            elif label.startswith('B-'):
                current_entity_type = label[2:]
                corrected_labels.append(label)
                corrected_confidences.append(conf)
                
            elif label.startswith('S-'):
                current_entity_type = None
                corrected_labels.append(label)
                corrected_confidences.append(conf)
                
            elif label.startswith('I-'):
                entity_type = label[2:]
                
                if current_entity_type == entity_type:
                    corrected_labels.append(label)
                    corrected_confidences.append(conf)
                else:
                    should_convert_to_b = False
                    
                    for j in range(i + 1, min(i + 10, len(labels))):
                        next_label = labels[j]
                        if next_label.startswith('I-') and next_label[2:] == entity_type:
                            should_convert_to_b = True
                            break
                        elif next_label.startswith('E-') and next_label[2:] == entity_type:
                            should_convert_to_b = True
                            break
                        elif next_label in ['O', '[SEP]'] or next_label.startswith('B-') or next_label.startswith('S-'):
                            break
                    
                    if should_convert_to_b:
                        corrected_label = f'B-{entity_type}'
                        corrected_labels.append(corrected_label)
                        corrected_confidences.append(conf * 0.9)
                        current_entity_type = entity_type
                        
                        corrections.append({
                            'position': i,
                            'token': token,
                            'original': label,
                            'corrected': corrected_label,
                            'reason': 'Invalid I- without B-: converted to B-'
                        })
                    else:
                        corrected_label = f'S-{entity_type}'
                        corrected_labels.append(corrected_label)
                        corrected_confidences.append(conf * 0.9)
                        current_entity_type = None
                        
                        corrections.append({
                            'position': i,
                            'token': token,
                            'original': label,
                            'corrected': corrected_label,
                            'reason': 'Invalid I- without B-: converted to S-'
                        })
                            
            elif label.startswith('E-'):
                entity_type = label[2:]
                
                if current_entity_type == entity_type:
                    corrected_labels.append(label)
                    corrected_confidences.append(conf)
                    current_entity_type = None
                else:
                    corrected_label = f'S-{entity_type}'
                    corrected_labels.append(corrected_label)
                    corrected_confidences.append(conf * 0.9)
                    current_entity_type = None
                    
                    corrections.append({
                        'position': i,
                        'token': token,
                        'original': label,
                        'corrected': corrected_label,
                        'reason': 'Invalid E- without B-/I-: converted to S-'
                    })
            else:
                corrected_labels.append('O')
                corrected_confidences.append(0.0)
                current_entity_type = None
        
        return corrected_labels, corrected_confidences, corrections
    
    def _process_windows_batched(
    self,
    tokens: List[str],
    window_ranges: List[Tuple[int, int]],
    batch_size: int = 4  # ✅ Optimal for NER (lower than LSTM)
) -> Tuple[List[List[str]], List[List[np.ndarray]]]:
        """
        OPTIMIZED: Process all windows in TRUE batches.
        NER uses larger models, so smaller batch size than LSTM.

        Args:
            tokens: Full token list
            window_ranges: List of (start, end) tuples for each window
            batch_size: Number of windows to process per batch (default: 4)

        Returns:
            Tuple of (all_labels, all_probs) for each window
        """
        num_windows = len(window_ranges)
        all_labels = [None] * num_windows
        all_probs = [None] * num_windows

        print(f"  Processing {num_windows} windows in batches of {batch_size}...")

        # ✅ OPTIMIZATION 1: Pre-encode all windows (no threading needed)
        print(f"  Pre-encoding windows...")

        window_encodings = []
        window_tokens_list = []

        for start, end in window_ranges:
            window_tokens = tokens[start:end]
            try:
                encoding = self.tokenizer(
                    window_tokens,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )
                window_encodings.append(encoding)
                window_tokens_list.append(window_tokens)
            except Exception as e:
                print(f"  Encoding error for window: {e}")
                window_encodings.append(None)
                window_tokens_list.append(window_tokens)

        # ✅ OPTIMIZATION 2: Group windows by length for efficient batching
        # Windows with same token length can be batched together
        length_groups = {}
        for idx, (encoding, window_tokens) in enumerate(zip(window_encodings, window_tokens_list)):
            if encoding is None:
                continue
            length = len(window_tokens)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(idx)

        print(f"  Grouped windows into {len(length_groups)} length buckets for efficient batching")

        self.model.eval()

        with torch.inference_mode():  # ✅ Faster than no_grad()
            # Process each length group separately
            for length, window_indices in length_groups.items():
                num_group_windows = len(window_indices)
                group_batches = (num_group_windows + batch_size - 1) // batch_size

                for batch_idx in range(group_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, num_group_windows)
                    batch_window_indices = window_indices[batch_start:batch_end]

                    # ✅ TRUE BATCHING: Stack multiple windows together
                    batch_input_ids = []
                    batch_attention_masks = []

                    for window_idx in batch_window_indices:
                        encoding = window_encodings[window_idx]
                        batch_input_ids.append(encoding['input_ids'].squeeze(0))
                        batch_attention_masks.append(encoding['attention_mask'].squeeze(0))

                    # Stack into single batch tensor
                    input_ids_batch = torch.stack(batch_input_ids).to(self.device)
                    attention_mask_batch = torch.stack(batch_attention_masks).to(self.device)

                    # ✅ SINGLE FORWARD PASS for entire batch
                    logits = self.model(input_ids_batch, attention_mask_batch)
                    probabilities = F.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)

                    # Process each window in the batch
                    for batch_pos, window_idx in enumerate(batch_window_indices):
                        window_tokens = window_tokens_list[window_idx]
                        encoding = window_encodings[window_idx]

                        word_ids = encoding.word_ids()

                        # Extract predictions for this window from batch
                        window_predictions = predictions[batch_pos]
                        window_probabilities = probabilities[batch_pos]

                        # Map predictions back to tokens
                        predicted_labels = []
                        predicted_probs = []

                        for word_idx in range(len(window_tokens)):
                            token_idx = None
                            for i, w_id in enumerate(word_ids):
                                if w_id == word_idx:
                                    token_idx = i
                                    break
                                
                            if token_idx is not None:
                                pred_id = window_predictions[token_idx].item()
                                pred_label = self.id_to_label.get(pred_id, 'O')
                                predicted_labels.append(pred_label)

                                token_probs = window_probabilities[token_idx].cpu().numpy()
                                predicted_probs.append(token_probs)
                            else:
                                predicted_labels.append('O')
                                o_probs = np.zeros(self.num_labels)
                                o_id = self.label_to_id.get('O', 0)
                                o_probs[o_id] = 1.0
                                predicted_probs.append(o_probs)

                        all_labels[window_idx] = predicted_labels
                        all_probs[window_idx] = predicted_probs

            # Handle failed encodings
            for window_idx, encoding in enumerate(window_encodings):
                if encoding is None and all_labels[window_idx] is None:
                    window_tokens = window_tokens_list[window_idx]
                    all_labels[window_idx] = ['O'] * len(window_tokens)
                    o_probs = np.zeros(self.num_labels)
                    o_id = self.label_to_id.get('O', 0)
                    o_probs[o_id] = 1.0
                    all_probs[window_idx] = [o_probs] * len(window_tokens)

        print(f"  ✓ All windows processed with true batching")

        return all_labels, all_probs
    
    def _post_process_entities(
        self, 
        tokens: List[str], 
        labels: List[str]
    ) -> List[str]:
        """Clean up entity boundaries after merging"""
        
        processed_labels = labels.copy()
        
        # Fix broken entities at window boundaries
        for i in range(len(labels)):
            # Fix orphaned I- or E- tags
            if labels[i].startswith('I-') or labels[i].startswith('E-'):
                entity_type = labels[i][2:]
                
                # Check if properly connected to previous
                if i > 0 and not (
                    labels[i-1].startswith(('B-', 'I-')) and 
                    entity_type in labels[i-1]
                ):
                    # Convert to B- or S- depending on next token
                    if i < len(labels) - 1 and labels[i+1].startswith('I-') and entity_type in labels[i+1]:
                        processed_labels[i] = f'B-{entity_type}'
                    else:
                        processed_labels[i] = f'S-{entity_type}'
            
            # Fix orphaned B- tags that should be S-
            elif labels[i].startswith('B-'):
                entity_type = labels[i][2:]
                # Check if followed by continuation
                if i < len(labels) - 1:
                    next_label = labels[i+1]
                    # If not followed by I- or E- of same type, convert to S-
                    if not ((next_label.startswith('I-') or next_label.startswith('E-')) and entity_type in next_label):
                        # Check if this is truly a standalone entity
                        if i == len(labels) - 1 or not (next_label.startswith(('I-', 'E-')) and entity_type in next_label):
                            processed_labels[i] = f'S-{entity_type}'
                else:
                    # Last token with B- should be S-
                    processed_labels[i] = f'S-{entity_type}'
        
        return processed_labels
    
    def _extract_entities_with_regex_dates(
    self,
    text: str,
    tokens: List[str],
    labels: List[str]
) -> Dict[str, List[str]]:
        """
        Extract entities from BIOES-tagged tokens WITHOUT LSTM verification.
        MODIFIED: 
        1. Mask company entities before extracting positions to prevent overlap.
        2. OPTIMIZED: Parallel MongoDB searches for speed.
        """
        entities = {'Company': [], 'Position': [], 'Date': [], 'Location': []}
    
        def is_clean_entity(entity):
            """Check if entity doesn't contain only special characters/pipes"""
            if not entity or not entity.strip():
                return False
    
            clean = re.sub(r'[^\w\s]', '', entity).strip()
            if len(clean) < 2:
                return False
    
            if re.fullmatch(r'[\|\-\+\=\*\/\\\<\>\.\,\;\:\'\"\!@#$%^&()\[\]{}]+', entity.strip()):
                return False
    
            return True
    
        # ✅ STEP 1: Extract COMPANIES FIRST and track their token positions
        company_token_positions = set()
        current_entity = None
        current_tokens = []
        current_token_indices = []
    
        for idx, (token, label) in enumerate(zip(tokens, labels)):
            if label == '[SEP]':
                if current_entity == 'Company' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Company'].append(entity_text)
                        company_token_positions.update(current_token_indices)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            elif label.startswith('B-Company'):
                if current_entity == 'Company' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Company'].append(entity_text)
                        company_token_positions.update(current_token_indices)
                current_entity = 'Company'
                current_tokens = [token]
                current_token_indices = [idx]
    
            elif label.startswith('I-Company') and current_entity == 'Company':
                current_tokens.append(token)
                current_token_indices.append(idx)
    
            elif label.startswith('E-Company') and current_entity == 'Company':
                current_tokens.append(token)
                current_token_indices.append(idx)
                entity_text = ' '.join(current_tokens)
                if is_clean_entity(entity_text):
                    entities['Company'].append(entity_text)
                    company_token_positions.update(current_token_indices)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            elif label.startswith('S-Company'):
                if current_entity == 'Company' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Company'].append(entity_text)
                        company_token_positions.update(current_token_indices)
                if is_clean_entity(token):
                    entities['Company'].append(token)
                    company_token_positions.add(idx)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            else:
                if current_entity == 'Company' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Company'].append(entity_text)
                        company_token_positions.update(current_token_indices)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
        if current_entity == 'Company' and current_tokens:
            entity_text = ' '.join(current_tokens)
            if is_clean_entity(entity_text):
                entities['Company'].append(entity_text)
                company_token_positions.update(current_token_indices)
    
        print(f"\n🔍 DEBUG: Company extraction phase:")
        print(f"   Companies extracted: {len(entities['Company'])}")
        print(f"   Company token positions masked: {len(company_token_positions)}")
    
        # ✅ STEP 2: Extract POSITIONS while SKIPPING tokens that are part of companies
        current_entity = None
        current_tokens = []
        current_token_indices = []
    
        for idx, (token, label) in enumerate(zip(tokens, labels)):
            # ✅ CRITICAL: Skip this token if it's part of a company
            if idx in company_token_positions:
                if current_entity == 'Position' and current_tokens:
                    # End current position if we hit a company token
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Position'].append(entity_text)
                    current_entity = None
                    current_tokens = []
                    current_token_indices = []
                continue
            
            if label == '[SEP]':
                if current_entity == 'Position' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Position'].append(entity_text)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            elif label.startswith('B-Position'):
                if current_entity == 'Position' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Position'].append(entity_text)
                current_entity = 'Position'
                current_tokens = [token]
                current_token_indices = [idx]
    
            elif label.startswith('I-Position') and current_entity == 'Position':
                current_tokens.append(token)
                current_token_indices.append(idx)
    
            elif label.startswith('E-Position') and current_entity == 'Position':
                current_tokens.append(token)
                current_token_indices.append(idx)
                entity_text = ' '.join(current_tokens)
                if is_clean_entity(entity_text):
                    entities['Position'].append(entity_text)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            elif label.startswith('S-Position'):
                if current_entity == 'Position' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Position'].append(entity_text)
                if is_clean_entity(token):
                    entities['Position'].append(token)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
            else:
                if current_entity == 'Position' and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if is_clean_entity(entity_text):
                        entities['Position'].append(entity_text)
                current_entity = None
                current_tokens = []
                current_token_indices = []
    
        if current_entity == 'Position' and current_tokens:
            entity_text = ' '.join(current_tokens)
            if is_clean_entity(entity_text):
                entities['Position'].append(entity_text)
    
        print(f"\n🔍 DEBUG: Position extraction phase (with masking):")
        print(f"   Positions extracted: {len(entities['Position'])}")
    
        # ✅ STEP 3: Extract dates using regex
        print(f"\n🔍 DEBUG: Searching for dates in text")
        print(f"   Text length: {len(text)} characters")
    
        if self.use_regex_dates:
            raw_dates = extract_all_dates(text)
            print(f"\n📅 DEBUG: extract_all_dates() returned {len(raw_dates)} dates")
            if raw_dates:
                print(f"   Raw dates found: {raw_dates[:10]}")
            entities['Date'] = [d for d in raw_dates if is_clean_entity(d)]
            print(f"   After filtering: {len(entities['Date'])} clean dates")
        else:
            raw_dates = self._extract_date_entities_bioes(tokens, labels)
            entities['Date'] = [d for d in raw_dates if is_clean_entity(d)]
    
        # ✅ STEP 4: Run MongoDB searches in PARALLEL for speed optimization
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        mongodb_results = {
            'companies': [],
            'positions': [],
            'locations': []
        }
        
        def search_mongodb_companies():
            """Search for companies around dates"""
            if self.company_set and entities['Date']:
                try:
                    ner_companies = entities['Company'].copy()
                    return search_companies_around_dates(
                        text=text,
                        date_pivots=entities['Date'],
                        company_set=self.company_set,
                        company_list=self.company_list,
                        ner_companies=ner_companies,
                        window_size=15
                    )
                except Exception as e:
                    print(f"⚠️  MongoDB company search error: {e}")
                    return []
            return []
        
        def search_mongodb_locations():
            """Search for location pairs"""
            if self.location_entities and self.mongodb_connection and self.mongodb_password:
                try:
                    return match_locations_from_mongodb(
                        text=text,
                        location_entities=self.location_entities,
                        mongodb_connection=self.mongodb_connection,
                        mongodb_password=self.mongodb_password
                    )
                except Exception as e:
                    print(f"⚠️  MongoDB location search error: {e}")
                    return []
            return []
        
        def search_matched_positions():
            """Search for position patterns"""
            if self.use_position_matching and self.position_matcher:
                try:
                    return find_unmatched_positions(
                        text=text,
                        ner_companies=entities['Company'],
                        ner_positions=entities['Position'],
                        ner_dates=entities['Date'],
                        position_matcher=self.position_matcher
                    )
                except Exception as e:
                    print(f"⚠️  Position matching error: {e}")
                    return []
            return []
        
        # ✅ Execute all 3 searches in parallel with 3 worker threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_search = {
                executor.submit(search_mongodb_companies): 'companies',
                executor.submit(search_mongodb_locations): 'locations',
                executor.submit(search_matched_positions): 'positions'
            }
            
            for future in as_completed(future_to_search):
                search_type = future_to_search[future]
                try:
                    result = future.result(timeout=5)  # 5 second timeout per search
                    mongodb_results[search_type] = result if result else []
                except TimeoutError:
                    print(f"⚠️  {search_type} search timed out after 5 seconds")
                    mongodb_results[search_type] = []
                except Exception as e:
                    print(f"⚠️  {search_type} search failed: {e}")
                    mongodb_results[search_type] = []
        
        # ✅ Add parallel search results to entities
        if mongodb_results['companies']:
            entities['Company'].extend(mongodb_results['companies'])
            print(f"\n🔍 MongoDB company search: Found {len(mongodb_results['companies'])} additional companies")
        
        if mongodb_results['positions']:
            entities['Position'].extend(mongodb_results['positions'])
            print(f"\n🎯 Position matching: Found {len(mongodb_results['positions'])} additional positions")
        
        if mongodb_results['locations']:
            entities['Location'] = mongodb_results['locations']
            print(f"\n📍 Location matching: Found {len(mongodb_results['locations'])} locations")
    
        # ✅ Merge sequential entities with unmatched brackets
        original_company_count = len(entities['Company'])
        entities['Company'] = merge_bracket_entities(entities['Company'])
        if len(entities['Company']) < original_company_count:
            print(f"\n🔗 Merged {original_company_count - len(entities['Company'])} company entities with unmatched brackets")
    
        return entities

    def _find_line_number(self, full_text: str, text_lines: List[str], 
                              token_idx: int, tokens: List[str]) -> int:
            """
            Find which line number a token belongs to.
            Returns the actual line number from numbered text (e.g., "1.", "2.").
            """
            try:
                # Get the token and surrounding context
                if token_idx >= len(tokens):
                    return 0

                target_token = tokens[token_idx]

                # Build a small context window around the token
                context_start = max(0, token_idx - 2)
                context_end = min(len(tokens), token_idx + 3)
                context_text = ' '.join(tokens[context_start:context_end])

                # Search each line for this context
                for line_num, line in enumerate(text_lines):
                    if not line.strip():
                        continue
                    
                    # Extract the actual line number from prefix like "1. ", "2. "
                    actual_line_num = line_num  # Default to array index
                    clean_line = line

                    if '. ' in line:
                        parts = line.split('. ', 1)
                        if parts[0].strip().isdigit():
                            actual_line_num = int(parts[0].strip())
                            clean_line = parts[1] if len(parts) > 1 else line

                    # Check if token appears in this line
                    if target_token.lower() in clean_line.lower():
                        # Verify with context to avoid false matches
                        if any(token.lower() in clean_line.lower() 
                               for token in tokens[context_start:context_end] if len(token) > 2):
                            return actual_line_num

                return 1  # Default to line 1 if not found

            except Exception as e:
                print(f"Warning: Error finding line number: {e}")
                return 1
    
    def _extract_date_entities_bioes(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[str]:
        """Extract only Date entities using BIOES logic (fallback method)"""
        date_entities = []
        current_tokens = []
        in_date = False
        
        for token, label in zip(tokens, labels):
            if label == '[SEP]':
                if in_date and current_tokens:
                    date_entities.append(' '.join(current_tokens))
                in_date = False
                current_tokens = []
                
            elif label.startswith('B-Date'):
                if in_date and current_tokens:
                    date_entities.append(' '.join(current_tokens))
                in_date = True
                current_tokens = [token]
                
            elif label.startswith('I-Date') and in_date:
                current_tokens.append(token)
                
            elif label.startswith('E-Date') and in_date:
                current_tokens.append(token)
                date_entities.append(' '.join(current_tokens))
                in_date = False
                current_tokens = []
                
            elif label.startswith('S-Date'):
                if in_date and current_tokens:
                    date_entities.append(' '.join(current_tokens))
                date_entities.append(token)
                in_date = False
                current_tokens = []
                
            else:
                if in_date and current_tokens:
                    date_entities.append(' '.join(current_tokens))
                in_date = False
                current_tokens = []
        
        if in_date and current_tokens:
            date_entities.append(' '.join(current_tokens))
        
        return date_entities
    
    def predict_with_debug(
    self,
    text: str,
    show_confidence: bool = True,
    confidence_threshold: float = 0.0,
    lstm_batch_size: int = 64,
    ner_batch_size: int = 4
) -> Dict:
        text = normalize_text_newlines(text)

        # STEP 1: TAG LINES WITH LSTM
        tagging_result = self._verify_and_tag_lines_with_lstm_batched(
            text,
            batch_size=lstm_batch_size
        )

        # ✅ FIXED: Use clean filtered_text for NER (no line numbers)
        filtered_text = tagging_result['filtered_text']
        # ✅ Keep numbered version separately for description mapping

        print(f"\n{'='*80}")
        print("DEBUG: TEXT AFTER DESC FILTERING")
        print(f"{'='*80}")
        print(f"Original text length: {len(text)} characters")
        print(f"Filtered text (for NER) length: {len(filtered_text)} characters")

        # Use CLEAN filtered text for NER
        tokens = filtered_text.split()

        if not tokens:
            return {
                'text': text,
                'original_text': text,
                'filtered_text': filtered_text,
                'tagging_result': tagging_result,
                'tokens': [],
                'predictions': {'Company': [], 'Position': [], 'Date': [], 'Location': []},
                'debug_info': [],
                'used_sliding_window': False,
                'num_windows': 0,
                'corrections_applied': False,
                'corrections': [],
                'used_regex_dates': self.use_regex_dates,
                'used_position_matching': self.use_position_matching,
                'matched_positions': [],
                'used_mongodb_search': bool(self.company_set),
                'mongodb_companies': []
            }

        encoding_test = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        num_subtokens = len(encoding_test['input_ids'])
        use_sliding_window = num_subtokens > self.max_seq_length

        if not use_sliding_window:
            result = self._predict_single_sequence(
                filtered_text,
                tokens,
                show_confidence,
                confidence_threshold
            )
            result['used_sliding_window'] = False
            result['num_windows'] = 1
            result['original_text'] = text
            result['filtered_text'] = filtered_text
            result['tagging_result'] = tagging_result
            return result

        print(f"\nProcessing with BATCHED sliding windows:")
        print(f"  - Total tokens: {len(tokens)}")
        print(f"  - NER batch size: {ner_batch_size}")

        window_ranges = self._create_fixed_windows(tokens)
        print(f"  - Number of windows: {len(window_ranges)}")

        all_labels, all_probs = self._process_windows_batched(
            tokens,
            window_ranges,
            batch_size=ner_batch_size
        )

        merged_labels, merged_probs = self._merge_with_weighted_voting(
            all_labels, all_probs, window_ranges, tokens
        )

        merged_labels = self._post_process_entities(tokens, merged_labels)

        confidences_list = [float(np.max(prob)) for prob in merged_probs]
        corrected_labels, corrected_confidences, corrections = self.correct_bioes_sequences(
            tokens, merged_labels, confidences_list
        )

        if corrections:
            print(f"\n  - Applied {len(corrections)} BIOES sequence corrections")

        debug_info = []
        for idx, (token, label, orig_label, conf) in enumerate(
            zip(tokens, corrected_labels, merged_labels, corrected_confidences)
        ):
            prob = merged_probs[idx]
            confidence_info = []
            if show_confidence:
                sorted_indices = np.argsort(prob)[::-1]
                for sorted_idx in sorted_indices[:5]:
                    conf_val = float(prob[sorted_idx])
                    if conf_val >= confidence_threshold:
                        lbl = self.id_to_label.get(sorted_idx, 'UNKNOWN')
                        confidence_info.append({'label': lbl, 'confidence': conf_val})

            debug_info.append({
                'token': token,
                'predicted_label': label,
                'original_label': orig_label if (orig_label != label) else None,
                'confidence': conf,
                'top_predictions': confidence_info,
                'corrected': (orig_label != label)
            })

        # Extract entities from CLEAN filtered text (no line numbers)
        entities = self._extract_entities_with_regex_dates(filtered_text, tokens, corrected_labels)

        ner_companies = []
        current_entity = None
        current_tokens_buf = []
        for token, label in zip(tokens, corrected_labels):
            if label == '[SEP]':
                if current_entity == 'Company' and current_tokens_buf:
                    ner_companies.append(' '.join(current_tokens_buf))
                current_entity = None
                current_tokens_buf = []
            elif label.startswith('B-Company'):
                if current_entity == 'Company' and current_tokens_buf:
                    ner_companies.append(' '.join(current_tokens_buf))
                current_entity = 'Company'
                current_tokens_buf = [token]
            elif label.startswith('I-Company') and current_entity == 'Company':
                current_tokens_buf.append(token)
            elif label.startswith('E-Company') and current_entity == 'Company':
                current_tokens_buf.append(token)
                ner_companies.append(' '.join(current_tokens_buf))
                current_entity = None
                current_tokens_buf = []
            elif label.startswith('S-Company'):
                if current_entity == 'Company' and current_tokens_buf:
                    ner_companies.append(' '.join(current_tokens_buf))
                ner_companies.append(token)
                current_entity = None
                current_tokens_buf = []
            else:
                if current_entity == 'Company' and current_tokens_buf:
                    ner_companies.append(' '.join(current_tokens_buf))
                current_entity = None
                current_tokens_buf = []
        if current_entity == 'Company' and current_tokens_buf:
            ner_companies.append(' '.join(current_tokens_buf))

        mongodb_companies = [c for c in entities['Company'] if c not in ner_companies]

        matched_positions = []
        if self.use_position_matching:
            matched_positions = find_unmatched_positions(
                filtered_text, entities['Company'], [], entities['Date'], self.position_matcher
            )

        return {
            'text': filtered_text,
            'original_text': text,
            'filtered_text': filtered_text,
            'tagging_result': tagging_result,
            'tokens': tokens,
            'predictions': entities,
            'debug_info': debug_info,
            'used_sliding_window': True,
            'num_windows': len(window_ranges),
            'corrections_applied': len(corrections) > 0,
            'corrections': corrections,
            'used_regex_dates': self.use_regex_dates,
            'used_position_matching': self.use_position_matching,
            'matched_positions': matched_positions,
            'used_mongodb_search': bool(self.company_set),
            'mongodb_companies': mongodb_companies
        }
        
    
    
    def _predict_single_sequence(
    self,
    text: str,
    tokens: List[str],
    show_confidence: bool,
    confidence_threshold: float,
    ner_batch_size: int = 4  # ✅ ADD THIS if needed
) -> Dict:
        """
        Original prediction logic for sequences <= 512 tokens.
        Now works on pre-filtered text (DESC lines already removed).
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        word_ids = encoding.word_ids()

        predicted_labels = []
        predicted_confidences = []
        all_probs = []

        for word_idx in range(len(tokens)):
            token_idx = None
            for i, w_id in enumerate(word_ids):
                if w_id == word_idx:
                    token_idx = i
                    break
                
            if token_idx is not None:
                pred_id = predictions[0][token_idx].item()
                pred_label = self.id_to_label.get(pred_id, 'O')
                predicted_labels.append(pred_label)

                token_probs = probabilities[0][token_idx].cpu().numpy()
                all_probs.append(token_probs)

                confidence = float(token_probs[pred_id])
                predicted_confidences.append(confidence)
            else:
                predicted_labels.append('O')
                predicted_confidences.append(1.0)

                o_probs = np.zeros(self.num_labels)
                o_id = self.label_to_id.get('O', 0)
                o_probs[o_id] = 1.0
                all_probs.append(o_probs)

        original_labels = predicted_labels.copy()

        corrected_labels, corrected_confidences, corrections = self.correct_bioes_sequences(
            tokens, predicted_labels, predicted_confidences
        )

        if corrections:
            print(f"\nApplied {len(corrections)} BIOES sequence corrections:")
            for corr in corrections[:10]:
                print(f"  - Token '{corr['token']}' (pos {corr['position']}): {corr['original']} -> {corr['corrected']}")
                print(f"    Reason: {corr['reason']}")
            if len(corrections) > 10:
                print(f"  ... and {len(corrections) - 10} more corrections")

        debug_info = []

        for word_idx in range(len(tokens)):
            token = tokens[word_idx]
            label = corrected_labels[word_idx]
            orig_label = original_labels[word_idx]
            conf = corrected_confidences[word_idx]
            token_probs = all_probs[word_idx]

            confidence_info = []
            if show_confidence:
                sorted_indices = np.argsort(token_probs)[::-1]
                for idx in sorted_indices[:5]:
                    conf_val = float(token_probs[idx])
                    if conf_val >= confidence_threshold:
                        lbl = self.id_to_label.get(idx, 'UNKNOWN')
                        confidence_info.append({
                            'label': lbl,
                            'confidence': conf_val
                        })

            was_corrected = (orig_label != label)

            debug_info.append({
                'token': token,
                'predicted_label': label,
                'original_label': orig_label if was_corrected else None,
                'confidence': conf,
                'top_predictions': confidence_info,
                'corrected': was_corrected
            })

        # Track NER companies and positions BEFORE augmentation
        ner_companies_before = []
        ner_positions_before = []
        current_entity = None
        current_tokens = []

        for token, label in zip(tokens, corrected_labels):
            if label == '[SEP]':
                if current_entity == 'Company' and current_tokens:
                    ner_companies_before.append(' '.join(current_tokens))
                elif current_entity == 'Position' and current_tokens:
                    ner_positions_before.append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
            elif label.startswith('B-Company'):
                if current_entity == 'Company' and current_tokens:
                    ner_companies_before.append(' '.join(current_tokens))
                current_entity = 'Company'
                current_tokens = [token]
            elif label.startswith('B-Position'):
                if current_entity == 'Position' and current_tokens:
                    ner_positions_before.append(' '.join(current_tokens))
                current_entity = 'Position'
                current_tokens = [token]
            elif label.startswith('I-Company') and current_entity == 'Company':
                current_tokens.append(token)
            elif label.startswith('I-Position') and current_entity == 'Position':
                current_tokens.append(token)
            elif label.startswith('E-Company') and current_entity == 'Company':
                current_tokens.append(token)
                ner_companies_before.append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
            elif label.startswith('E-Position') and current_entity == 'Position':
                current_tokens.append(token)
                ner_positions_before.append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
            elif label.startswith('S-Company'):
                if current_entity == 'Company' and current_tokens:
                    ner_companies_before.append(' '.join(current_tokens))
                ner_companies_before.append(token)
                current_entity = None
                current_tokens = []
            elif label.startswith('S-Position'):
                if current_entity == 'Position' and current_tokens:
                    ner_positions_before.append(' '.join(current_tokens))
                ner_positions_before.append(token)
                current_entity = None
                current_tokens = []
            else:
                if current_entity == 'Company' and current_tokens:
                    ner_companies_before.append(' '.join(current_tokens))
                elif current_entity == 'Position' and current_tokens:
                    ner_positions_before.append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []

        if current_entity == 'Company' and current_tokens:
            ner_companies_before.append(' '.join(current_tokens))
        elif current_entity == 'Position' and current_tokens:
            ner_positions_before.append(' '.join(current_tokens))

        print(f"\n NER Model extracted {len(ner_companies_before)} company(s)")
        print(f" NER Model extracted {len(ner_positions_before)} position(s)")

        # Extract entities from filtered text
        entities = self._extract_entities_with_regex_dates(text, tokens, corrected_labels)

        # Calculate augmented entities
        mongodb_companies = []
        ner_companies_normalized = {c.lower().strip() for c in ner_companies_before}

        for comp in entities['Company']:
            if comp.lower().strip() not in ner_companies_normalized:
                mongodb_companies.append(comp)

        matched_positions = []
        ner_positions_normalized = {p.lower().strip() for p in ner_positions_before}

        for pos in entities['Position']:
            if pos.lower().strip() not in ner_positions_normalized:
                matched_positions.append(pos)

        if self.use_regex_dates:
            print(f"\n… Regex date extraction: Found {len(entities['Date'])} dates")

        if self.company_set:
            if mongodb_companies:
                print(f"\n MongoDB search: Found {len(mongodb_companies)} NEW company(s)")
            else:
                print(f"\n MongoDB search: No new companies found")

        if self.use_position_matching:
            if matched_positions:
                print(f"\n Position matching: Found {len(matched_positions)} NEW position(s)")
            else:
                print(f"\n Position matching: No new positions found")

        return {
            'text': text,
            'tokens': tokens,
            'predictions': entities,
            'debug_info': debug_info,
            'corrections_applied': len(corrections) > 0,
            'corrections': corrections,
            'used_regex_dates': self.use_regex_dates,
            'used_position_matching': self.use_position_matching,
            'matched_positions': matched_positions,
            'ner_positions': ner_positions_before,
            'used_mongodb_search': bool(self.company_set),
            'mongodb_companies': mongodb_companies,
            'ner_companies': ner_companies_before
        }
    
    def predict_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """Process multiple texts in batch"""
        results = []
        
        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Processing texts")
            except ImportError:
                pass
        
        for text in iterator:
            result = self.predict_with_debug(text, show_confidence=False)
            results.append(result)
        
        return results
    
    def print_debug_results(self, result: Dict, min_confidence: float = 0.5):
        """
        Print debug results with DESC tagging information displayed first.
        MODIFIED: Filter out entities with pipes/special characters in display.
        """
        # [Previous code remains the same until entity display...]

        # Enhanced filtering for display
        def should_display_entity(entity):
            """Check if entity should be displayed (filter out pipes/special chars)"""
            if not entity or not entity.strip():
                return False

            # Remove all non-alphanumeric and check length
            clean = re.sub(r'[^\w\s]', '', entity).strip()
            if len(clean) < 2:
                return False

            # Filter out entities that are just special characters
            if re.fullmatch(r'[\|\-\+\=\*\/\\\<\>\.\,\;\:\'\"\!@#$%^&()\[\]{}]+', entity.strip()):
                return False

            return True

        print("\n" + "-" * 80)
        print("\nEXTRACTED ENTITIES:")
        print("-" * 80)

        matched_positions_set = set(result.get('matched_positions', []))
        mongodb_companies_set = set(result.get('mongodb_companies', []))

        for entity_type, entities_list in result['predictions'].items():
            if entities_list:
                print(f"\n{entity_type}:")

                # Filter entities for display
                filtered_entities = [e for e in entities_list if should_display_entity(e)]

                if not filtered_entities:
                    print(f"   (All entities filtered out as noise)")
                    continue

                if entity_type == 'Company':
                    ner_count = 0
                    mongodb_count = 0

                    for entity in filtered_entities:
                        if entity in mongodb_companies_set:
                            print(f"    {entity}  [MONGODB SEARCH]")
                            mongodb_count += 1
                        else:
                            print(f"   {entity}  [NER MODEL]")
                            ner_count += 1

                    if result.get('used_mongodb_search', False):
                        print(f"\n   Company Breakdown:")
                        print(f"     NER Model: {ner_count} companies")
                        print(f"     MongoDB Search: {mongodb_count} companies")
                        print(f"     Total: {len(filtered_entities)} companies")

                elif entity_type == 'Position':
                    ner_count = 0
                    pattern_count = 0

                    for entity in filtered_entities:
                        if entity in matched_positions_set:
                            print(f"    {entity}  [PATTERN MATCHED]")
                            pattern_count += 1
                        else:
                            print(f"   {entity}  [NER MODEL]")
                            ner_count += 1

                    if result.get('used_position_matching', False):
                        print(f"\n   Position Breakdown:")
                        print(f"     NER Model: {ner_count} positions")
                        print(f"     Pattern Matching: {pattern_count} positions")
                        print(f"     Total: {len(filtered_entities)} positions")

                else:  # Date and Location entities
                    for entity in filtered_entities:
                        print(f"   {entity}")

                    if entity_type == 'Date' and result.get('used_regex_dates', False):
                        print(f"\n  … (Extracted using comprehensive regex patterns)")

        if not any(result['predictions'].values()):
            print("  (No entities found)")

        print("\n" + "=" * 80)


# ==================== MAIN TESTING CODE ====================

#==================MAPPING==================================================================================================

"""
Description Mapping Functions for Triplet-Description Association
Maps DESC lines back to triplets based on line proximity in original text
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def extract_line_numbers_from_text(text: str) -> Dict[int, str]:
    """
    Build line_map using SEQUENTIAL non-empty line numbering,
    consistent with LSTM tagging output.
    Empty lines are skipped but do NOT consume a line number.
    """
    line_map = {}
    seq_num = 0  # sequential counter, only increments for non-empty lines
    
    for raw_line in text.split('\n'):
        if raw_line.strip():
            seq_num += 1
            line_map[seq_num] = raw_line.strip()
    
    return line_map


def find_triplet_line_positions(
    original_text: str,
    triplets: List[Dict],
    line_map: Dict[int, str]
) -> List[Dict]:
    triplets_with_positions = []

    for triplet in triplets:
        position_text = triplet['Position']
        company_text = triplet['Company']
        date_text = triplet['Year']

        is_copied_company = triplet.get('company_source') in ('COPIED_FROM_PREVIOUS', 'NOT_FOUND')

        # ── normalise helper ──────────────────────────────────────────────
        def norm(s: str) -> str:
            """Lowercase + collapse all whitespace/punctuation variants."""
            s = s.lower().strip()
            # collapse whitespace around punctuation (e.g. "alabama ," → "alabama,")
            s = re.sub(r'\s+([,.])', r'\1', s)
            s = re.sub(r'\s+', ' ', s)
            return s

        norm_position = norm(position_text)
        norm_company  = norm(company_text)
        norm_date     = norm(date_text)

        # ── Step 1: find date line (dates are almost always unique) ───────
        date_lines = [
            ln for ln, lc in line_map.items()
            if norm_date in norm(lc)
        ]
        date_line = date_lines[0] if date_lines else None

        # ── Step 2: find position line ────────────────────────────────────
        position_candidates = [
            ln for ln, lc in line_map.items()
            if norm_position in norm(lc)
        ]

        if date_line is not None and position_candidates:
            position_line = min(position_candidates,
                                key=lambda x: abs(x - date_line))
        elif position_candidates:
            position_line = position_candidates[0]
        else:
            position_line = None

        # ── Step 3: find company line ─────────────────────────────────────
        # Anchor = date_line if available, else position_line
        # Only accept candidates within HEADER_WINDOW lines of the anchor
        company_line = None

        if not is_copied_company:
            HEADER_WINDOW = 5
            anchor = date_line if date_line is not None else position_line

            if anchor is not None:
                # Search within window first
                window_candidates = [
                    ln for ln, lc in line_map.items()
                    if norm_company in norm(lc)
                    and abs(ln - anchor) <= HEADER_WINDOW
                ]

                if window_candidates:
                    company_line = min(window_candidates,
                                       key=lambda x: abs(x - anchor))
                else:
                    # Fallback: use position_line as tighter anchor
                    # Only accept lines that are BEFORE or AT the
                    # position line (company always comes before/with position)
                    if position_line is not None:
                        before_pos = [
                            ln for ln, lc in line_map.items()
                            if norm_company in norm(lc)
                            and ln <= position_line + 2  # tiny slack
                        ]
                        if before_pos:
                            company_line = max(before_pos)  # latest before position

        # ── Step 4: compute min/max strictly from header lines ────────────
        # IMPORTANT: max_line must NOT include a company line that is far
        # from the date/position — that would push description mapping
        # into a wrong section.
        header_lines = [l for l in [position_line, date_line] if l is not None]

        # Only include company_line in range if it is genuinely close
        if company_line is not None and date_line is not None:
            if abs(company_line - date_line) <= 5:
                header_lines.append(company_line)
            # else: company_line is recorded but NOT used for max_line

        if header_lines:
            min_line = min(header_lines)
            max_line = max(header_lines)
        else:
            min_line = None
            max_line = None

        triplet_with_pos = triplet.copy()
        triplet_with_pos['line_positions'] = {
            'position_line': position_line,
            'company_line':  company_line,
            'date_line':     date_line,
            'min_line':      min_line,
            'max_line':      max_line
        }
        triplets_with_positions.append(triplet_with_pos)

    return triplets_with_positions


def contains_section_header(text: str, section_keywords=None) -> tuple:
    """
    Check if text is a pure dash-separator line (any number of dashes >= 1).

    Square-bracket header detection ([HEADER] before dashes) is handled
    separately inside map_descriptions_to_triplets with lookahead logic.

    Returns:
        (True, matched_text)  if the line is a dash separator
        (False, None)         otherwise
    """
    text_stripped = text.strip()

    # Any line consisting solely of one or more dashes
    if re.match(r'^-+$', text_stripped):
        return True, text_stripped

    return False, None



def extract_all_triplet_entities(triplets: List[Dict]) -> Dict[str, Set[str]]:
    """
    Extract all unique entities (positions, companies, dates) from ALL triplets.
    These will be used as stop markers during description mapping.
    
    FIXED: Filters out overly generic/short entities that cause false positives.
    
    Args:
        triplets: List of all triplets (including DUOs)
    
    Returns:
        Dictionary with sets of unique entities:
        - 'positions': Set of position strings (lowercase for matching)
        - 'companies': Set of company strings (lowercase for matching)
        - 'dates': Set of date strings (lowercase for matching)
    """
    positions = set()
    companies = set()
    dates = set()
    
    # List of generic words that should NOT be used as stop markers
    generic_stopwords = {
        'data', 'analyst', 'engineer', 'scientist', 'developer', 'manager',
        'consultant', 'intern', 'assistant', 'associate', 'specialist',
        'coordinator', 'lead', 'senior', 'junior', 'staff', 'principal',
        'research', 'software', 'systems', 'business', 'project', 'product',
        'technical', 'hr', 'ai', 'ml', 'admin', 'sales', 'marketing'
    }
    
    def is_valid_stop_entity(entity: str, min_words: int = 2, min_chars: int = 8) -> bool:
        """
        Check if entity is specific enough to be used as a stop marker.
        
        Criteria:
        - Must have at least min_words words OR
        - Must have at least min_chars characters AND not be a generic word
        """
        if not entity or not entity.strip():
            return False
        
        entity_lower = entity.lower().strip()
        words = entity_lower.split()
        
        # Reject single generic words
        if len(words) == 1 and entity_lower in generic_stopwords:
            return False
        
        # Accept multi-word entities (they're specific enough)
        if len(words) >= min_words:
            return True
        
        # For single-word entities, must be long enough and not generic
        if len(entity_lower) >= min_chars and entity_lower not in generic_stopwords:
            return True
        
        return False
    
    for triplet in triplets:
        # Extract and validate position
        if triplet.get('Position'):
            pos = triplet['Position'].strip()
            if is_valid_stop_entity(pos, min_words=2, min_chars=10):
                positions.add(pos.lower())
        
        # Extract and validate company
        if triplet.get('Company'):
            comp = triplet['Company'].strip()
            if is_valid_stop_entity(comp, min_words=2, min_chars=8):
                companies.add(comp.lower())
        
        # Extract and validate date (dates are usually specific enough)
        if triplet.get('Year'):
            date = triplet['Year'].strip()
            # Dates are almost always valid stop markers if they have multiple words
            if len(date.split()) >= 2:
                dates.add(date.lower())
    
    return {
        'positions': positions,
        'companies': companies,
        'dates': dates
    }

def line_contains_triplet_entity(
    line_text: str,
    triplet_entities: Dict[str, Set[str]],
    current_triplet: Dict
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a line contains ANY triplet entity (position, company, or date)
    that is NOT part of the current triplet.
    
    FIXED: Uses more strict matching to avoid false positives from partial word matches.
    
    Args:
        line_text: Text of the line to check
        triplet_entities: Dictionary of all triplet entities
        current_triplet: The current triplet being processed (to exclude its own entities)
    
    Returns:
        Tuple of (found, entity_type, entity_text)
        - found: True if a different triplet's entity was found
        - entity_type: Type of entity found ('position', 'company', 'date', or None)
        - entity_text: The actual entity text found (or None)
    """
    line_lower = line_text.lower().strip()
    
    # Get current triplet's entities (normalized)
    current_position = current_triplet.get('Position', '').lower().strip()
    current_company = current_triplet.get('Company', '').lower().strip()
    current_date = current_triplet.get('Year', '').lower().strip()
    
    # Helper function for stricter matching
    def is_entity_match(entity: str, line: str) -> bool:
        """
        Check if entity appears as a complete phrase in the line.
        Uses word boundaries and case-insensitive matching.
        
        Returns True only if:
        1. Entity appears as complete phrase (not substring)
        2. Entity has proper word boundaries OR is surrounded by punctuation
        """
        if not entity or len(entity.strip()) < 3:
            # Skip very short entities to avoid false positives
            return False
        
        entity_lower = entity.lower().strip()
        
        # For single-word entities, use strict word boundary matching
        if ' ' not in entity_lower:
            # Create pattern with word boundaries
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            return bool(re.search(pattern, line, re.IGNORECASE))
        
        # For multi-word entities, check if the complete phrase exists
        # Allow for some flexibility with punctuation
        entity_pattern = re.escape(entity_lower)
        # Make spaces flexible (can have punctuation between words)
        entity_pattern = entity_pattern.replace(r'\ ', r'[\s,\-]+')
        
        return bool(re.search(entity_pattern, line, re.IGNORECASE))
    
    # Check for OTHER positions (not the current one)
    for position in triplet_entities['positions']:
        if not position or position == current_position:
            continue
        
        if is_entity_match(position, line_lower):
            return True, 'position', position
    
    # Check for OTHER companies (not the current one)
    for company in triplet_entities['companies']:
        if not company or company == current_company:
            continue
        
        if is_entity_match(company, line_lower):
            return True, 'company', company
    
    # Check for OTHER dates (not the current one)
    for date in triplet_entities['dates']:
        if not date or date == current_date:
            continue
        
        # Dates can have special characters, so use more flexible matching
        # But still require substantial match
        date_clean = re.sub(r'[^\w\s]', ' ', date.lower()).strip()
        if len(date_clean.split()) >= 2:  # Multi-word dates
            # Check if most of the date words appear in sequence
            date_words = date_clean.split()
            # Create pattern: all words must appear with max 2 words gap
            pattern_parts = [re.escape(w) for w in date_words]
            pattern = r'\b' + r'[\s\-,]*'.join(pattern_parts) + r'\b'
            if re.search(pattern, line_lower):
                return True, 'date', date
    
    return False, None, None



def map_descriptions_to_triplets(
    original_raw_text: str,
    tagging_result: Dict,
    triplets_with_positions: List[Dict],
    line_map: Dict[int, str],
    duos_with_positions: List[Dict] = None,
    window_before: int = 0,
    window_after: int = 50,
    section_keywords: Dict[str, List[str]] = None
) -> List[Dict]:
    """
    Map ALL lines between triplets as descriptions using RAW TEXT.
    
    FIXED V8.0:
    1. Section header detection stops mapping
    2. Next triplet's entities stop mapping
    3. ✅ REMOVED: Stopping for other random company/position names in descriptions
    """
    print(f"\n{'='*80}")
    print("DESCRIPTION MAPPING - FIXED LOGIC V8.0")
    print("Improvements:")
    print("  - Descriptions map entirely between triplets")
    print("  - Only stops at: (1) Section headers, (2) Next triplet entities")
    print("  - NO longer stops for random company/position names in descriptions")
    print(f"{'='*80}")
    
    # Default section keywords if not provided
    if section_keywords is None:
        section_keywords = {
            'education_sections': [
                'education', 'educational background', 'academic qualifications',
                'education and training', 'academic background', 'educational qualifications',
                'degrees', 'degree', 'academic credentials', 'schooling'
            ],
            'avoid_sections': [
                'work experience', 'projects', 'skills', 'experience', 'achievements',
                'certifications', 'awards', 'publications', 'contact', 'summary',
                'objective', 'profile', 'employment', 'professional experience', 'languages',
                'technical skills', 'research projects', 'top_section', 'unassigned',
                'prior experience', 'previous experience', 'previous employment',
                'past experience', 'employment history', 'career history'
            ]
        }
    
    # Combine all triplets (including DUOs)
    all_triplets = triplets_with_positions.copy()
    if duos_with_positions:
        all_triplets.extend(duos_with_positions)
    
    # ✅ REMOVED: No longer need to extract all triplet entities since we don't use CHECK 3
    
    # Sort triplets by their max_line
    sorted_triplets = sorted(
        triplets_with_positions, 
        key=lambda t: t['line_positions']['max_line'] if t['line_positions']['max_line'] else float('inf')
    )
    
    triplets_with_descriptions = []
    
    for i, triplet in enumerate(sorted_triplets):
        line_pos = triplet['line_positions']
        
        if line_pos['min_line'] is None or line_pos['max_line'] is None:
            triplet_copy = triplet.copy()
            triplet_copy['descriptions'] = []
            triplet_copy['description_text'] = ""
            triplets_with_descriptions.append(triplet_copy)
            continue
        
        # Get current triplet's entities for reference
        current_position = clean_line_numbers_from_entity(triplet.get('Position', '')).lower().strip()
        current_company = clean_line_numbers_from_entity(triplet.get('Company', '')).lower().strip()
        current_date = triplet.get('Year', '').lower().strip()
        
        # Start from AFTER current triplet's max line
        search_start = line_pos['max_line'] + 1
        
        # Find next triplet's MIN line
        next_triplet_min_line = None
        next_triplet_entities = None
        # Replace the block that extracts next_triplet_entities:

        if i + 1 < len(sorted_triplets):
            next_triplet = sorted_triplets[i + 1]
            next_line_pos = next_triplet['line_positions']
            if next_line_pos['min_line']:
                next_triplet_min_line = next_line_pos['min_line']

                next_is_copied_company = next_triplet.get('company_source') in ('COPIED_FROM_PREVIOUS', 'NOT_FOUND')

                next_triplet_entities = {
                    'position': clean_line_numbers_from_entity(next_triplet.get('Position', '')).lower().strip(),
                    # Don't use copied company as a stop marker — it belongs to a different job entry
                    'company': '' if next_is_copied_company else clean_line_numbers_from_entity(next_triplet.get('Company', '')).lower().strip(),
                    'date': next_triplet.get('Year', '').lower().strip()
                }
        
        # Determine search_end
        if next_triplet_min_line and next_triplet_min_line > search_start:
            search_end = next_triplet_min_line - 1      # ← no window cap
        else:
            search_end = search_start + window_after
        
        # Collect description lines from RAW text
        matched_descriptions = []
        stopped_by_header = False
        stopped_by_next_triplet = False
        stop_reason = None
        stop_line = None
        
        if search_start <= search_end:
            for line_num in range(search_start, search_end + 1):
                if line_num not in line_map:
                    continue
                
                raw_text = line_map[line_num].strip()
                
                # Skip empty lines but continue searching
                if not raw_text:
                    continue
                
                raw_text_lower = raw_text.lower()
                
                # Check if this is a line continuation (ends with hyphen)
                is_continuation = False
                prev_line_ends_with_hyphen = False
                
                if line_num > search_start and (line_num - 1) in line_map:
                    prev_line = line_map[line_num - 1].strip()
                    if prev_line.endswith('-'):
                        prev_line_ends_with_hyphen = True
                        is_continuation = True
                
                # For continuation lines, be more lenient on stopping conditions
                if not is_continuation:
                    # ✅ CHECK 1: Section header detection
                    # Case A: Pure dash line (any length >= 1)
                    has_header, matched_header = contains_section_header(raw_text)
                    if has_header:
                        stopped_by_header = True
                        stop_line         = line_num
                        stop_reason       = "section separator (dashes)"
                        print(f"   ⚠ Triplet {triplet['id']}: {stop_reason} at line {line_num}, stopping BEFORE this line")
                        break
                    
                    # Case B: [HEADER] line — stop here if the NEXT non-empty line is a dash line
                    bracket_match = re.match(r'^\[.+\]', raw_text.strip())
                    if bracket_match:
                        # Look ahead for the next non-empty line
                        next_line_text = None
                        for lookahead_num in range(line_num + 1, line_num + 5):
                            if lookahead_num in line_map:
                                lookahead_text = line_map[lookahead_num].strip()
                                if lookahead_text:
                                    next_line_text = lookahead_text
                                    break
                                
                        if next_line_text and re.match(r'^-+$', next_line_text):
                            stopped_by_header = True
                            stop_line         = line_num
                            stop_reason       = f"square bracket header '{bracket_match.group(0)}' before dashes"
                            print(f"   ⚠ Triplet {triplet['id']}: {stop_reason} at line {line_num}, stopping BEFORE this line")
                            break
                    
                    # ✅ CHECK 2: Next triplet's entities (KEEP THIS)
                    if next_triplet_entities:
                        # Check position
                        if next_triplet_entities['position'] and len(next_triplet_entities['position']) > 3:
                            # ✅ ADDED: Skip if next triplet's position is same as current triplet's position
                            if next_triplet_entities['position'] != current_position:
                                position_pattern = r'\b' + re.escape(next_triplet_entities['position']) + r'\b'
                                if re.search(position_pattern, raw_text_lower, re.IGNORECASE):
                                    stopped_by_next_triplet = True
                                    stop_line = line_num
                                    stop_reason = f"next triplet's position"
                                    print(f"   ⚠ Triplet {triplet['id']}: Found {stop_reason} at line {line_num}, stopping BEFORE this line")
                                    break
                                
                        # Check company
                        if next_triplet_entities['company'] and len(next_triplet_entities['company']) > 3:
                            # ✅ ADDED: Skip if next triplet's company is same as current triplet's company
                            if next_triplet_entities['company'] != current_company:
                                company_pattern = r'\b' + re.escape(next_triplet_entities['company']) + r'\b'
                                if re.search(company_pattern, raw_text_lower, re.IGNORECASE):
                                    stopped_by_next_triplet = True
                                    stop_line = line_num
                                    stop_reason = f"next triplet's company"
                                    print(f"   ⚠ Triplet {triplet['id']}: Found {stop_reason} at line {line_num}, stopping BEFORE this line")
                                    break
                            
                        # Check next triplet's date
                        if next_triplet_entities['date'] and len(next_triplet_entities['date']) > 5:
                            if next_triplet_entities['date'] in raw_text_lower:
                                stopped_by_next_triplet = True
                                stop_line = line_num
                                stop_reason = f"next triplet's date"
                                print(f"   ⚠ Triplet {triplet['id']}: Found {stop_reason} at line {line_num}, stopping BEFORE this line")
                                break
                    
                    # ✅ REMOVED: CHECK 3 - No longer checking for other triplet entities
                    # This allows descriptions to include company/position names without stopping
                
                # ✅ Add line if it passed all checks (or is a continuation)
                distance = line_num - line_pos['max_line']
                
                # Merge hyphenated line breaks
                if prev_line_ends_with_hyphen and matched_descriptions:
                    # Merge with previous line, removing the hyphen
                    prev_desc = matched_descriptions[-1]
                    prev_text = prev_desc['text']
                    
                    # Remove trailing hyphen and merge
                    if prev_text.endswith('-'):
                        prev_text = prev_text[:-1]  # Remove hyphen
                    
                    merged_text = prev_text + raw_text
                    prev_desc['text'] = merged_text
                    
                    print(f"   ✅ Triplet {triplet['id']}: Merged continuation line {line_num} with line {prev_desc['line_num']}")
                else:
                    matched_descriptions.append({
                        'line_num': line_num,
                        'text': raw_text,
                        'distance': distance
                    })
        
        # Sort by line number
        matched_descriptions.sort(key=lambda x: x['line_num'])
        
        # Add to triplet
        triplet_copy = triplet.copy()
        triplet_copy['descriptions'] = matched_descriptions
        triplet_copy['description_text'] = ' '.join([d['text'].strip() for d in matched_descriptions])
        triplets_with_descriptions.append(triplet_copy)
        
        # Logging
        if stopped_by_header or stopped_by_next_triplet:
            last_included_line = matched_descriptions[-1]['line_num'] if matched_descriptions else search_start - 1
            print(f"   ✅ Triplet {triplet['id']}: Mapped {len(matched_descriptions)} lines "
                  f"(lines {search_start}-{last_included_line}) - stopped BEFORE line {stop_line}")
        elif search_start > search_end:
            print(f"   ⚠ Triplet {triplet['id']}: No description lines available")
        else:
            print(f"   ✅ Triplet {triplet['id']}: Mapped {len(matched_descriptions)} lines "
                  f"(lines {search_start}-{search_end}) - reached window limit")
    
    total_desc_lines = sum(len(t['descriptions']) for t in triplets_with_descriptions)
    print(f"\n✅ Total description lines mapped: {total_desc_lines}")
    print(f"{'='*80}\n")
    
    return triplets_with_descriptions

def calculate_work_duration(date_string: str) -> Dict:
    """
    COMPREHENSIVE work duration calculator supporting ALL patterns from extract_all_dates() regex.
    
    Supported formats:
    - Month YYYY: "July 2024- Present", "Jan 2023 - Apr 2023"
    - Month'YY: "Aug'19 to Nov'19", "March '22 – April '24"
    - MM/YYYY: "01/2015 - 04/2018", "06/2025 - 06/2035"
    - MM-YYYY: "01-2015 - 04-2018"
    - MM.YYYY: "01.2015 - 04.2018"
    - YYYY-YYYY: "2020-2021", "2023-2024"
    - Seasons: "Spring 2023 - Fall 2023", "Summer 2022 - Winter 2023"
    - Quarters: "Q1 2023 - Q2 2024", "Q1-Q4 2023"
    - Academic terms: "Fall Semester 2023 - Spring Semester 2024"
    - With days: "March 15, 2023 - June 20, 2023"
    - Duration in parens: "June 2025 - June 2035 (10 years)"
    - Parentheses/brackets: "(Sep 2023 - Dec 2023)", "[2020-2021]"
    - Newlines: "June\n2013 - June 2014"
    """
    import re
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    if not date_string or not date_string.strip():
        return {
            'start_date': None,
            'end_date': None,
            'duration_months': 0,
            'duration_years': 0.0,
            'duration_display': 'Unknown',
            'is_ongoing': False
        }
    
    # ============= COMPREHENSIVE MAPPINGS =============
    
    # Month mapping (full + abbreviated)
    MONTH_MAP = {
        'january': 1, 'jan': 1, 'janv': 1,
        'february': 2, 'feb': 2, 'févr': 2,
        'march': 3, 'mar': 3, 'mars': 3,
        'april': 4, 'apr': 4, 'avr': 4, 'abr': 4,
        'may': 5, 'mai': 5,
        'june': 6, 'jun': 6, 'juin': 6,
        'july': 7, 'jul': 7, 'juill': 7,
        'august': 8, 'aug': 8, 'août': 8, 'ago': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12, 'déc': 12, 'dic': 12
    }
    
    # Season mapping (start month)
    SEASON_MAP = {
        'spring': 3, 'spr': 3, 'vernal': 3,
        'summer': 6, 'sum': 6,
        'fall': 9, 'autumn': 9, 'aut': 9, 'autumnal': 9,
        'winter': 12, 'win': 12,
        'dry season': 1, 'wet season': 6, 'rainy season': 6,
        'harvest season': 9, 'holiday season': 12
    }
    
    # Quarter mapping (start month)
    QUARTER_MAP = {
        'q1': 1, '1': 1,
        'q2': 4, '2': 4,
        'q3': 7, '3': 7,
        'q4': 10, '4': 10
    }
    
    # Academic term mapping (start month)
    TERM_MAP = {
        'fall semester': 9, 'spring semester': 1, 'summer semester': 6, 'winter semester': 12,
        'fall quarter': 9, 'spring quarter': 3, 'summer quarter': 6, 'winter quarter': 12,
        'fall term': 9, 'spring term': 1, 'summer term': 6, 'winter term': 12,
        'first semester': 9, 'second semester': 1, 'third semester': 6, 'fourth semester': 12,
        'semester 1': 9, 'semester 2': 1, 'semester 3': 6, 'semester 4': 12,
        'trimester 1': 1, 'trimester 2': 5, 'trimester 3': 9
    }
    
    # Half-year mapping
    HALF_MAP = {
        'h1': 1, 'h2': 7
    }
    
    # ============= PREPROCESSING =============
    
    # Check if ongoing
    is_ongoing = any(word in date_string.lower() for word in 
                     ['present', 'now', 'current', 'ongoing', 'till date', 'to date', 
                      'continuing', 'today', 'nowadays', 'currently', 'presently', 
                      'still', 'active', 'in progress', 'in process', 'underway', 
                      'in operation'])
    
    # Clean up date string
    date_clean = date_string.strip()
    
    # Remove duration in parentheses (e.g., "(10 years)")
    date_clean = re.sub(r'\(\s*\d+[\s]*(?:Years?|Yrs?|Months?|Mos?)\s*\)', '', date_clean)
    
    # Remove parentheses/brackets/braces
    date_clean = re.sub(r'[\(\)\[\]\{\}«»‹›]', ' ', date_clean)
    
    # Replace newlines with spaces
    date_clean = date_clean.replace('\n', ' ').replace('\\n', ' ')
    
    # Remove line numbers (e.g., "52. July 2013" -> "July 2013")
    date_clean = re.sub(r'^\d+\.\s*', '', date_clean)
    date_clean = re.sub(r'\s+\d+\.\s+', ' ', date_clean)
    
    # Normalize separators
    date_clean = date_clean.replace('–', '-').replace('—', '-').replace('−', '-')
    date_clean = date_clean.replace('‐', '-').replace('‑', '-').replace('‒', '-')
    date_clean = date_clean.replace('―', '-').replace('⁻', '-').replace('₋', '-')
    date_clean = date_clean.replace('﹣', '-').replace('－', '-')
    
    # Multiple spaces to single space
    date_clean = re.sub(r'\s+', ' ', date_clean).strip()
    
    # ============= SPLIT DATE RANGE =============
    
    date_parts = None
    separators = [
        ' - ', ' – ', ' — ', ' − ', ' to ', ' till ', ' until ', 
        ' through ', ' thru ', ' throughout ', ' and ', ' & ',
        '-', '–', '—', '/', '\\', ',', ';', '.'
    ]
    
    for sep in separators:
        if sep in date_clean:
            parts = date_clean.split(sep, 1)
            if len(parts) == 2:
                date_parts = [parts[0].strip(), parts[1].strip()]
                break
    
    if not date_parts or len(date_parts) != 2:
        print(f"⚠️  Warning: Date string '{date_string}' is not a valid range (no separator found)")
        return {
            'start_date': None,
            'end_date': None,
            'duration_months': 0,
            'duration_years': 0.0,
            'duration_display': 'Invalid date range',
            'is_ongoing': False
        }
    
    # ============= FLEXIBLE DATE PARSER =============
    
    def parse_flexible_date(date_str):
        """
        Parse ANY date format from your regex patterns.
        Returns datetime object or None.
        """
        date_str = date_str.strip().lower()
        
        # Remove line numbers again
        date_str = re.sub(r'^\d+\.\s*', '', date_str)
        
        # Remove quotes/backticks
        date_str = date_str.replace('"', '').replace("'", '').replace(''', '')
        date_str = date_str.replace(''', '').replace('`', '').replace('´', '')
        
        # ===== PATTERN 1: Month name + Year =====
        # Matches: "July 2024", "Jan'2023", "Aug'19", "March '22", "Nov. 2021"
        # Also handles: "July, 2024", "July 2024", "July'2024"
        match = re.search(r'([a-z]+)\.?\s*[,\'\s]*(\d{2,4})', date_str, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            year_str = match.group(2)
            
            # Convert 2-digit year to 4-digit
            if len(year_str) == 2:
                year_num = int(year_str)
                # Assume 20xx for years 00-50, 19xx for years 51-99
                year = 2000 + year_num if year_num <= 50 else 1900 + year_num
            else:
                year = int(year_str)
            
            if month_str in MONTH_MAP:
                month = MONTH_MAP[month_str]
                return datetime(year, month, 1)
        
        # ===== PATTERN 2: Season + Year =====
        # Matches: "Spring 2023", "Fall 2023", "Summer 2022"
        for season_name, season_month in SEASON_MAP.items():
            pattern = rf'\b{season_name}\s*(\d{{4}})'
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                return datetime(year, season_month, 1)
        
        # ===== PATTERN 3: Academic Terms + Year =====
        # Matches: "Fall Semester 2023", "Spring Quarter 2024"
        for term_name, term_month in TERM_MAP.items():
            pattern = rf'\b{term_name}\s*(\d{{4}})'
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                return datetime(year, term_month, 1)
        
        # ===== PATTERN 4: Quarter + Year =====
        # Matches: "Q1 2023", "Q2-2024", "1st Quarter 2023"
        match = re.search(r'\bq?([1-4])(?:st|nd|rd|th)?\s*(?:quarter)?\s*[,-]?\s*(\d{4})', date_str, re.IGNORECASE)
        if match:
            quarter = match.group(1)
            year = int(match.group(2))
            month = QUARTER_MAP[quarter]
            return datetime(year, month, 1)
        
        # ===== PATTERN 5: Half-year + Year =====
        # Matches: "H1 2023", "H2 2024"
        match = re.search(r'\bh([1-2])\s*(\d{4})', date_str, re.IGNORECASE)
        if match:
            half = f'h{match.group(1)}'
            year = int(match.group(2))
            month = HALF_MAP[half]
            return datetime(year, month, 1)
        
        # ===== PATTERN 6: Numeric MM/YYYY =====
        # Matches: "01/2015", "04/2018", "6/2020"
        match = re.search(r'\b(\d{1,2})[/\\](\d{4})\b', date_str)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12:
                return datetime(year, month, 1)
        
        # ===== PATTERN 7: Numeric MM-YYYY =====
        # Matches: "01-2015", "04-2018"
        match = re.search(r'\b(\d{1,2})-(\d{4})\b', date_str)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12:
                return datetime(year, month, 1)
        
        # ===== PATTERN 8: Numeric MM.YYYY =====
        # Matches: "01.2015", "04.2018"
        match = re.search(r'\b(\d{1,2})\.(\d{4})\b', date_str)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12:
                return datetime(year, month, 1)
        
        # ===== PATTERN 9: Full date with day =====
        # Matches: "March 15, 2023", "15 March 2023", "March 15th, 2023"
        # Try: Day Month Year
        match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s*,?\s*(\d{4})', date_str, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month_str = match.group(2).lower()
            year = int(match.group(3))
            if month_str in MONTH_MAP and 1 <= day <= 31:
                month = MONTH_MAP[month_str]
                return datetime(year, month, day)
        
        # Try: Month Day Year
        match = re.search(r'\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})', date_str, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3))
            if month_str in MONTH_MAP and 1 <= day <= 31:
                month = MONTH_MAP[month_str]
                return datetime(year, month, day)
        
        # ===== PATTERN 10: Numeric date formats =====
        # DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY
        match = re.search(r'\b(\d{1,2})[./\\-](\d{1,2})[./\\-](\d{4})\b', date_str)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            year = int(match.group(3))
            
            # Ambiguous: could be DD-MM or MM-DD
            # Heuristic: if first > 12, it must be day
            if first > 12:
                day, month = first, second
            elif second > 12:
                month, day = first, second
            else:
                # Assume MM-DD (US format)
                month, day = first, second
            
            if 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day)
        
        # ===== PATTERN 11: YYYY/MM/DD (ISO-like) =====
        match = re.search(r'\b(\d{4})[/\\-](\d{1,2})[/\\-](\d{1,2})\b', date_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day)
        
        # ===== PATTERN 12: Year only (YYYY) =====
        # Matches: "2020", "2021"
        match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if match:
            year = int(match.group(0))
            return datetime(year, 1, 1)  # Default to January 1st
        
        return None
    
    # ============= PARSE START AND END DATES =============
    
    try:
        # Parse start date
        start_date = parse_flexible_date(date_parts[0])
        if not start_date:
            raise ValueError(f"Could not parse start date from '{date_parts[0]}'")
        
        # Parse end date or use "Present"
        if is_ongoing:
            end_date = datetime.now()
            end_date_str = "Present"
        else:
            end_date = parse_flexible_date(date_parts[1])
            if not end_date:
                raise ValueError(f"Could not parse end date from '{date_parts[1]}'")
            end_date_str = end_date.strftime("%B %Y")
        
        # ============= CALCULATE DURATION =============
        
        # Calculate difference
        delta = relativedelta(end_date, start_date)
        
        # Total months (inclusive - add 1 for the starting month)
        total_months = delta.years * 12 + delta.months + 1
        
        # Ensure minimum of 1 month
        if total_months < 1:
            total_months = 1
        
        # Years as decimal (e.g., 1.5 years = 18 months / 12)
        duration_years = round(total_months / 12, 1)
        
        # ============= HUMAN-READABLE DISPLAY =============
        
        years = delta.years
        months = delta.months + 1  # Inclusive counting
        
        # Adjust if months >= 12
        if months >= 12:
            years += months // 12
            months = months % 12
        
        # Build display string
        if years > 0 and months > 0:
            duration_display = f"{years} year{'s' if years > 1 else ''} {months} month{'s' if months > 1 else ''}"
        elif years > 0:
            duration_display = f"{years} year{'s' if years > 1 else ''}"
        elif months > 0:
            duration_display = f"{months} month{'s' if months > 1 else ''}"
        else:
            duration_display = "Less than 1 month"
        
        return {
            'start_date': start_date.strftime("%B %Y"),
            'end_date': end_date_str,
            'duration_months': total_months,
            'duration_years': duration_years,
            'duration_display': duration_display,
            'is_ongoing': is_ongoing
        }
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to parse date range '{date_string}': {e}")
        return {
            'start_date': None,
            'end_date': None,
            'duration_months': 0,
            'duration_years': 0.0,
            'duration_display': 'Parse error',
            'is_ongoing': False
        }


def print_triplets_with_descriptions(triplets_with_descriptions: List[Dict]):
    """
    Pretty print triplets with their COMPLETE descriptions from raw text.
    
    Args:
        triplets_with_descriptions: Triplets with descriptions mapped
    """
    print(f"\n{'='*80}")
    print("TRIPLETS WITH DESCRIPTIONS")
    print(f"{'='*80}")
    
    for triplet in triplets_with_descriptions:
        print(f"\nTriplet {triplet['id']}:")
        print(f"  Position: {triplet['Position']}")
        print(f"  Company:  {triplet['Company']}")
        print(f"  Year:     {triplet['Year']}")
        
        # Show line positions
        line_pos = triplet['line_positions']
        print(f"  Line Range: {line_pos['min_line']} - {line_pos['max_line']}")
        
        # Show COMPLETE descriptions (no truncation)
        if triplet['descriptions']:
            print(f"  Descriptions ({len(triplet['descriptions'])} lines):")
            for desc in triplet['descriptions']:
                # Print COMPLETE text without truncation
                print(f"    {desc['text']}")
        else:
            print(f"  Descriptions: (None found)")
        
        print(f"  {'-'*76}")
    
    print(f"\n{'='*80}")
    print(f"Total triplets: {len(triplets_with_descriptions)}")
    print(f"{'='*80}")


def print_detailed_view(triplets_with_descriptions: List[Dict], max_triplets: int = 100):
    """
    Print detailed view of first N triplets with FULL descriptions.
    
    Args:
        triplets_with_descriptions: Triplets with descriptions
        max_triplets: Number of triplets to show in detail (default: 3)
    """
    print(f"\n{'='*80}")
    print(f"DETAILED VIEW - FIRST {max_triplets} TRIPLETS WITH FULL DESCRIPTIONS")
    print(f"{'='*80}")
    
    for triplet in triplets_with_descriptions[:max_triplets]:
        print(f"\n{''*80}")
        print(f"TRIPLET {triplet['id']}")
        print(f"{''*80}")
        print(f"Position: {triplet['Position']}")
        print(f"Company:  {triplet['Company']}")
        print(f"Year:     {triplet['Year']}")
        print()
        
        line_pos = triplet['line_positions']
        print("Line Positions:")
        print(f"  Position found at line: {line_pos['position_line']}")
        print(f"  Company found at line:  {line_pos['company_line']}")
        print(f"  Date found at line:     {line_pos['date_line']}")
        print(f"  Context range: lines {line_pos['min_line']}-{line_pos['max_line']}")
        print()
        
        # Print COMPLETE descriptions
        if triplet['descriptions']:
            print(f"{''*80}")
            print(f"DESCRIPTION ({len(triplet['descriptions'])} lines):")
            print(f"{''*80}")
            for desc in triplet['descriptions']:
                # Print line number and COMPLETE text
                print(f"{desc['line_num']}. {desc['text']}")
        else:
            print("No descriptions found.")
        
        print(f"{''*80}\n")


# ==================== INTEGRATION FUNCTION ====================

def integrate_description_mapping(
    raw_text: str,
    tagging_result: Dict,
    triplets: List[Dict],
    duos: List[Dict] = None,
    section_keywords: Dict[str, List[str]] = None
) -> List[Dict]:
    """
    Complete pipeline to map descriptions to triplets using RAW TEXT.
    """
    print(f"\n{'='*80}")
    print("DESCRIPTION MAPPING PIPELINE V3")  # ✅ Updated version
    print("Strategy: Map text between triplets, EXCLUDING current triplet's entities")
    print(f"{'='*80}")
    
    # ✅ ENHANCED: Default section keywords
    if section_keywords is None:
        section_keywords = {
            'education_sections': [
                'education', 'educational background', 'academic qualifications',
                'education and training', 'academic background', 'educational qualifications',
                'degrees', 'degree', 'academic credentials', 'schooling', 
                'courses and certifications', 'training and education', 'formal education',
                'higher education', 'university education', 'college education',
                'professional education', 'education details', 'academic achievements',
                'studies', 'qualifications'
            ],
            'avoid_sections': [
                'work experience', 'projects', 'skills', 'experience', 'achievements',
                'certifications', 'awards', 'publications', 'contact', 'summary',
                'objective', 'profile', 'employment', 'employment history',
                'professional experience', 'career history', 'internships',
                'technical skills', 'soft skills', 'languages', 'professional summary',
                'career objective', 'work history', 'volunteer work', 'volunteer experience',
                'research experience', 'professional development', 'training',
                'conferences', 'workshops', 'seminars', 'references', 'personal details',
                'hobbies', 'interests', 'extracurricular activities', 'patents',
                'leadership', 'relevant experience', 'freelance experience',
                'entrepreneurial experience', 'teaching experience', 'industry experience',
                'client projects', 'professional affiliations', 'memberships', 'clubs',
                'honors', 'scholarships', 'grants', 'fellowships', 'professional licenses',
                'academic projects', 'thesis', 'dissertation', 'presentations',
                'technical expertise', 'core competencies', 'key competencies',
                'areas of expertise', 'career highlights', 'notable achievements',
                'professional goals', 'relevant skills', 'top_section', 'unassigned', 
                'research projects',
                # ✅ ADDED: Missing section markers
                'prior experience', 'previous experience', 'previous employment',
                'past experience', 'past employment', 'former positions'
            ]
        }
    
    # [Rest of function remains the same...]
    print("\nStep 1: Extracting line numbers from RAW text...")
    line_map = extract_line_numbers_from_text(raw_text)
    print(f"   Found {len(line_map)} numbered lines in raw text")
    
    print("\nStep 2: Finding line positions for triplets IN RAW TEXT...")
    triplets_with_positions = find_triplet_line_positions(
        raw_text, triplets, line_map
    )
    print(f"   Mapped positions for {len(triplets_with_positions)} triplets")
    
    duos_with_positions = None
    if duos:
        print("\nStep 3: Finding line positions for DUOs IN RAW TEXT...")
        duos_with_positions = find_triplet_line_positions(
            raw_text, duos, line_map
        )
        print(f"   Mapped positions for {len(duos_with_positions)} duos")
    
    print("\nStep 4: Mapping descriptions from ORIGINAL RAW TEXT...")
    triplets_with_descriptions = map_descriptions_to_triplets(
        original_raw_text=raw_text,
        tagging_result=tagging_result,
        triplets_with_positions=triplets_with_positions, 
        line_map=line_map,
        duos_with_positions=duos_with_positions,
        window_before=0,
        window_after=50,
        section_keywords=section_keywords
    )
    
    print(f"\n{'='*80}")
    
    return triplets_with_descriptions

#=================================================================================================================================


def print_desc_tag_debug(raw_text: str, tagging_result: Dict) -> None:
    tagged_lines = tagging_result.get('tagged_lines', [])

    sep = '=' * 80
    print(f"\n{sep}")
    print("  DESC TAG DEBUG — SENTENCE LEVEL (LSTM PREDICTIONS)")
    print(f"{sep}")
    print(f"  {'#':>5}  {'TAG':<12}  TEXT")
    print(f"  {'----':>5}  {'---':<12}  ----")

    for entry in tagged_lines:
        tag_str = f"[{entry['tag']}]"
        print(f"  {entry['line_num']:>5}  {tag_str:<12}  {entry['text']}")

    print(f"{sep}\n")


def main():
    """Main testing function with description mapping"""
    
    # ===== CONFIGURATION =====
    MODEL_PATH = r"C:\Users\bhanu\OneDrive\Desktop\ai mock interview\outputs\outputs\final_model"
    
    # MongoDB Configuration
    MONGODB_CONNECTION = "mongodb://Deep:<db_password>@43.239.92.101:27017/Experience?authSource=admin"
    MONGODB_PASSWORD = "deep12345"
    
    # ===== OPTIONS =====
    USE_REGEX_DATES = True
    USE_POSITION_MATCHING = True
    
    # ===== PASTE YOUR TEXT HERE =====
    RAW_TEXT = """


======================================================
  PAGE 1
======================================================
Chintan K
, A
Manager
[EDUCATION]
Masters in Data Science
& Artificial Intelligence
Campbellsville
University, USA
Masters in Information
Technology & Analyt ics
Rutgers University Newark,
New Jersey, USA
Bach elor of Engineering in
E lectronics &
T elecommunications
D.J. Sanghvi College of
Engineering, Mumbai,
India
[SKILLS]
Programming Languages:
SQL, Python, R, HTML
IDEs:
Anaconda Navigator
(Spyder, Jupyter
Notebook), PyCharm,
R studio
Visualization tools:
Power BI, Tableau, MS
Excel
Cloud Technologies:
Microsoft Azure, AWS
Database:
MS SQL Server, MySQL,
Oracle DB
Version Control/Other
Tools:
GitHub, MSOffice
(PowerPoint, Word,
Access), SharePoint
[CERTIFICATIONS]
• AWS Certified Solutions
Architect Associate
• Data Camp - Data
Analyst with Python,
SQL Server,
Fundamental of
Tableau, Spreadsheet,
Power BI, Data
Visualiza tion with
Python, Python Data
Scientist

Ketan Mody (347)614-6591
chintanmody95@gmail.com
Analytics
Georgia
[SUMMARY]
• Strategic and results-driven Analytics Manager with a strong foundation in analytical thinking and data
e
storytelling. Adept at collaborating with technical teams and business stakeholders to translate complex business
challenges into data-driven solutions. Proven ability to lead analytics initiatives that drive operational efficiency,
strategic insights, and measurable business impact.
• Proficient in writing complex SQL queries, Stored procedures, functions, packages, tables, views, and triggers
using relational databases like MS SQL Server, MySQL, Oracle DB.
• Experienced in building and maintaining Microsoft Power BI and Tableau reports and dashboards and publishing
to the end users for Executive-level Business Decisions.
• Proficient in MS Excel, including Pivot tables, charts, and dashboard for day-to-day data analysis and statistics.
• Skilled in Python-based environment, along with data analytics, data wrangling, and libraries like Pandas,
NumPy, Matplotlib, and SciPy.
• Good Knowledge of Data warehouse (OLAP, Data Modeling, ETL).
[EXPERIENCE]
FirstKey Homes, Georgia, USA | Mar 2024 – Present| Manager, Analytics
• Developed and maintained executive-level dashboards and KPI reports for monthly and quarterly C-suite
reporting, partnering with the parent company to ensure alignment across business units.
• Led cross-functional data initiatives by gathering requirements, analyzing data, and delivering ad-hoc reporting
solutions; mentored team members to enhance their technical reporting capabilities.
• Identified business trends and seasonal patterns to optimize model parameters and guide strategic decision-
making through actionable insights.
• Built advanced SQL queries and Python scripts to support IT processes, and established Microsoft Azure
cloud infrastructure to automate data workflows and reporting pipelines.
FirstKey Homes, Georgia, USA | Sep 2021 – Mar
2024
| Senior Analyst
• Developed and maintained Revenue Management Pricing Performance Power BI dashboard by writing complex
SQL queries and advanced DAX functions to analyze essential KPIs used for day-to-day pricing reporting.
• Managed weekly pricing rent growth and acquisition meetings to analyze rent growth by market and
recommend pricing model parameters based on demand score.
• Engineered custom pipelines and optimized existing pipelines leveraging Microsoft Azure Factory and Data Lake
to create custom reports in SandBox.
• Collaborated with cross-departmental teams, such as Acquisitions, Vendor Management, Procurement,
Maintenance, and Leasing, to build Ad-Hoc Excel reports and Power BI dashboards by writing custom queries
from core systems like BLTYardi, BLTYardiPre-Yeti, Hub, Homes, FKHEDW, Dynamics.
• Established a comprehensive data framework for revenue management, laying the groundwork for advanced
analytics and reporting.
• Conducted a thorough analysis of Core database systems, identifying and rectifying critical data discrepancies
and gaps to ensure the integrity and reliability of reporting.
• Orchestrated comprehensive training sessions, presentations, and mentorship programs for interns, cross-
departments, and field agents on SQL queries, Power BI dashboard development, and utilization.
Pidilite Ind. Ltd, Mumbai, India | Dec 2017– March 2019 | Data Analyst
• Implemented an EDW Model on Azure SQL Data Warehouse over the Enterprise Data Warehouse hosted in on-
premises Oracle database to consolidate the data across all applications and databases.
• Identified and documented project constraints, assumptions, business impacts, risks, and scope exclusions and
transformed project data requirements into project data models.
• Worked with the ETL team for Data Cleaning and Data Integration from different source systems.
• Collaborated with the Sales and Marketing team to build KPIs dashboards to review business performance.
• Synthesized current business intelligence data to produce reports and polished presentations, highlighting
findings and recommending changes using Power BI.
• Used advanced Excel functions to generate spreadsheets and created dashboards in Excel for data reporting.
• Created SharePoint workflows for a centralized Content Management System (CMS).
• Managed an approval network cycle to resolve employee issues using Workflow.

   
"""
    
    print("\n" + "=" * 80)
    print("DeBERTa BIOES NER Testing Script")
    print("WITH Description Mapping to Triplets")
    print("=" * 80)
    
    # ===== INITIALIZE TESTER =====
    print("\nLoading model...")
    tester = NERTester(
        MODEL_PATH, 
        use_regex_dates=USE_REGEX_DATES,
        use_position_matching=USE_POSITION_MATCHING,
        mongodb_connection=MONGODB_CONNECTION,
        mongodb_password=MONGODB_PASSWORD
    )
    
    # ===== PROCESS TEXT =====
    print(f"\nProcessing resume text...")
    result = tester.predict_with_debug(
        RAW_TEXT,
        show_confidence=True,
        confidence_threshold=0.3,
        lstm_batch_size=64,  # ✅ ADD THIS
        ner_batch_size=4 
    )
    
    # Print standard NER results
    tester.print_debug_results(result, min_confidence=0.3)
    
    # ===== EXTRACT ENTITIES =====
    all_companies = [c for c in result['predictions']['Company'] 
                     if not re.fullmatch(r'[\d\W]+', c.strip())]
    all_positions = [p for p in result['predictions']['Position'] 
                     if not re.fullmatch(r'[\d\W]+', p.strip())]
    all_dates = result['predictions']['Date']
    
    print(f"\n{'='*80}")
    print("ENTITY EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"  Companies: {len(all_companies)}")
    print(f"  Positions: {len(all_positions)}")
    print(f"  Dates: {len(all_dates)}")
    
    # **FIX: Check if we have dates before proceeding**
    if not all_dates:
        print(f"\n⚠️ WARNING: No dates found - cannot form experience triplets")
        print(f"{'='*80}")
        print("PROCESSING STOPPED - No dates available for triplet formation")
        print(f"{'='*80}")
        return  # Exit early
    
    # ===== FORM TRIPLETS =====
    tokens = result['tokens']
    text_for_ner = result['text']                    # clean, for NER ops
    text_for_duos = result['text']  

    triplets, used_dates = tester._form_triplets_by_date_pivot(
        tokens=tokens,
        text=text_for_ner,       # NER token space — keep clean
        companies=all_companies,
        positions=all_positions,
        dates=all_dates,
        window_size=12
    )

    print(f"\n{'='*80}")
    print("TRIPLET FORMATION")
    print(f"{'='*80}")
    print(f"Formed {len(triplets)} triplets using {len(used_dates)} dates")

    # **FIX: Check if we have triplets before forming DUOs**
    if len(triplets) == 0:
        print(f"\n⚠️ WARNING: No triplets formed - skipping DUO formation and description mapping")
        print(f"{'='*80}")
        print("PROCESSING COMPLETE - No experience entries extracted")
        print(f"{'='*80}")
        return  # Exit early

    # Form DUOs from unused dates
    duos = tester._form_duos_from_unused_dates(
        tokens=tokens,
        text=text_for_duos,      
        positions=all_positions,
        dates=all_dates,
        used_dates=used_dates,
        triplets=triplets,
        window_size=10
    )

    # ✅ ADD THIS SECTION HERE (REPLACE THE EXISTING "Combine for description mapping" line)
    
    # NEW: Filter education-related entries BEFORE combining
    print("\n🎓 Filtering education-related entries...")
    education_keywords = get_education_keywords()
    
    triplets, duos, triplet_removed, duo_removed = filter_education_from_combined_triplets(
        triplets=triplets,
        duos=duos,
        education_keywords=education_keywords
    )
    
    print(f"\n📊 Education Filtering Summary:")
    print(f"   Triplets removed: {triplet_removed}")
    print(f"   DUOs removed: {duo_removed}")
    print(f"   Total removed: {triplet_removed + duo_removed}")
    
    # Combine for description mapping
    all_triplets = triplets + duos

    # NEW: Filter out triplets with stopwords in positions
    print("\n🔍 Filtering triplets by stopwords...")
    stopwords = {
                'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'any', 'are', 
                'aren’t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
                'but', 'by', 'can', 'can’t', 'cannot', 'could', 'couldn’t', 'did', 'didn’t', 'do', 'does', 
                'doesn’t', 'doing', 'don’t', 'down', 'during', 'each', 'few', 'from', 'further', 
                'had', 'hadn’t', 'has', 'hasn’t', 'have', 'haven’t', 'having', 'he', 'he’d', 'he’ll', 'he’s', 
                'her', 'here', 'here’s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how’s',  
                'i’d', 'i’ll', 'i’m', 'i’ve', 'if', 'in', 'into', 'is', 'isn’t', 'it’s', 'its', 'itself', 
                'let’s', 'me', 'more', 'most', 'mustn’t', 'my', 'myself', 'no', 'nor', 'not', 'off', 
                'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
                'own', 'same', 'she', 'she’d', 'she’ll', 'she’s', 'should', 'shouldn’t', 'so', 'some', 
                'such', 'than', 'that', 'that’s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
                'there', 'there’s', 'these', 'they', 'they’d', 'they’ll', 'they’re', 'they’ve', 'this', 
                'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn’t', 'we', 
                'we’d', 'we’ll', 'we’re', 'we’ve', 'were', 'weren’t', 'what', 'what’s', 'when', 'when’s', 
                'where', 'where’s', 'which', 'while', 'who', 'who’s', 'whom', 'why', 'why’s', 'with', 
                'won’t', 'would', 'wouldn’t', 'you', 'you’d', 'you’ll', 'you’re', 'you’ve', 'your', 
                'yours', 'yourself', 'yourselves','certification','certifications','certificate','certificates'

            }
    all_triplets = filter_triplets_by_stopwords(all_triplets, stopwords)
    print(f"✅ Remaining triplets after filtering: {len(all_triplets)}")

    if not all_triplets:
        print("⚠ No triplets or DUOs remaining after stopword filtering")
        return

    print(f"\n{'='*80}")
    print("COMBINED RESULTS")
    print(f"{'='*80}")
    print(f"Total Triplets: {len(triplets)}")
    print(f"Total DUOs: {len(duos)}")
    print(f"Combined Total: {len(all_triplets)}")
    
    # ... rest of the main function continues as before ...
        
    all_section_keywords = {
    'education_sections': [
        'education', 'educational background', 'academic qualifications',
        'education and training', 'academic background', 'educational qualifications',
        'degrees', 'degree', 'academic credentials', 'schooling', 
        'courses and certifications', 'training and education', 'formal education',
        'higher education', 'university education', 'college education',
        'professional education', 'education details', 'academic achievements',
        'studies', 'qualifications'
    ],
    'avoid_sections': [
        'work experience', 'projects', 'skills', 'experience', 'achievements',
        'certifications', 'awards', 'publications', 'contact', 'summary',
        'objective', 'profile', 'employment', 'employment history',
        'professional experience', 'career history', 'internships',
        'technical skills', 'soft skills', 'languages', 'professional summary',
        'career objective', 'work history', 'volunteer work', 'volunteer experience',
        'research experience', 'professional development', 'training',
        'conferences', 'workshops', 'seminars', 'references', 'personal details',
        'hobbies', 'interests', 'extracurricular activities', 'patents',
        'leadership', 'relevant experience', 'freelance experience',
        'entrepreneurial experience', 'teaching experience', 'industry experience',
        'client projects', 'professional affiliations', 'memberships', 'clubs',
        'honors', 'scholarships', 'grants', 'fellowships', 'professional licenses',
        'academic projects', 'thesis', 'dissertation', 'presentations',
        'technical expertise', 'core competencies', 'key competencies',
        'areas of expertise', 'career highlights', 'notable achievements',
        'professional goals', 'relevant skills', 'top_section', 'unassigned','activities'
    ]
}
    

    # ===== MAP DESCRIPTIONS TO TRIPLETS =====
    triplets_with_descriptions = integrate_description_mapping(
        raw_text=RAW_TEXT,
        tagging_result=result['tagging_result'],
        triplets=all_triplets,  # This combines triplets and duos
        duos=duos,
        section_keywords=all_section_keywords
    )
    
    print_desc_tag_debug(
        raw_text=RAW_TEXT,
        tagging_result=result['tagging_result']
    )
    
    # ===== DISPLAY RESULTS =====
    print_triplets_with_descriptions(triplets_with_descriptions)

    # Export detailed format (original)
    detailed_output_file = "triplets_with_descriptions.json"

    def export_triplets_to_json(triplets_with_descriptions, output_file):
        """
        Export triplets with descriptions to a JSON file.
        """
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(triplets_with_descriptions, f, ensure_ascii=False, indent=2)
        print(f" Detailed results exported to {output_file}")

    export_triplets_to_json(triplets_with_descriptions, detailed_output_file)

    #  NEW: Export standard integration format
    standard_output_file = "experience_output.json"
    location_list = result['predictions'].get('Location', [])
    export_experience_to_standard_json(
        triplets_with_descriptions,
        standard_output_file,
        raw_text=RAW_TEXT,
        location_list=location_list
    )
    
    # ===== DETAILED VIEW OF FIRST 3 TRIPLETS =====
    print(f"\n{'='*80}")
    print("DETAILED VIEW - FIRST 3 TRIPLETS WITH FULL DESCRIPTIONS")
    print(f"{'='*80}")
    
    for triplet in triplets_with_descriptions[:100]:
        print(f"\n{''*80}")
        print(f"TRIPLET {triplet['id']}")
        print(f"{''*80}")
        print(f"Position: {triplet['Position']}")
        print(f"Company:  {triplet['Company']}")
        print(f"Year:     {triplet['Year']}")
        print(f"\nLine Positions:")
        print(f"  Position found at line: {triplet['line_positions']['position_line']}")
        print(f"  Company found at line:  {triplet['line_positions']['company_line']}")
        print(f"  Date found at line:     {triplet['line_positions']['date_line']}")
        print(f"  Context range: lines {triplet['line_positions']['min_line']}-{triplet['line_positions']['max_line']}")
        
        print(f"\n{''*80}")
        print(f"DESCRIPTION ({len(triplet['descriptions'])} lines):")
        print(f"{''*80}")
        
        if triplet['description_text']:
            print(triplet['description_text'])
        else:
            print("(No descriptions found)")
        
        print(f"{''*80}\n")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f" Extracted {len(all_companies)} companies, {len(all_positions)} positions, {len(all_dates)} dates")
    print(f" Formed {len(triplets)} triplets")
    print(f" Mapped descriptions to triplets")
    print(f"{'='*80}")
    
    
def export_experience_to_standard_json(
    triplets_with_descriptions: List[Dict],
    output_file: str = "experience_output.json",
    raw_text: str = "",
    location_list: List[str] = None
):
    """
    Export experience data in standard JSON format with duration calculation.
    Adds current_location only for entries with ongoing/present dates.
    """
    import json

    print(f"\n{'='*80}")
    print("EXPORTING EXPERIENCE TO STANDARD JSON FORMAT WITH DURATION")
    print(f"{'='*80}")

    experience_list = []

    for triplet in triplets_with_descriptions:
        position = triplet.get('Position', '')
        company = triplet.get('Company', '')
        date = triplet.get('Year', '')
        description = triplet.get('description_text', '')

        # Calculate duration
        duration_info = calculate_work_duration(date)

        # Determine current_location only for ongoing dates
        current_location = None
        if is_ongoing_date(date) and raw_text and location_list:
            current_location = find_location_near_date(
                date_string=date,
                raw_text=raw_text,
                location_list=location_list,
                window_lines=3
            )
            if current_location:
                print(f"   📍 Current location found: '{current_location}' (near ongoing date '{date}')")
            else:
                print(f"   📍 No location found near ongoing date '{date}'")

        experience_entry = {
            "position": position,
            "company": company,
            "date": date,
            "start_date": duration_info['start_date'],
            "end_date": duration_info['end_date'],
            "duration_years": duration_info['duration_years'],
            "duration_months": duration_info['duration_months'],
            "duration_display": duration_info['duration_display'],
            "is_ongoing": duration_info.get('is_ongoing', False),
            "description": description
        }

        # Only add current_location field if it's an ongoing role
        if is_ongoing_date(date):
            experience_entry["current_location"] = current_location  # None if not found

        experience_list.append(experience_entry)

        print(f"\n ✅ Exporting: {position} at {company}")
        print(f"   📅 Duration: {duration_info['duration_display']} ({duration_info['duration_years']} years)")
        if duration_info.get('is_ongoing'):
            print(f"   🔄 Status: Currently working")

    output_data = {"experience": experience_list}

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*80}")
        print(f" ✅ EXPORT COMPLETE")
        print(f"{'='*80}")
        print(f"Total experience entries: {len(experience_list)}")
        total_years = sum(e['duration_years'] for e in experience_list)
        print(f"Total experience: {round(total_years, 1)} years")
        print(f"Output file: {output_file}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n ❌ ERROR: Failed to write JSON file: {e}")
    
def extract_experience_entities(text: str, model_path: str) -> List[Dict]:
    """
    Public API with DUO formation and proper description mapping with enhanced stopping logic.
    OPTIMIZED: Uses cached NERTester instance for 2.5s speedup per request.
    
    Args:
        text: Resume text to extract from
        model_path: Path to NER model
    
    Returns:
        List of experience entry dictionaries
    """
    print("\n" + "="*80)
    print("EXPERIENCE EXTRACTION STARTING (WITH MONGODB CACHE)")
    print("="*80)
    
    # Normalize newlines at entry point
    text = normalize_text_newlines(text)
    print(f"Text length: {len(text)}, Lines: {text.count(chr(10)) + 1}")
    
    try:
        # MongoDB Configuration
        MONGODB_CONNECTION = "mongodb://Deep:<db_password>@43.239.92.101:27017/Experience?authSource=admin"
        MONGODB_PASSWORD = "deep12345"
        
        print("⚙️ Getting NER system (cached if available)...")
         
        # ✅ CRITICAL CHANGE: Use cached tester instead of creating new one
        tester = get_cached_tester(
            model_path=model_path,
            mongodb_connection=MONGODB_CONNECTION,
            mongodb_password=MONGODB_PASSWORD,
            use_regex_dates=True,
            use_position_matching=True
        )
        
        print("✅ NER system ready (MongoDB data cached)")
        print("📄 Processing resume text...")
        
        # Process text
        result = tester.predict_with_debug(
            text, 
            show_confidence=False,
            lstm_batch_size=64,
            ner_batch_size=4
        )
        print("✅ Text processing complete")
        
        # Extract entities
        print("\n🔍 Extracting entities for triplet formation...")
        all_companies = [c for c in result['predictions'].get('Company', []) 
                        if c and not re.fullmatch(r'[\d\W]+', c.strip())]
        all_positions = [p for p in result['predictions'].get('Position', []) 
                        if p and not re.fullmatch(r'[\d\W]+', p.strip())]
        all_dates = result['predictions'].get('Date', [])
        
        print(f"   📊 Companies (cleaned): {len(all_companies)}")
        print(f"   📊 Positions (cleaned): {len(all_positions)}")
        print(f"   📊 Dates: {len(all_dates)}")
        
        # Check if we have enough entities to form triplets
        if not all_dates:
            print("⚠️ No dates found - cannot form experience triplets")
            print("="*80)
            print("✅ EXPERIENCE EXTRACTION COMPLETE - Found 0 entries (no dates)")
            print("="*80 + "\n")
            return []
        
        if not all_companies and not all_positions:
            print("⚠️ No companies or positions found")
            print("="*80)
            print("✅ EXPERIENCE EXTRACTION COMPLETE - Found 0 entries (no companies/positions)")
            print("="*80 + "\n")
            return []
        
        tokens = result.get('tokens', [])
        text_for_ner = result.get('text', text)
        text_for_duos = result.get('text', text)
        
        # Form triplets
        print("\n🔧 Forming experience triplets...")
        triplets, used_dates = tester._form_triplets_by_date_pivot(
        tokens=tokens,
        text=text_for_ner,
        companies=all_companies,
        positions=all_positions,
        dates=all_dates,
        window_size=15
    )
        
        
        print(f"✅ Formed {len(triplets)} primary triplets")
        
        # Form DUOs from unused dates
        print("\n🔧 Forming DUOs from unused dates...")
        duos = tester._form_duos_from_unused_dates(
        tokens=tokens,
        text=text_for_duos,  # ✅
        positions=all_positions,
        dates=all_dates,
        used_dates=used_dates,
        triplets=triplets,
        window_size=10
    )
        
        print(f"✅ Formed {len(duos)} DUOs")
        
        # Filter education entries
        print("\n🎓 Filtering education-related entries...")
        education_keywords = get_education_keywords()
        
        triplets, duos, triplet_removed, duo_removed = filter_education_from_combined_triplets(
            triplets=triplets,
            duos=duos,
            education_keywords=education_keywords
        )
        
        print(f"📊 Removed {triplet_removed + duo_removed} education entries")
        
        # Combine triplets and duos
        all_triplets = triplets + duos
        
        # Filter out triplets with stopwords in positions
        print("\n🔍 Filtering triplets by stopwords...")
        stopwords = {
            
            'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'any', 'are', 
                'aren’t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
                'but', 'by', 'can', 'can’t', 'cannot', 'could', 'couldn’t', 'did', 'didn’t', 'do', 'does', 
                'doesn’t', 'doing', 'don’t', 'down', 'during', 'each', 'few', 'from', 'further', 
                'had', 'hadn’t', 'has', 'hasn’t', 'have', 'haven’t', 'having', 'he', 'he’d', 'he’ll', 'he’s', 
                'her', 'here', 'here’s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how’s',  
                'i’d', 'i’ll', 'i’m', 'i’ve', 'if', 'in', 'into', 'is', 'isn’t', 'it’s', 'its', 'itself', 
                'let’s', 'me', 'more', 'most', 'mustn’t', 'my', 'myself', 'no', 'nor', 'not', 'off', 
                'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
                'own', 'same', 'she', 'she’d', 'she’ll', 'she’s', 'should', 'shouldn’t', 'so', 'some', 
                'such', 'than', 'that', 'that’s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
                'there', 'there’s', 'these', 'they', 'they’d', 'they’ll', 'they’re', 'they’ve', 'this', 
                'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn’t', 'we', 
                'we’d', 'we’ll', 'we’re', 'we’ve', 'were', 'weren’t', 'what', 'what’s', 'when', 'when’s', 
                'where', 'where’s', 'which', 'while', 'who', 'who’s', 'whom', 'why', 'why’s', 'with', 
                'won’t', 'would', 'wouldn’t', 'you', 'you’d', 'you’ll', 'you’re', 'you’ve', 'your', 
                'yours', 'yourself', 'yourselves','certification','certifications','certificate','certificates'
                 
            }
        
        all_triplets = filter_triplets_by_stopwords(all_triplets, stopwords)
        
        if not all_triplets:
            print("⚠️ No triplets or DUOs remaining after filtering - skipping description mapping")
            print("="*80)
            print("✅ EXPERIENCE EXTRACTION COMPLETE - Found 0 entries")
            print("="*80 + "\n")
            return []
        
        # Add description mapping
        print(f"\n📝 Mapping descriptions to {len(all_triplets)} entries (triplets + DUOs)...")
        
        # Extract line map
        line_map = extract_line_numbers_from_text(text)
        
        # Find line positions for ALL entries (triplets + duos)
        all_triplets_with_positions = find_triplet_line_positions(
            text, all_triplets, line_map
        )
        
        # Map descriptions
        section_keywords = {
            'education_sections': [
                'education', 'educational background', 'academic qualifications',
                'education and training', 'degrees', 'degree', 'academic credentials'
            ],
            'avoid_sections': [
                'work experience', 'projects', 'skills', 'experience', 'achievements',
                'certifications', 'awards', 'publications', 'contact', 'summary',
                'objective', 'profile', 'relevant skills', 'top_section', 'unassigned'
            ]
        }
        
        triplets_with_descriptions = map_descriptions_to_triplets(
            original_raw_text=text,
            tagging_result=result.get('tagging_result', {}),
            triplets_with_positions=all_triplets_with_positions,
            line_map=line_map,
            duos_with_positions=None,
            window_before=0,
            window_after=50,
            section_keywords=section_keywords
        )
        
        print(f"✅ Descriptions mapped successfully")
        
        # Convert to expected format
        experience_entries = []
        for i, triplet in enumerate(triplets_with_descriptions, 1):
            description_text = triplet.get('description_text', '')
            if not description_text and triplet.get('descriptions'):
                description_text = ' '.join([d['text'].strip() for d in triplet['descriptions']])
            
            # Determine entry type (TRIPLET or DUO)
            entry_id = triplet.get('id', i)
            is_duo = isinstance(entry_id, str) and entry_id.startswith('DUO_')
            
            experience_entries.append({
                'id': entry_id,
                'type': 'DUO' if is_duo else 'TRIPLET',
                'position': triplet.get('Position', ''),
                'company': triplet.get('Company', ''),
                'year': triplet.get('Year', ''),
                'description': description_text,
                'line_positions': triplet.get('line_positions', {})
            })
            
            desc_preview = description_text[:100] + "..." if len(description_text) > 100 else description_text
            entry_label = f"{'DUO' if is_duo else 'TRIPLET'} {entry_id}"
            print(f"   {i}. [{entry_label}] {triplet.get('Position', 'N/A')} at {triplet.get('Company', 'N/A')} ({triplet.get('Year', 'N/A')})")
            if description_text:
                print(f"      Description: {desc_preview}")
        
        print("="*80)
        print(f"✅ EXPERIENCE EXTRACTION COMPLETE (MONGODB CACHED)")
        print(f"   Primary Triplets: {len(triplets)}")
        print(f"   DUOs: {len(duos)}")
        print(f"   Total Entries: {len(experience_entries)}")
        print("="*80 + "\n")
        
        # Export to standard JSON format automatically
        # ✅ MODIFIED: Export to standard JSON format with duration calculation
        try:
            entries_for_export = []
            for entry in experience_entries:
                export_entry = {
                    'Position': entry.get('position', ''),
                    'Company': entry.get('company', ''),
                    'Year': entry.get('year', ''),
                    'description_text': entry.get('description', '')
                }
                entries_for_export.append(export_entry)
            
            location_list = result['predictions'].get('Location', [])
            export_experience_to_standard_json(
                entries_for_export,
                "experience_output.json",
                raw_text=text,
                location_list=location_list
            )
        except Exception as e:
            print(f"Warning: Could not export to JSON: {e}")
            import traceback
            traceback.print_exc()
        
        return experience_entries
        
    except Exception as e:
        print(f"\n❌ ERROR in experience extraction: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return []


if __name__ == "__main__":
    main()