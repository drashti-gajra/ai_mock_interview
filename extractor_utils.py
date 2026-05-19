"""
extractor_utils.py
==================
Shared utilities for resume skill extraction:
  - Resume text loading (PDF or plain text)
  - Skill text normalization
  - Noise filtering
  - Console + JSON output
"""

import re
import json
import os
from typing import List, Dict, Any


# ── Text Loading ────────────────────────────────────────────────────────────

def load_jd_text(jd_input: str) -> str:
    """
    Load job description text from a file path or return raw pasted text.

    Args:
        jd_input: Either a file path (.pdf/.txt) or raw JD text pasted directly.

    Returns:
        JD text as a single string.
    """
    # If it looks like a valid file path and exists, load it as a file
    if os.path.exists(jd_input):
        return load_resume_text(jd_input)
    # Otherwise treat it as raw pasted text
    return jd_input.strip()


def _fix_pdf_camelcase_merges(text: str) -> str:
    """
    Fix pdfminer layout artifacts where adjacent words from multi-column
    bullet lists get concatenated without a space, e.g. 'MongoDBExcel'.

    Splits on the pattern: uppercase letter that is PRECEDED by another
    uppercase and FOLLOWED by a lowercase letter — i.e. the start of a new
    TitleCase word after an acronym-capped segment.
      'MongoDBExcel' → 'MongoDB Excel'   (B→E: upper preceded E, E followed by x)
      'IBMCloud'     → 'IBM Cloud'
      'MySQLServer'  → 'MySQL Server'
    Does NOT split legitimate single tech words:
      'JavaScript', 'TypeScript', 'PowerShell', 'GitHub', 'OpenEdge', 'PostgreSQL'
    Only applied to tokens of 8+ chars to avoid short false-positives.
    """
    def split_token(token: str) -> str:
        if len(token) < 8:
            return token
        # Split before uppercase that follows uppercase and precedes lowercase
        # e.g. MongoD|BExcel → D precedes B which precedes E which precedes x
        # Pattern: (?<=[A-Z])(?=[A-Z][a-z]) — insert space before TitleCase start
        # after an uppercase sequence
        result = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', token)
        return result

    return ' '.join(split_token(t) for t in text.split())


def load_resume_text(path: str) -> str:
    """
    Load resume text from a PDF or plain-text file.

    Args:
        path: Absolute or relative path to the resume file (.pdf or .txt)

    Returns:
        Extracted text as a single string.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(path)
            if text and text.strip():
                return _fix_pdf_camelcase_merges(text.strip())
        except ImportError:
            pass

        # Fallback: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        except ImportError:
            raise ImportError(
                "PDF parsing requires either 'pdfminer.six' or 'pypdf'. "
                "Install with: pip install pdfminer.six"
            )

    elif ext in (".txt", ".md", ""):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    else:
        # Try reading as plain text anyway
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()


# ── Skill Text Normalization ─────────────────────────────────────────────────

def normalize_skill_text(text: str) -> str:
    return text.strip()


# ── Noise Filter ─────────────────────────────────────────────────────────────

def passes_noise_filter(phrase: str, doc, label_allowlist: set = None) -> bool:
    """
    Return False if the phrase looks like noise (common English word,
    determiner-led, verb-led, adjective-led, or role-like phrase).

    Args:
        phrase: Normalized skill phrase
        doc: spaCy Doc object of the full resume text
        label_allowlist: Optional set of known label names (lowercased).
                         If the phrase matches a known label, skip strict checks.
    """
    words = phrase.split()
    phrase_lower = phrase.lower()

    # Reject organizational units / metrics that aren't skills
    if phrase_lower in {"pmo", "kpi", "okr", "roi", "sla"}:
        return False

    # If phrase exactly matches a known label, trust it
    if label_allowlist and phrase_lower in label_allowlist:
        return True

    # If label_allowlist is provided and phrase is NOT a known label,
    # apply strict mode: only accept acronyms or tech-formatted terms.
    # This prevents noise from slipping through when POS tags are unreliable.
    if label_allowlist and phrase_lower not in label_allowlist:
        # Reject phrases containing digits (metrics, quantities, dates)
        if any(c.isdigit() for c in phrase):
            return False
        # Allow short all-uppercase acronyms (e.g., OLAP, SSIS, SOAR)
        if len(words) == 1 and phrase.isupper() and len(phrase) <= 6:
            return True
        # Allow single words with / + # (e.g., CI/CD, C++, C#)
        # Only / + # are reliable tech indicators; exclude . - _ as they
        # appear in abbreviations (Sr.), hyphenated words, and filenames
        if any(c in phrase for c in '/+#'):
            return True
        return False

    if len(words) > 4:
        return False

    first_lower = words[0].lower()
    if first_lower in ('the', 'a', 'an', 'this', 'that', 'these', 'those',
                       'my', 'our', 'their', 'its', 'his', 'her'):
        return False

    # Reject verb-led multi-word phrases (gerund VBG, past tense VBD, participle VBN)
    if len(words) >= 2:
        for token in doc:
            if token.text.lower() == first_lower:
                if token.tag_ in ('VBG', 'VBD', 'VBN'):
                    return False
                break

    # Reject adjective-led multi-word phrases (e.g. "complex business problems",
    # "executive decisions", "seasonal patterns", "custom pipelines")
    if len(words) >= 2:
        for token in doc:
            if token.text.lower() == first_lower:
                if token.pos_ == 'ADJ':
                    return False
                break

    if len(words) == 1:
        # Short acronyms and symbol-containing words always pass
        if phrase.isupper() and len(phrase) <= 6:
            return True
        if any(c in phrase for c in '/-_.+#'):
            return True

        # Reject single-word past-tense verbs / participles (e.g. "Mentored")
        for token in doc:
            if token.text.lower() == phrase_lower:
                if token.tag_ in ('VBD', 'VBN', 'VBG', 'VB', 'VBP', 'VBZ'):
                    return False
                break

        # Reject very common English words that aren't proper nouns
        for token in doc:
            if token.text.lower() == phrase_lower:
                is_common_english = -15 < token.prob < -7
                is_proper_noun = token.pos_ == 'PROPN'
                if is_common_english and not is_proper_noun and not phrase[0].isupper():
                    return False
                break

    if len(words) >= 2:
        has_tech_word = False
        all_common = True

        for word in words:
            w_lower = word.lower()
            if any(c in word for c in '/-_.+#'):
                has_tech_word = True
                break

            found_in_doc = False
            for token in doc:
                if token.text.lower() == w_lower:
                    found_in_doc = True
                    if token.pos_ == 'PROPN':
                        has_tech_word = True
                    if token.prob < -15 or not token.has_vector:
                        all_common = False
                    if token.prob >= -7:
                        all_common = False
                    break

            if not found_in_doc:
                all_common = False
            if has_tech_word:
                break

        if all_common and not has_tech_word:
            tech_subs = ('sql', 'db', 'js', 'ml', 'ai', 'api', 'ci',
                         'cd', 'os', 'http', 'ssh', 'tcp', 'udp')
            if not any(ts in phrase_lower for ts in tech_subs):
                return False

    return True


# ── Output ───────────────────────────────────────────────────────────────────

TIER_NAMES = {1: "Technology", 2: "Marketing", 3: "Sales", 4: "Finance & Accounting"}


def save_output(
    skills: List[Dict[str, Any]],
    domain: str,
    tier: int,
    resume_path: str,
    output_json_path: str = "skills_output.json",
    detection_method: str = "auto-detected",
) -> None:
    """
    Print a formatted table to console and save results as JSON.

    Args:
        skills: List of skill dicts from extractor.extract()
        domain: Domain string (e.g. "technology")
        tier: Tier number (1, 2, 3)
        resume_path: Path to original resume file
        output_json_path: Where to save the JSON output
        detection_method: "auto-detected" or "manual"
    """
    tier_label = TIER_NAMES.get(tier, f"Tier {tier}")

    # ── Classify into 3 categories ──
    hard_skills = [s for s in skills if s.get("category") == "hard_skill"]
    concepts    = [s for s in skills if s.get("category") == "concept"]
    soft_skills = [s for s in skills if s.get("category") == "soft_skill"]

    # Fallback: skills without a "category" field default to hard_skill
    uncategorized = [s for s in skills if "category" not in s]
    hard_skills.extend(uncategorized)

    # ── Console ──
    print("\n" + "=" * 80)
    print(f"  Domain Detected : {tier_label} (Tier {tier})  [{detection_method}]")
    print(f"  Resume File     : {os.path.basename(resume_path)}")
    print(f"  Total Skills    : {len(skills)}")
    print("=" * 80)

    def _print_table(rows, start=1):
        header = f"  {'#':>3}  {'Skill':<30}  {'Confidence':>10}  {'Domains'}"
        print(header)
        print("  " + "-" * 70)
        for i, s in enumerate(rows, start):
            domains = s.get("domains", [])
            domain_str = ", ".join(domains) if domains else ""
            print(f"  {i:>3}. {s['skill']:<30}  {s['confidence']:>10.4f}  [{domain_str}]")

    if hard_skills:
        print(f"\n  Hard Skills ({len(hard_skills)})")
        print("  " + "─" * 52)
        _print_table(hard_skills, start=1)
    else:
        print("  No hard skills extracted.")

    if concepts:
        print(f"\n  Concepts ({len(concepts)})")
        print("  " + "─" * 52)
        _print_table(concepts, start=1)

    if soft_skills:
        print(f"\n  Soft Skills ({len(soft_skills)})")
        print("  " + "─" * 52)
        _print_table(soft_skills, start=1)

    print("\n" + "=" * 80 + "\n")

    # ── JSON ──
    all_names = sorted(set(s["skill"] for s in skills))
    output = {
        "skills": all_names,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  JSON saved to: {output_json_path}\n")
