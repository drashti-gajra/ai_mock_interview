"""
run_extraction.py
=================
Entry point for the modular resume skill extraction system.

Usage:
    python run_extraction.py
    python run_extraction.py --domain marketing
    python run_extraction.py --output my_skills.json

Supported domains: technology, finance
"""

import argparse
import importlib
import sys
import os

from extractor_utils import load_resume_text, save_output
from domain_detector import detect_domain

# ── Hardcode Resume here (optional) ───────────────────────────────────────────
# Option 1 – file path:
#   RESUME_INPUT = "/path/to/resume.pdf"
# Option 2 – paste text directly between the triple quotes:
#   RESUME_INPUT = """
#   John Doe | john@example.com
#   Skills: Python, React, SQL ...
#   """
# Leave as None to use --resume flag instead.
RESUME_INPUT = """
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
FirstKey Homes, Georgia, USA | M a r 2 024 – Present| Manager, Analytics
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

# ── Domain Registry ───────────────────────────────────────────────────────────
DOMAIN_REGISTRY = {
    "technology":  ("extractor_tier1_technology", "TechnologySkillExtractor"),
    "finance":     ("extractor_finance",           "FinanceSkillExtractor"),
}

DOMAIN_ALIASES = {
    "tech":        "technology",
    "accounting":  "finance",
    "accounts":    "finance",
}


def resolve_domain(raw: str) -> str:
    lower = raw.lower().strip()
    return DOMAIN_ALIASES.get(lower, lower)


def load_extractor(domain: str, nlp=None, gliner_model=None):
    if domain not in DOMAIN_REGISTRY:
        available = ", ".join(DOMAIN_REGISTRY.keys())
        print(f"ERROR: Unknown domain '{domain}'. Available: {available}")
        sys.exit(1)
    module_name, class_name = DOMAIN_REGISTRY[domain]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(nlp=nlp, gliner_model=gliner_model)


def merge_skills(all_skills):
    """Merge skills from multiple extractors, deduplicate by name, track domains."""
    seen = {}
    for s in all_skills:
        key = s["skill"].lower()
        if key not in seen:
            seen[key] = {**s, "domains": [s.get("domain", "unknown")]}
        else:
            existing = seen[key]
            existing["domains"].append(s.get("domain", "unknown"))
            if s["confidence"] > existing["confidence"]:
                existing["confidence"] = s["confidence"]
                existing["skill"] = s["skill"]
            existing["frequency"] = existing.get("frequency", 1) + s.get("frequency", 1)
    # Deduplicate domain lists
    for v in seen.values():
        v["domains"] = list(dict.fromkeys(v["domains"]))

    # Length-based dominance dedup across domains:
    # "SAP FICO" > "SAP", "Microsoft Excel" > "Excel"
    merged = list(seen.values())
    merged = sorted(merged, key=lambda x: len(x["skill"]), reverse=True)
    selected = []
    for s in merged:
        name = s["skill"].lower()
        if not any(name in existing["skill"].lower() for existing in selected):
            selected.append(s)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract skills from a resume using domain-specific GLiNER labels."
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to resume file (.pdf or .txt) — optional if RESUME_INPUT is set above"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Domain: technology, marketing, sales, finance, law, operations, healthcare (optional — auto-detected if omitted)"
    )
    parser.add_argument(
        "--output", default="skills_output.json",
        help="Path for the output JSON file (default: skills_output.json)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=1,
        help="Minimum skill occurrence frequency to include (default: 1)"
    )
    args = parser.parse_args()

    # ── 1. Load resume ──
    if RESUME_INPUT:
        # Pasted text or file path hardcoded above
        if os.path.exists(RESUME_INPUT.strip()):
            resume_path = RESUME_INPUT.strip()
            print(f"\nLoading resume: {resume_path}")
            resume_text = load_resume_text(resume_path)
        else:
            resume_path = "hardcoded_input"
            resume_text = RESUME_INPUT.strip()
            print(f"\nUsing hardcoded resume text ({len(resume_text)} characters)")
    elif args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            print(f"ERROR: Resume file not found: {resume_path}")
            sys.exit(1)
        print(f"\nLoading resume: {resume_path}")
        resume_text = load_resume_text(resume_path)
    else:
        print("ERROR: Provide a resume via --resume flag or set RESUME_INPUT at the top of this file.")
        sys.exit(1)

    print(f"Resume loaded ({len(resume_text)} characters)\n")

    # ── 2. Domain detection (for info only) ──
    domain, tier, scores = detect_domain(resume_text)
    print(f"Domain signals: { {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1]) if k in ('technology', 'finance')} }")

    # ── 3. Load models once, run ALL extractors ──
    import time as _time
    _t_import_start = _time.time()
    import spacy
    _t_spacy_imported = _time.time()
    from gliner import GLiNER
    _t_gliner_imported = _time.time()
    print(f"\n[TIMING] import spacy: {_t_spacy_imported - _t_import_start:.2f}s")
    print(f"[TIMING] import gliner (torch+transformers): {_t_gliner_imported - _t_spacy_imported:.2f}s")

    print("\nLoading models...")
    _t_load_start = _time.time()
    nlp = spacy.load("en_core_web_sm")
    _t_spacy_loaded = _time.time()
    gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", local_files_only=False)
    _t_gliner_loaded = _time.time()
    print(f"[TIMING] spacy.load(): {_t_spacy_loaded - _t_load_start:.2f}s")
    print(f"[TIMING] GLiNER.from_pretrained(): {_t_gliner_loaded - _t_spacy_loaded:.2f}s")
    print("Models loaded.\n")

    # Pre-compute spaCy doc once for reuse across all domain extractors
    print("Pre-processing resume text...")
    original_doc = nlp(resume_text)
    print("Pre-processing done.\n")

    if args.domain:
        # Manual override: run single domain
        domain = resolve_domain(args.domain)
        domains_to_run = [domain]
        print(f"Running single domain (manual): {domain}")
    else:
        # Smart selection: run top domain + any domain with meaningful signal
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top_domain, top_score = sorted_scores[0]
        domains_to_run = [top_domain]

        # Add secondary domains if they have real signal (>= 50% of top AND >= 10 keyword hits)
        # This catches hybrid resumes (tech+finance, healthcare+ops, law+sales, etc.)
        if top_score > 0:
            for other_domain, other_score in sorted_scores[1:]:
                if other_score >= 10 and other_score / top_score >= 0.50:
                    domains_to_run.append(other_domain)

        print(f"Running extractors: {domains_to_run}")

    all_skills = []
    for d in domains_to_run:
        print(f"\n  Extracting {d} skills...")
        extractor = load_extractor(d, nlp=nlp, gliner_model=gliner_model)
        skills = extractor.extract(resume_text, min_freq=args.min_freq, original_doc=original_doc)
        for s in skills:
            s["domain"] = d
        all_skills.extend(skills)

    # ── 4. Merge & deduplicate across domains ──
    if len(domains_to_run) > 1:
        merged = merge_skills(all_skills)
        print(f"\n  Merged: {len(all_skills)} raw → {len(merged)} unique skills")
    else:
        merged = all_skills

    # ── 5. Output ──
    save_output(
        skills=merged,
        domain=domain,
        tier=tier,
        resume_path=resume_path,
        output_json_path=args.output,
        detection_method="all-domains",
    )


if __name__ == "__main__":
    main()
