"""
extractor_finance.py
====================
Skill extractor for the Finance & Accounting domain.

Extracts: accounting tools, ERP systems, financial analysis, audit,
tax, banking, compliance, financial modeling, and related concepts.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Any
from extractor_tier1_technology import (
    _run_extraction_pipeline as _base_pipeline,
    _apply_final_scoring,
)
import extractor_tier1_technology as _t1
from extractor_marketing import _SharedLabels

DOMAIN     = "finance"
TIER       = 4
THRESHOLD  = 0.45
GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


# ── Finance Labels ────────────────────────────────────────────────────────────

class FinanceLabels:

    # ── Hard Skills: ERP & Accounting Software ──
    ERP_ACCOUNTING = [
        "SAP", "SAP FICO", "SAP S/4HANA", "Oracle Financials", "Oracle EBS",
        "Oracle Cloud", "Oracle Fusion", "NetSuite", "QuickBooks",
        "Tally ERP", "Xero", "FreshBooks", "Sage", "PeopleSoft",
        "Workday", "Workiva", "Great Plains", "JD Edwards",
    ]

    # ── Financial Tools & Platforms ──
    FINANCE_TOOLS = [
        "Bloomberg Terminal", "Capital IQ", "PitchBook", "FactSet",
        "Morningstar", "Reuters Eikon", "Hyperion", "Essbase",
        "Anaplan", "Adaptive Insights", "BlackLine", "Concur",
        "Coupa", "Certify", "Expensify",
    ]

    # ── Data & BI Tools ──
    DATA_BI = [
        "Microsoft Excel", "Excel", "Power BI", "Tableau",
        "Google Sheets", "MS Access", "VBA", "SQL",
        "Python", "R programming", "SPSS", "SAS",
        "Alteryx", "Cognos", "OBIEE", "Crystal Reports",
    ]

    # ── Audit & Compliance Tools ──
    AUDIT_TOOLS = [
        "ACL Analytics", "IDEA", "AuditBoard", "TeamMate",
        "Workiva", "Thomson Reuters", "CCH", "Wolters Kluwer",
    ]

    # ── Banking & Payment Platforms ──
    BANKING_PLATFORMS = [
        "SWIFT", "Bloomberg", "Murex", "Calypso",
        "Finastra", "Temenos", "FIS", "Fiserv",
        "Stripe", "PayPal", "Square",
    ]

    # ── Accounting & Finance Concepts ──
    CONCEPTS_ACCOUNTING = [
        "Financial Reporting", "Financial Analysis", "Financial Modeling",
        "Financial Planning", "Budgeting", "Forecasting",
        "Variance Analysis", "Cost Accounting", "Management Accounting",
        "General Ledger", "Accounts Payable", "Accounts Receivable",
        "Bank Reconciliation", "Month-end Close", "Year-end Close",
        "Fixed Asset Management", "Revenue Recognition",
        "Intercompany Accounting", "Financial Consolidation",
        "Cash Flow Analysis", "Cash Flow Forecasting",
        "Working Capital Management", "Revenue Analysis",
        "Management Reporting", "Cost Analysis",
        "Treasury Management", "P&L Management",
    ]

    # ── Tax & Compliance Concepts ──
    CONCEPTS_TAX = [
        "Tax Planning", "Tax Compliance", "Corporate Tax",
        "Transfer Pricing", "Sales Tax", "VAT", "GST",
        "GAAP Compliance", "IFRS Reporting", "SOX Compliance",
        "Internal Controls", "Regulatory Compliance",
        "Anti-Money Laundering", "KYC Compliance",
    ]

    # ── Audit Concepts ──
    CONCEPTS_AUDIT = [
        "Internal Audit", "External Audit", "Risk Assessment",
        "Audit Planning", "Financial Audit", "Operational Audit",
        "Compliance Audit", "Fraud Detection", "Due Diligence",
    ]

    # ── Banking & Investment Concepts ──
    CONCEPTS_BANKING = [
        "Investment Banking", "Mergers & Acquisitions", "Due Diligence",
        "Private Equity", "Venture Capital", "Equity Research",
        "Credit Analysis", "Portfolio Management", "Wealth Management",
        "Asset Allocation", "Risk Management", "Derivatives Trading",
        "Forex Management", "Trade Finance", "Capital Markets",
        "Fund Accounting", "DCF Modeling", "Sensitivity Analysis",
        "Valuation", "LBO Modeling",
    ]

    # ── FP&A and Strategy Concepts ──
    CONCEPTS_FPA = [
        "Financial Planning & Analysis", "Budget Management",
        "Revenue Forecasting", "Expense Management",
        "KPI Tracking", "Dashboard Reporting",
        "Business Intelligence", "Data-Driven Decision Making",
        "Scenario Analysis", "What-If Analysis",
        "Capital Budgeting", "ROI Analysis",
    ]

    # ── Process & Operations Concepts ──
    CONCEPTS_OPERATIONS = [
        "Process Improvement", "Lean Six Sigma", "ERP Implementation",
        "System Integration", "Data Migration", "Automation",
        "Accounts Payable Automation", "Invoice Processing",
        "Procurement Management", "Vendor Management",
        "Contract Management", "Spend Analysis",
    ]

    @classmethod
    def all_hard(cls) -> list:
        return (
            cls.ERP_ACCOUNTING + cls.FINANCE_TOOLS +
            cls.DATA_BI + cls.AUDIT_TOOLS + cls.BANKING_PLATFORMS +
            _SharedLabels.PROJECT_COLLAB + _SharedLabels.PRODUCTIVITY
        )

    @classmethod
    def all_concepts(cls) -> list:
        return (
            cls.CONCEPTS_ACCOUNTING + cls.CONCEPTS_TAX +
            cls.CONCEPTS_AUDIT + cls.CONCEPTS_BANKING +
            cls.CONCEPTS_FPA + cls.CONCEPTS_OPERATIONS
        )

    @classmethod
    def all(cls) -> list:
        return cls.all_hard() + cls.all_concepts() + _SharedLabels.SOFT

    @classmethod
    def batches(cls) -> list:
        return [
            cls.ERP_ACCOUNTING + cls.FINANCE_TOOLS,
            cls.DATA_BI + cls.AUDIT_TOOLS + cls.BANKING_PLATFORMS,
            _SharedLabels.PROJECT_COLLAB + _SharedLabels.PRODUCTIVITY,
            cls.CONCEPTS_ACCOUNTING + cls.CONCEPTS_TAX,
            cls.CONCEPTS_AUDIT + cls.CONCEPTS_BANKING,
            cls.CONCEPTS_FPA + cls.CONCEPTS_OPERATIONS,
            _SharedLabels.SOFT,
        ]


# ── Classification ────────────────────────────────────────────────────────────

_CATEGORY_MAP = {}
for _l in FinanceLabels.all_hard():
    _CATEGORY_MAP[_l.lower()] = "hard_skill"
for _l in FinanceLabels.all_concepts():
    _CATEGORY_MAP[_l.lower()] = "concept"
for _l in _SharedLabels.SOFT:
    _CATEGORY_MAP[_l.lower()] = "soft_skill"

_CONCEPT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "concept"}
_SOFT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "soft_skill"}

_orig_classify = _t1._classify_skill


def _classify_finance(skill_name: str, skill_labels: list) -> str:
    name_lower = skill_name.lower()
    if name_lower in _CATEGORY_MAP:
        return _CATEGORY_MAP[name_lower]
    for label in skill_labels:
        ll = label.lower()
        if ll in _SOFT_SET:
            return "soft_skill"
        if ll in _CONCEPT_SET:
            return "concept"
    return "hard_skill"


def _finance_scoring(skills, min_score=0.40):
    """Finance-specific scoring: trust structured skill sections."""
    result = []
    for s in skills:
        confidence = s.get("confidence", 0)
        frequency = s.get("frequency", 1)
        ctx = s.get("context", "").lower()
        category = s.get("category", "hard_skill")
        from_skill_section = "skilled in" in ctx or "proficient in" in ctx

        if from_skill_section:
            s["confidence"] = min(s["confidence"] * 1.15, 1.0)
            confidence = s["confidence"]

        # Multi-word boost
        word_count = len(s.get("skill", "").split())
        if word_count >= 2:
            s["confidence"] = min(s["confidence"] * 1.1, 1.0)
            confidence = s["confidence"]

        # Boost long concept phrases (3+ words like "Working Capital Management")
        if category == "concept" and word_count >= 3:
            s["confidence"] = min(s["confidence"] * 1.1, 1.0)
            confidence = s["confidence"]

        if category == "hard_skill":
            if from_skill_section:
                pass
            elif confidence < 0.60 and frequency == 1:
                continue
            elif confidence < 0.60:
                continue

        elif category == "concept":
            if from_skill_section:
                pass
            elif confidence < 0.45:
                continue

        elif category == "soft_skill":
            if from_skill_section:
                pass
            # Keep multi-word soft skills at lower threshold
            elif word_count >= 2 and confidence < 0.40:
                continue
            elif word_count == 1 and confidence < 0.55:
                continue

        from extractor_tier1_technology import _final_score
        score = _final_score(s)
        if score >= min_score:
            result.append(s)
    return result


# ── Extractor ─────────────────────────────────────────────────────────────────

class FinanceSkillExtractor:
    """Extracts accounting & finance skills from resume text."""

    def __init__(self, nlp=None, gliner_model=None):
        if nlp and gliner_model:
            self.nlp = nlp
            self.gliner_model = gliner_model
        else:
            import spacy
            from gliner import GLiNER
            print("Loading spaCy model...")
            self.nlp = spacy.load(SPACY_MODEL_NAME)
            print("spaCy model loaded.")
            print("Loading GLiNER model...")
            self.gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME, local_files_only=False)
            print("GLiNER model loaded.")

        self.threshold = THRESHOLD
        self.labels = FinanceLabels.all()

    def extract(self, resume_text: str, min_freq: int = 1, original_doc=None) -> List[Dict[str, Any]]:
        _t1._classify_skill = _classify_finance
        _orig_scoring = _t1._apply_final_scoring
        _t1._apply_final_scoring = _finance_scoring
        try:
            return _base_pipeline(
                resume_text=resume_text,
                nlp=self.nlp,
                gliner_model=self.gliner_model,
                labels=self.labels,
                threshold=self.threshold,
                min_freq=min_freq,
                domain_name="finance",
                label_batches=FinanceLabels.batches(),
                original_doc=original_doc,
            )
        finally:
            _t1._classify_skill = _orig_classify
            _t1._apply_final_scoring = _orig_scoring
