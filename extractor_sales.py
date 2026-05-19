"""
extractor_sales.py
==================
Skill extractor for the Sales domain.

Extracts: CRM, sales tools, pipeline management, account management,
forecasting, business development, revenue ops tools and concepts.
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

DOMAIN     = "sales"
TIER       = 2
THRESHOLD  = 0.45
GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


# ── Sales Labels ──────────────────────────────────────────────────────────────

class SalesLabels:

    # ── Hard Skills ──
    CRM = [
        "Salesforce", "HubSpot CRM", "Zoho CRM", "Pipedrive",
        "Microsoft Dynamics", "Freshsales", "Copper CRM",
        "Monday CRM", "Close CRM", "Insightly",
    ]

    SALES_TOOLS = [
        "SalesLoft", "Gong", "Chorus",
        "LinkedIn Sales Navigator", "ZoomInfo", "Apollo", "Clearbit",
        "Lusha", "Seamless.ai", "Revenue.io", "Clari",
    ]

    # ── Concepts ──
    CONCEPTS_SALES = [
        "Sales Pipeline", "Sales Forecasting", "Sales Enablement",
        "Account-Based Marketing", "Go-to-Market Strategy",
        "Territory Management", "Quota Management", "Sales Operations",
        "Revenue Operations", "Deal Closing", "Objection Handling",
        "Consultative Selling", "Solution Selling", "SPIN Selling",
        "Cold Calling", "Prospecting", "Social Selling",
        "Account Management", "Pipeline Management", "Sales Analytics",
        "Sales Strategy", "Client Relationship Management",
        "CRM Management", "Revenue Growth", "Business Development",
    ]

    CONCEPTS_CUSTOMER = [
        "Customer Acquisition", "Customer Retention", "Customer Lifecycle",
        "Customer Journey Mapping", "Net Promoter Score", "Churn Analysis",
        "Customer Success", "Customer Experience", "Voice of Customer",
        "Customer Onboarding", "Upselling", "Cross-Selling",
    ]

    CONCEPTS_REVENUE = [
        "Market Research", "Market Segmentation", "Competitive Analysis",
        "Value Proposition", "Sales Cycle", "Lead Generation",
        "Demand Generation", "Customer Lifetime Value",
    ]

    @classmethod
    def all_hard(cls) -> list:
        return (
            cls.CRM + cls.SALES_TOOLS +
            _SharedLabels.ANALYTICS + _SharedLabels.PROJECT_COLLAB +
            _SharedLabels.PRODUCTIVITY
        )

    @classmethod
    def all_concepts(cls) -> list:
        return (
            cls.CONCEPTS_SALES + cls.CONCEPTS_CUSTOMER + cls.CONCEPTS_REVENUE +
            _SharedLabels.CROSS_FUNCTIONAL_CONCEPTS
        )

    @classmethod
    def all(cls) -> list:
        return cls.all_hard() + cls.all_concepts() + _SharedLabels.SOFT

    @classmethod
    def batches(cls) -> list:
        return [
            cls.CRM + cls.SALES_TOOLS,
            _SharedLabels.ANALYTICS + _SharedLabels.PROJECT_COLLAB,
            _SharedLabels.PRODUCTIVITY,
            cls.CONCEPTS_SALES,
            cls.CONCEPTS_CUSTOMER + cls.CONCEPTS_REVENUE + _SharedLabels.CROSS_FUNCTIONAL_CONCEPTS,
            _SharedLabels.SOFT,
        ]


# ── Classification ────────────────────────────────────────────────────────────

_CATEGORY_MAP = {}
for _l in SalesLabels.all_hard():
    _CATEGORY_MAP[_l.lower()] = "hard_skill"
for _l in SalesLabels.all_concepts():
    _CATEGORY_MAP[_l.lower()] = "concept"
for _l in _SharedLabels.SOFT:
    _CATEGORY_MAP[_l.lower()] = "soft_skill"

_CONCEPT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "concept"}
_SOFT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "soft_skill"}

_orig_classify = _t1._classify_skill


def _classify_sales(skill_name: str, skill_labels: list) -> str:
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


def _sales_scoring(skills, min_score=0.40):
    """Sales-specific scoring: trust structured skill sections for hard skills."""
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

        if category == "hard_skill":
            if from_skill_section:
                pass  # Trust structured skills section
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
            elif confidence < 0.40:
                continue

        from extractor_tier1_technology import _final_score
        score = _final_score(s)
        if score >= min_score:
            result.append(s)
    return result


# ── Extractor ─────────────────────────────────────────────────────────────────

class SalesSkillExtractor:
    """Extracts sales-specific skills from resume text."""

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
        self.labels = SalesLabels.all()

    def extract(self, resume_text: str, min_freq: int = 1, original_doc=None) -> List[Dict[str, Any]]:
        _t1._classify_skill = _classify_sales
        _orig_scoring = _t1._apply_final_scoring
        _t1._apply_final_scoring = _sales_scoring
        try:
            return _base_pipeline(
                resume_text=resume_text,
                nlp=self.nlp,
                gliner_model=self.gliner_model,
                labels=self.labels,
                threshold=self.threshold,
                min_freq=min_freq,
                domain_name="sales",
                label_batches=SalesLabels.batches(),
                original_doc=original_doc,
            )
        finally:
            _t1._classify_skill = _orig_classify
            _t1._apply_final_scoring = _orig_scoring
