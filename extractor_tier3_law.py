"""
extractor_tier3_law.py
=======================
GLiNER-based skill extractor for the Law domain (Tier 3).

Labels cover: legal research, litigation, compliance, corporate law,
IP law, contract law, legal tools, regulatory frameworks, and soft skills.

Labels are EXCLUSIVE to this domain — no tech stack or finance instrument terms.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Dict, Any
from extractor_utils import normalize_skill_text, passes_noise_filter
from extractor_tier1_technology import _run_extraction_pipeline

DOMAIN    = "law"
TIER      = 3
THRESHOLD = 0.2
GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


class LawLabels:

    # ── Core Legal Skills ──
    CORE = [
        "Legal Research", "Legal Analysis", "Legal Writing",
        "Legal Drafting", "Statutory Interpretation",
        "Case Law Analysis", "Legal Citation",
        "Due Diligence", "Legal Advisory",
        "Regulatory Analysis", "Legal Opinion Writing",
    ]

    # ── Litigation & Dispute Resolution ──
    LITIGATION = [
        "Litigation", "Trial Preparation", "Case Management",
        "Motion Practice", "Discovery", "E-Discovery",
        "Oral Advocacy", "Cross-Examination", "Deposition",
        "Arbitration", "Mediation", "Alternative Dispute Resolution",
        "Settlement Negotiation", "Class Action",
        "Appellate Practice", "Moot Court",
    ]

    # ── Practice Areas / Specializations ──
    SPECIALIZATIONS = [
        "Corporate Law", "Criminal Law", "Civil Law",
        "Contract Law", "Employment Law", "Labor Law",
        "Intellectual Property Law", "IP Law", "Patent Law",
        "Trademark Law", "Copyright Law",
        "Family Law", "Real Estate Law", "Property Law",
        "Constitutional Law", "Environmental Law",
        "International Law", "Tax Law", "Banking Law",
        "Insurance Law", "Immigration Law",
        "Mergers and Acquisitions", "M&A",
        "Securities Law", "Antitrust Law",
        "Insolvency", "Bankruptcy Law",
        "Healthcare Law", "Cyber Law", "Data Privacy Law",
    ]

    # ── Compliance & Regulatory ──
    COMPLIANCE = [
        "Regulatory Compliance", "Legal Compliance",
        "GDPR", "CCPA", "HIPAA", "SOX Compliance",
        "AML Compliance", "KYC Compliance", "FCPA",
        "Corporate Governance", "Board Advisory",
        "Risk Assessment", "Compliance Audit",
        "Data Privacy", "SEBI Regulations", "FEMA",
        "FDA Compliance", "EPA Compliance", "OSHA",
        "EEOC", "Anti-Money Laundering",
    ]

    # ── Contract & Transaction ──
    CONTRACTS = [
        "Contract Drafting", "Contract Review", "Contract Negotiation",
        "Contract Management", "Commercial Contracts",
        "Non-Disclosure Agreement", "NDA",
        "Service Level Agreement", "SLA",
        "Licensing Agreement", "Franchise Agreement",
        "Joint Venture Agreement", "Partnership Agreement",
        "Lease Agreement", "Employment Agreement",
    ]

    # ── Legal Tools & Software ──
    TOOLS = [
        "LexisNexis", "Westlaw", "Manupatra", "SCC Online",
        "Clio", "MyCase", "PracticePanther",
        "Relativity", "Concordance", "Nuix",
        "Ironclad", "Agiloft", "ContractPodAi",
        "NetDocuments", "iManage", "DocuSign",
        "Aderant", "Legal Tracker", "Thomson Reuters",
        "Bloomberg Law", "HeinOnline",
    ]

    # ── Administrative & General ──
    ADMIN = [
        "Microsoft Office", "Google Workspace",
        "Document Management", "Legal Billing",
        "Case Tracking", "Court Filing",
        "Pro Bono", "Legal Aid",
        "Bar Council", "Bar Admission",
    ]

    # ── Soft Skills ──
    SOFT = [
        "Negotiation", "Advocacy", "Communication",
        "Critical Thinking", "Analytical Thinking",
        "Attention to Detail", "Research Skills",
        "Public Speaking", "Persuasion",
        "Leadership", "Teamwork", "Problem Solving",
        "Time Management", "Client Relationship Management",
        "Ethical Judgment", "Conflict Resolution",
    ]

    @classmethod
    def all(cls) -> list:
        return (
            cls.CORE + cls.LITIGATION + cls.SPECIALIZATIONS +
            cls.COMPLIANCE + cls.CONTRACTS + cls.TOOLS +
            cls.ADMIN + cls.SOFT
        )

    @classmethod
    def batches(cls) -> list:
        return [
            cls.CORE,
            cls.LITIGATION,
            cls.SPECIALIZATIONS,
            cls.COMPLIANCE,
            cls.CONTRACTS,
            cls.TOOLS,
            cls.ADMIN + cls.SOFT,
        ]


GLINER_LABELS: List[str] = LawLabels.all()


class LawSkillExtractor:
    """
    Extracts law / legal skills from resume text using GLiNER.
    Uses the Law (Tier 3) label set exclusively.
    """

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
        self.labels    = GLINER_LABELS

    def extract(self, resume_text: str, min_freq: int = 1, original_doc=None) -> List[Dict[str, Any]]:
        return _run_extraction_pipeline(
            resume_text=resume_text,
            nlp=self.nlp,
            gliner_model=self.gliner_model,
            labels=self.labels,
            threshold=self.threshold,
            min_freq=min_freq,
            domain_name="law",
            label_batches=LawLabels.batches(),
            original_doc=original_doc,
        )
