"""
extractor_tier4_operations.py
==============================
GLiNER-based skill extractor for the Operations domain (Tier 4).

Labels cover: process improvement, supply chain, project management,
procurement, quality management, warehouse & logistics, facilities,
manufacturing, data analytics, ERP/operations tools, and soft skills.

Labels are EXCLUSIVE to this domain — no legal-procedure or pure
finance-instrument terms.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Dict, Any
from extractor_utils import normalize_skill_text, passes_noise_filter
from extractor_tier1_technology import _run_extraction_pipeline

DOMAIN    = "operations"
TIER      = 4
THRESHOLD = 0.25
GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


class OpsLabels:

    # ── Process Improvement & Continuous Improvement ──
    PROCESS = [
        "Process Improvement", "Continuous Improvement",
        "Business Process Management", "Lean Management", "Lean",
        "Six Sigma", "Lean Six Sigma", "Kaizen",
        "Operational Efficiency", "Workflow Optimization",
        "Standard Operating Procedures", "SOPs",
        "Root Cause Analysis", "Value Stream Mapping",
        "Business Process Reengineering", "5S Methodology",
        "Total Quality Management", "TQM", "DMAIC", "PDCA",
    ]

    # ── Supply Chain & Logistics ──
    SUPPLY_CHAIN = [
        "Supply Chain Management", "Logistics Management",
        "Inventory Management", "Demand Planning", "Demand Forecasting",
        "Warehouse Management", "Distribution Management",
        "Freight Management", "Last Mile Delivery",
        "Cold Chain Management", "Transportation Management",
        "Supplier Management", "Order Fulfillment",
        "Supply Chain Optimization", "Import Export",
        "Customs Clearance", "Inbound Logistics", "Outbound Logistics",
        "Reverse Logistics", "Fleet Management", "3PL",
    ]

    # ── Project Management ──
    PROJECT = [
        "Project Management", "Program Management", "Portfolio Management",
        "Agile", "Scrum", "Kanban", "Waterfall", "SAFe",
        "Project Planning", "Project Scheduling",
        "Risk Management", "Change Management",
        "Stakeholder Management", "Project Delivery",
        "Resource Allocation", "Budget Management",
        "Milestone Tracking", "PMP", "PRINCE2", "CAPM",
        "Sprint Planning", "Backlog Management",
    ]

    # ── Procurement & Sourcing ──
    PROCUREMENT = [
        "Procurement", "Strategic Sourcing", "Category Management",
        "Spend Analysis", "Vendor Negotiation", "Contract Management",
        "Supplier Relationship Management", "Purchase Order Management",
        "RFP", "RFQ", "Vendor Evaluation",
        "E-Procurement", "Indirect Procurement", "Direct Procurement",
        "Supplier Onboarding", "Cost Reduction",
    ]

    # ── Quality Management ──
    QUALITY = [
        "Quality Assurance", "Quality Control",
        "Quality Management System", "ISO 9001", "ISO Certification",
        "Audit Management", "Corrective Action", "CAPA",
        "Statistical Process Control", "SPC",
        "FMEA", "Defect Management",
        "Compliance Management", "Inspection",
    ]

    # ── Manufacturing & Production ──
    MANUFACTURING = [
        "Production Planning", "Manufacturing Operations",
        "Capacity Planning", "Assembly Line Management",
        "Maintenance Management", "Plant Operations",
        "Production Scheduling", "Materials Management",
        "Bill of Materials", "Shop Floor Management",
        "OEE", "Industrial Engineering",
    ]

    # ── Facilities & Office Operations ──
    FACILITIES = [
        "Facilities Management", "Vendor Coordination",
        "Office Administration", "Space Planning",
        "Property Management", "Health and Safety",
        "EHS", "Occupational Health",
    ]

    # ── Data, Reporting & Analytics ──
    ANALYTICS = [
        "Operational Analytics", "KPI Reporting",
        "Performance Metrics", "Business Intelligence",
        "Data Analysis", "Dashboard Development",
        "Operations Forecasting", "Data-Driven Decision Making",
        "SQL", "Power BI", "Tableau",
    ]

    # ── Operations Tools & Platforms ──
    TOOLS = [
        "SAP", "Oracle ERP", "Microsoft Dynamics",
        "NetSuite", "ERP", "WMS", "TMS",
        "Manhattan Associates", "Blue Yonder",
        "Coupa", "Jaggaer", "SAP Ariba",
        "ServiceNow", "Jira", "Monday.com",
        "Smartsheet", "Microsoft Project",
    ]

    # ── Soft Skills ──
    SOFT = [
        "Leadership", "Communication", "Interpersonal Skills",
        "Analytical Thinking", "Organizational Skills",
        "Negotiation", "Problem Solving",
        "Cross-Functional Collaboration", "Decision Making",
        "Time Management", "Adaptability",
        "Teamwork", "Critical Thinking",
        "Strategic Thinking", "Mentoring",
        "Resilience", "Initiative", "Attention to Detail",
        "Accountability", "Conflict Resolution",
        "Presentation Skills", "Emotional Intelligence",
    ]

    @classmethod
    def all(cls) -> list:
        return (
            cls.PROCESS + cls.SUPPLY_CHAIN + cls.PROJECT + cls.PROCUREMENT +
            cls.QUALITY + cls.MANUFACTURING + cls.FACILITIES + cls.ANALYTICS +
            cls.TOOLS + cls.SOFT
        )

    @classmethod
    def batches(cls) -> list:
        return [
            cls.PROCESS,
            cls.SUPPLY_CHAIN,
            cls.PROJECT,
            cls.PROCUREMENT,
            cls.QUALITY,
            cls.MANUFACTURING + cls.FACILITIES,
            cls.ANALYTICS,
            cls.TOOLS,
            cls.SOFT,
        ]


GLINER_LABELS: List[str] = OpsLabels.all()


class OperationsSkillExtractor:
    """
    Extracts operations skills from resume text using GLiNER.
    Uses the Operations (Tier 4) label set exclusively.
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
            domain_name="operations",
            label_batches=OpsLabels.batches(),
            original_doc=original_doc,
        )
