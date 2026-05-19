"""
extractor_tier5_healthcare.py
==============================
GLiNER-based skill extractor for the Healthcare domain (Tier 5).

Labels cover: clinical skills, medical knowledge, healthcare IT (EHR/EMR),
clinical research (GCP/IRB/FDA), healthcare administration, regulatory
compliance (HIPAA/Joint Commission/CMS), public health, nursing, pharmacy,
and soft skills specific to healthcare professionals.

Labels are EXCLUSIVE to this domain — no legal-procedure or operations
terms that would bleed into other domain extractors.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Dict, Any
from extractor_utils import normalize_skill_text, passes_noise_filter
from extractor_tier1_technology import _run_extraction_pipeline

DOMAIN    = "healthcare"
TIER      = 5
THRESHOLD = 0.25
GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


class HCLabels:

    # ── Clinical & Patient Care ──
    CLINICAL = [
        "Patient Care", "Patient Assessment", "Clinical Assessment",
        "Clinical Documentation", "Patient Monitoring", "Vital Signs",
        "Physical Examination", "Nursing Care", "Wound Care",
        "Medication Administration", "Patient Education", "Discharge Planning",
        "Care Coordination", "Triage", "Emergency Care", "Critical Care",
        "Ambulatory Care", "Palliative Care", "Geriatric Care",
        "Pediatric Care", "Maternal Care", "Mental Health Care",
        "Home Health Care", "IV Therapy", "Phlebotomy",
        "Specimen Collection", "Infection Control",
    ]

    # ── Medical Knowledge & Specializations ──
    MEDICAL = [
        "Pharmacology", "Anatomy", "Physiology", "Pathophysiology",
        "Disease Management", "Chronic Disease Management", "Preventive Care",
        "Differential Diagnosis", "Radiology", "Diagnostics",
        "Clinical Decision Making", "Evidence-Based Practice",
        "Treatment Planning", "Clinical Protocols",
        "ICD-10 Coding", "CPT Coding", "Medical Coding", "Medical Billing",
        "Revenue Cycle Management",
    ]

    # ── Healthcare IT & Digital Health ──
    HEALTH_IT = [
        "Electronic Health Records", "EHR", "EMR",
        "Epic", "Cerner", "Meditech", "Allscripts", "Athenahealth",
        "Health Information Management", "Clinical Informatics",
        "Telehealth", "Telemedicine", "Remote Patient Monitoring",
        "Health Data Analytics", "Population Health Management",
        "HL7", "FHIR", "PACS", "LIS",
        "Clinical Decision Support", "CPOE",
        "Healthcare Interoperability",
    ]

    # ── Clinical Research & Trials ──
    RESEARCH = [
        "Clinical Research", "Clinical Trials", "Clinical Trial Management",
        "Good Clinical Practice", "GCP", "IRB Protocol", "Informed Consent",
        "Clinical Data Management", "Adverse Event Reporting",
        "Pharmacovigilance", "Drug Safety", "FDA Submission",
        "IND Application", "NDA Submission",
        "Protocol Development", "Research Methodology", "Biostatistics",
        "Post-Market Surveillance", "Medidata", "Oracle Clinical",
        "CTMS", "EDC",
    ]

    # ── Healthcare Administration ──
    ADMINISTRATION = [
        "Healthcare Administration", "Hospital Operations",
        "Healthcare Operations", "Healthcare Project Management",
        "Healthcare Quality Improvement", "Healthcare Finance",
        "Medical Staff Credentialing", "Healthcare Contract Management",
        "Prior Authorization", "Claims Processing",
        "Medical Billing and Coding", "Healthcare Scheduling",
        "Patient Flow Management", "Case Management",
        "Utilization Management", "Health Insurance Operations",
        "Bed Management", "Provider Enrollment",
    ]

    # ── Regulatory & Compliance ──
    COMPLIANCE = [
        "HIPAA Compliance", "HIPAA", "Joint Commission",
        "CMS Compliance", "CMS Regulations", "FDA Regulations",
        "OSHA Healthcare", "Infection Prevention",
        "Accreditation", "Risk Management",
        "Patient Safety", "Quality Assurance",
        "Privacy and Security Compliance", "Healthcare Audit",
        "Medicare", "Medicaid", "Stark Law", "Anti-Kickback",
    ]

    # ── Public Health & Epidemiology ──
    PUBLIC_HEALTH = [
        "Public Health", "Epidemiology", "Health Promotion",
        "Community Health", "Disease Surveillance", "Health Education",
        "Global Health", "Health Policy", "Environmental Health",
        "Occupational Health", "Infection Prevention",
        "Contact Tracing", "Immunization",
        "Social Determinants of Health",
    ]

    # ── Pharmacy & Life Sciences ──
    PHARMACY = [
        "Pharmacy", "Pharmaceutical", "Drug Dispensing",
        "Medication Therapy Management", "Pharmacokinetics",
        "Drug Interactions", "Formulary Management",
        "Compounding", "Sterile Compounding",
        "Clinical Pharmacy", "Oncology Pharmacy", "Specialty Pharmacy",
    ]

    # ── Healthcare Tools & Certifications ──
    TOOLS = [
        "BLS", "ACLS", "PALS",
        "Medical Imaging", "Laboratory Information System",
        "Pharmacy Information System",
        "Care Management Platform", "Credentialing Software",
        "Incident Reporting", "Nurse Scheduling",
        "CCRP", "CRA", "CRC",
    ]

    # ── Soft Skills ──
    SOFT = [
        "Leadership", "Communication", "Empathy",
        "Active Listening", "Cultural Competence",
        "Teamwork", "Problem Solving", "Critical Thinking",
        "Time Management", "Stress Management",
        "Patient Advocacy", "Compassion", "Resilience",
        "Adaptability", "Attention to Detail", "Collaboration",
        "Conflict Resolution", "Mentoring", "Accountability",
        "Emotional Intelligence", "Decision Making",
    ]

    @classmethod
    def all(cls) -> list:
        return (
            cls.CLINICAL + cls.MEDICAL + cls.HEALTH_IT + cls.RESEARCH +
            cls.ADMINISTRATION + cls.COMPLIANCE + cls.PUBLIC_HEALTH +
            cls.PHARMACY + cls.TOOLS + cls.SOFT
        )

    @classmethod
    def batches(cls) -> list:
        return [
            cls.CLINICAL,
            cls.MEDICAL,
            cls.HEALTH_IT,
            cls.RESEARCH,
            cls.ADMINISTRATION,
            cls.COMPLIANCE,
            cls.PUBLIC_HEALTH + cls.PHARMACY,
            cls.TOOLS,
            cls.SOFT,
        ]


GLINER_LABELS: List[str] = HCLabels.all()


class HealthcareSkillExtractor:
    """
    Extracts healthcare skills from resume text using GLiNER.
    Uses the Healthcare (Tier 5) label set exclusively.
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
            domain_name="healthcare",
            label_batches=HCLabels.batches(),
            original_doc=original_doc,
        )
