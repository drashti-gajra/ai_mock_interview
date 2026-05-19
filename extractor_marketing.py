"""
extractor_marketing.py
======================
Skill extractor for the Marketing domain.

Extracts: digital marketing, content marketing, SEO/SEM, social media,
email marketing, paid ads, analytics, brand, PR, growth, product marketing
tools and concepts.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Any
from extractor_tier1_technology import (
    _run_extraction_pipeline as _base_pipeline,
    _apply_final_scoring,
)
import extractor_tier1_technology as _t1

DOMAIN     = "marketing"
TIER       = 2
THRESHOLD  = 0.45
GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


# ── Shared labels (also used by sales) ────────────────────────────────────────

class _SharedLabels:
    """Labels shared across marketing and sales domains."""

    PROJECT_COLLAB = [
        "Asana", "Monday.com", "Trello", "Notion", "Slack",
        "Confluence", "Airtable", "ClickUp", "Basecamp",
    ]

    PRODUCTIVITY = [
        "Microsoft Excel", "Google Sheets", "Google Docs",
        "Microsoft PowerPoint", "Google Slides", "MS Office",
        "Office 365", "Google Workspace",
    ]

    ANALYTICS = [
        "Google Analytics", "Adobe Analytics", "Mixpanel", "Amplitude",
        "Hotjar", "Looker", "Tableau", "Power BI", "Google Data Studio",
        "Heap Analytics", "Pendo", "FullStory", "Kissmetrics",
    ]

    # Cross-functional tools used by BOTH marketing and sales teams
    CROSS_FUNCTIONAL_TOOLS = [
        "Salesforce", "HubSpot CRM", "Zoho CRM", "Pipedrive",
        "LinkedIn Sales Navigator", "Microsoft Dynamics",
    ]

    # Cross-functional concepts that appear on both marketing and sales resumes
    CROSS_FUNCTIONAL_CONCEPTS = [
        "Lead Generation", "Demand Generation", "Pipeline Management",
        "Sales Forecasting", "Customer Relationship Management",
        "Account Management", "Customer Acquisition", "Customer Retention",
        "Market Research", "Customer Segmentation", "Revenue Growth",
        "Campaign Management", "Digital Marketing",
        "Search Engine Optimization", "Search Engine Marketing",
    ]

    SOFT = [
        "Leadership", "Communication", "Collaboration",
        "Teamwork", "Team Leadership", "Cross-Functional Collaboration",
        "Problem Solving", "Critical Thinking", "Analytical Thinking",
        "Strategic Thinking", "Creative Thinking", "Design Thinking",
        "Logical Thinking", "Systems Thinking",
        "Project Management", "Time Management", "Priority Management",
        "Stakeholder Management", "Change Management", "Conflict Resolution",
        "Decision Making", "Risk Assessment",
        "Creativity", "Adaptability", "Resilience", "Self-Motivation",
        "Accountability", "Integrity", "Emotional Intelligence",
        "Growth Mindset", "Continuous Learning",
        "Public Speaking", "Presentation Skills", "Written Communication",
        "Active Listening", "Storytelling", "Persuasion", "Negotiation",
        "Mentoring", "Coaching", "Training", "Team Building",
        "Talent Development", "Performance Management",
        "Attention to Detail", "Multitasking", "Organization",
        "Planning", "Resourcefulness", "Initiative",
        "Relationship Building", "Client Management", "Customer Focus",
        "Networking", "Influencing", "Empathy",
    ]


# ── Marketing Labels ──────────────────────────────────────────────────────────

class MarketingLabels:

    # ── Hard Skills ──
    MARKETING_PLATFORMS = [
        "HubSpot", "Marketo", "Mailchimp", "Pardot", "ActiveCampaign",
        "Klaviyo", "Braze", "Iterable", "Constant Contact", "ConvertKit",
        "Drip", "Sendinblue", "Campaign Monitor", "Eloqua",
    ]

    SEO_SEM = [
        "Google Ads", "Google Search Console", "SEMrush", "Ahrefs",
        "Moz", "Screaming Frog", "Yoast", "Google Tag Manager",
        "SpyFu", "Ubersuggest", "Majestic", "BrightEdge",
    ]

    SOCIAL_MEDIA = [
        "Facebook Ads", "Instagram Ads", "LinkedIn Ads", "TikTok Ads",
        "Twitter Ads", "Hootsuite", "Buffer", "Sprout Social", "Later",
        "SocialBee", "Brandwatch", "Mention", "BuzzSumo",
    ]

    CONTENT_DESIGN = [
        "WordPress", "Webflow", "Squarespace", "Wix",
        "Canva", "Figma", "Adobe Creative Suite", "Adobe Photoshop",
        "Adobe Illustrator", "Adobe InDesign", "Adobe Premiere Pro",
        "Final Cut Pro", "DaVinci Resolve",
    ]

    EMAIL_AUTOMATION = [
        "SendGrid", "Intercom", "Drift", "Zapier", "Make",
        "Segment", "Customer.io", "Autopilot", "Leanplum",
    ]

    VIDEO_MEDIA = [
        "YouTube Studio", "Wistia", "Vimeo", "Loom",
        "Vidyard", "Riverside", "StreamYard",
    ]

    AD_PLATFORMS = [
        "Google Display Network", "Meta Ads Manager", "DV360",
        "The Trade Desk", "Taboola", "Outbrain", "Amazon Advertising",
        "Criteo", "AdRoll", "StackAdapt",
    ]

    ECOMMERCE = [
        "Shopify", "WooCommerce", "Magento", "BigCommerce",
        "Stripe", "PayPal", "Square",
    ]

    # ── Concepts ──
    CONCEPTS_DIGITAL = [
        "SEO", "SEM", "PPC", "Content Marketing", "Email Marketing",
        "Social Media Marketing", "Influencer Marketing", "Affiliate Marketing",
        "Display Advertising", "Programmatic Advertising", "Native Advertising",
        "Retargeting", "Remarketing", "Pay Per Click",
    ]

    CONCEPTS_GROWTH = [
        "Lead Generation", "Demand Generation", "Pipeline Generation",
        "Growth Hacking", "Conversion Rate Optimization", "A/B Testing",
        "Landing Page Optimization", "Lead Nurturing", "Lead Scoring",
        "Marketing Qualified Lead", "Sales Qualified Lead",
        "Inbound Marketing", "Outbound Marketing",
    ]

    CONCEPTS_BRAND = [
        "Brand Strategy", "Brand Positioning", "Brand Awareness",
        "Public Relations", "Media Relations", "Press Release",
        "Crisis Communications", "Corporate Communications",
        "Thought Leadership", "Employer Branding", "Brand Guidelines",
    ]

    CONCEPTS_CONTENT = [
        "Content Strategy", "Copywriting", "Editorial Calendar",
        "Blog Management", "Podcast Production", "Webinar Production",
        "Video Marketing", "Content Distribution", "Content Audit",
        "SEO Copywriting", "Technical Writing", "Storytelling",
    ]

    CONCEPTS_ANALYTICS = [
        "Marketing Analytics", "Attribution Modeling", "Customer Segmentation",
        "Cohort Analysis", "Funnel Analysis", "ROI Analysis", "KPI Tracking",
        "Marketing Mix Modeling", "Multi-Touch Attribution",
        "Campaign Performance", "Conversion Tracking", "Data-Driven Marketing",
    ]

    CONCEPTS_PRODUCT = [
        "Product Marketing", "Product Launch", "Competitive Analysis",
        "Market Research", "Customer Persona", "Value Proposition",
        "Positioning", "Messaging Framework", "Win/Loss Analysis",
        "Sales Collateral", "Battlecard",
    ]

    CONCEPTS_CUSTOMER = [
        "Customer Acquisition", "Customer Retention", "Customer Lifecycle",
        "Customer Journey Mapping", "Net Promoter Score", "Churn Analysis",
        "Customer Experience", "Voice of Customer",
    ]

    CONCEPTS_STRATEGY = [
        "Marketing Strategy", "Digital Strategy", "Omnichannel Marketing",
        "Integrated Marketing", "Marketing Automation", "Performance Marketing",
        "Event Marketing", "Field Marketing", "Channel Marketing",
        "Partner Marketing", "Co-Marketing", "Marketing Budget",
    ]

    @classmethod
    def all_hard(cls) -> list:
        return (
            cls.MARKETING_PLATFORMS + cls.SEO_SEM + cls.SOCIAL_MEDIA +
            cls.CONTENT_DESIGN + cls.EMAIL_AUTOMATION + cls.VIDEO_MEDIA +
            cls.AD_PLATFORMS + cls.ECOMMERCE +
            _SharedLabels.ANALYTICS + _SharedLabels.PROJECT_COLLAB +
            _SharedLabels.PRODUCTIVITY + _SharedLabels.CROSS_FUNCTIONAL_TOOLS
        )

    @classmethod
    def all_concepts(cls) -> list:
        return (
            cls.CONCEPTS_DIGITAL + cls.CONCEPTS_GROWTH +
            cls.CONCEPTS_BRAND + cls.CONCEPTS_CONTENT +
            cls.CONCEPTS_ANALYTICS + cls.CONCEPTS_PRODUCT +
            cls.CONCEPTS_CUSTOMER + cls.CONCEPTS_STRATEGY +
            _SharedLabels.CROSS_FUNCTIONAL_CONCEPTS
        )

    @classmethod
    def all(cls) -> list:
        return cls.all_hard() + cls.all_concepts() + _SharedLabels.SOFT

    @classmethod
    def batches(cls) -> list:
        return [
            cls.MARKETING_PLATFORMS + _SharedLabels.CROSS_FUNCTIONAL_TOOLS,
            _SharedLabels.ANALYTICS + cls.SEO_SEM,
            cls.SOCIAL_MEDIA + cls.CONTENT_DESIGN,
            cls.EMAIL_AUTOMATION + cls.VIDEO_MEDIA + cls.AD_PLATFORMS + cls.ECOMMERCE,
            _SharedLabels.PROJECT_COLLAB + _SharedLabels.PRODUCTIVITY,
            cls.CONCEPTS_DIGITAL + cls.CONCEPTS_GROWTH,
            cls.CONCEPTS_BRAND + cls.CONCEPTS_CONTENT,
            cls.CONCEPTS_ANALYTICS + cls.CONCEPTS_PRODUCT,
            cls.CONCEPTS_CUSTOMER + cls.CONCEPTS_STRATEGY + _SharedLabels.CROSS_FUNCTIONAL_CONCEPTS,
            _SharedLabels.SOFT,
        ]


# ── Classification ────────────────────────────────────────────────────────────

_CATEGORY_MAP = {}
for _l in MarketingLabels.all_hard():
    _CATEGORY_MAP[_l.lower()] = "hard_skill"
for _l in MarketingLabels.all_concepts():
    _CATEGORY_MAP[_l.lower()] = "concept"
for _l in _SharedLabels.SOFT:
    _CATEGORY_MAP[_l.lower()] = "soft_skill"

_CONCEPT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "concept"}
_SOFT_SET = {k for k, v in _CATEGORY_MAP.items() if v == "soft_skill"}

_orig_classify = _t1._classify_skill


def _classify_marketing(skill_name: str, skill_labels: list) -> str:
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


def _marketing_scoring(skills, min_score=0.40):
    """Marketing-specific scoring: trust structured skill sections for hard skills."""
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
                pass  # Trust structured skills section
            elif confidence < 0.45:
                continue

        elif category == "soft_skill":
            if from_skill_section:
                pass  # Trust structured skills section
            elif confidence < 0.40:
                continue

        from extractor_tier1_technology import _final_score
        score = _final_score(s)
        if score >= min_score:
            result.append(s)
    return result


# ── Extractor ─────────────────────────────────────────────────────────────────

class MarketingSkillExtractor:
    """Extracts marketing-specific skills from resume text."""

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
        self.labels = MarketingLabels.all()

    def extract(self, resume_text: str, min_freq: int = 1, original_doc=None) -> List[Dict[str, Any]]:
        _t1._classify_skill = _classify_marketing
        _orig_scoring = _t1._apply_final_scoring
        _t1._apply_final_scoring = _marketing_scoring
        try:
            return _base_pipeline(
                resume_text=resume_text,
                nlp=self.nlp,
                gliner_model=self.gliner_model,
                labels=self.labels,
                threshold=self.threshold,
                min_freq=min_freq,
                domain_name="marketing",
                label_batches=MarketingLabels.batches(),
                original_doc=original_doc,
            )
        finally:
            _t1._classify_skill = _orig_classify
            _t1._apply_final_scoring = _orig_scoring
