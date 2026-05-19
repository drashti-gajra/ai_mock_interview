"""
domain_detector.py
==================
Auto-detect the dominant domain of a resume using keyword-frequency scoring.

Supported domains:
  - technology  (Tier 1)
  - marketing   (Tier 2)
  - sales       (Tier 3)
  - finance     (Tier 4)
  - law         (Tier 5)
  - operations  (Tier 6)
  - healthcare  (Tier 7)
"""

import re
from typing import Dict, Tuple

# ── Curated keyword seeds per domain ─────────────────────────────────────────
# Each list is hand-picked for high signal, low cross-domain overlap.

DOMAIN_KEYWORDS: Dict[str, list] = {
    "technology": [
        # Software / Web Development
        "api", "microservices", "kubernetes", "docker", "git", "github", "gitlab",
        "neural network", "machine learning", "deep learning", "ci/cd", "devops",
        "backend", "frontend", "full stack", "cloud", "aws", "azure", "gcp",
        "framework", "repository", "pipeline", "deployment", "containerization",
        "rest", "graphql", "kafka", "redis", "postgresql", "mongodb", "terraform",
        "ansible", "jenkins", "pytest", "junit", "selenium", "react", "angular",
        "vue", "node.js", "django", "flask", "spring boot", "microservice",
        "helm", "prometheus", "grafana", "elasticsearch",
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "golang", "rust",
        "php", "ruby", "scala", "kotlin", "swift",
        # IT Support / Networking
        "help desk", "service desk", "technical support", "tech support",
        "troubleshooting", "ticketing system", "spiceworks", "servicenow",
        "jira", "zendesk", "freshdesk", "remote desktop", "active directory",
        "group policy", "itil", "incident management",
        "network", "firewall", "router", "tcp/ip", "dns", "dhcp", "vpn",
        "cisco", "juniper", "linux", "windows server", "vmware",
        "cybersecurity", "penetration testing", "vulnerability", "siem",
        # Data / Analytics
        "data analysis", "data visualization", "etl", "data warehouse",
        # More tech signals
        "react native", "flutter", "swiftui", "langchain", "pytorch",
        "tensorflow", "scikit-learn", "jwt", "oauth", "websocket",
        "devsecops", "mlops", "sre", "platform engineering",
        "sonarqube", "github actions", "argocd", "sagemaker", "vertex ai",
        "databricks", "apache spark", "apache kafka", "apache flink",
        "figma", "cypress", "playwright", "jest", "postman", "swagger",
    ],
    "marketing": [
        # Marketing platforms
        "marketo", "hubspot", "pardot", "mailchimp", "klaviyo", "braze",
        "activecampaign", "constant contact", "convertkit", "eloqua",
        # SEO / SEM
        "semrush", "ahrefs", "moz", "screaming frog", "yoast",
        "google tag manager", "google search console",
        # Social media
        "facebook ads", "instagram ads", "linkedin ads", "tiktok ads",
        "hootsuite", "buffer", "sprout social", "brandwatch",
        # Content / Design
        "wordpress", "webflow", "squarespace", "canva", "adobe creative suite",
        "adobe photoshop", "adobe illustrator",
        # Ad platforms
        "google display network", "meta ads manager", "dv360", "the trade desk",
        "taboola", "outbrain", "amazon advertising",
        # Concepts
        "seo", "sem", "ppc", "content marketing", "email marketing",
        "social media marketing", "influencer marketing", "affiliate marketing",
        "programmatic advertising", "retargeting", "remarketing",
        "lead generation", "demand generation", "growth hacking",
        "conversion rate optimization", "a/b testing", "inbound marketing",
        "brand strategy", "brand positioning", "public relations",
        "content strategy", "copywriting", "editorial calendar",
        "marketing analytics", "attribution modeling", "marketing automation",
        "product marketing", "product launch", "competitive analysis",
        "customer acquisition", "customer retention", "omnichannel marketing",
    ],
    "sales": [
        # CRM
        "salesforce", "hubspot crm", "zoho crm", "pipedrive",
        "microsoft dynamics", "freshsales", "copper crm",
        # Sales tools
        "salesloft", "gong", "chorus", "linkedin sales navigator",
        "zoominfo", "apollo", "clearbit", "lusha", "clari",
        # Concepts
        "sales pipeline", "sales forecasting", "sales enablement",
        "account-based marketing", "go-to-market strategy",
        "territory management", "quota management", "sales operations",
        "revenue operations", "deal closing", "objection handling",
        "consultative selling", "solution selling", "spin selling",
        "cold calling", "prospecting", "social selling",
        "account management", "pipeline management", "sales analytics",
        "business development", "revenue growth",
        "customer success", "upselling", "cross-selling",
        "lead generation", "saas", "arr", "mrr", "quota",
        "account executive", "sdr", "sales funnel",
    ],
    "finance": [
        # ERP / Accounting
        "sap fico", "oracle financials", "netsuite", "quickbooks",
        "tally erp", "xero", "freshbooks", "sage", "workday",
        # Finance tools
        "bloomberg terminal", "capital iq", "pitchbook", "factset",
        "morningstar", "anaplan", "blackline", "concur",
        # Concepts
        "financial reporting", "financial analysis", "financial modeling",
        "budgeting", "forecasting", "variance analysis",
        "cost accounting", "management accounting", "general ledger",
        "accounts payable", "accounts receivable", "bank reconciliation",
        "month-end close", "year-end close", "revenue recognition",
        "cash flow analysis", "working capital", "p&l management",
        "tax planning", "tax compliance", "gaap compliance",
        "ifrs reporting", "sox compliance", "internal controls",
        "internal audit", "external audit", "risk assessment",
        "investment banking", "m&a", "due diligence",
        "private equity", "venture capital", "equity research",
        "portfolio management", "wealth management", "derivatives trading",
        "dcf modeling", "lbo modeling", "valuation",
        "financial planning", "budget management", "revenue forecasting",
        "ebitda", "cfa", "cpa", "acca",
    ],
    "law": [
        "litigation", "jurisdiction", "statute", "plaintiff", "defendant", "tort",
        "affidavit", "precedent", "habeas corpus", "arbitration", "injunction",
        "contract law", "criminal law", "civil law", "corporate law",
        "crpc", "constitution", "supreme court", "high court", "district court",
        "legal drafting", "legal research", "due diligence", "compliance",
        "intellectual property", "trademark", "patent", "copyright",
        "insolvency", "bankruptcy", "sebi regulations",
        "fema", "legal notice", "plaint", "written statement", "moot court",
        "bar council", "advocate", "solicitor", "barrister",
        "negligence", "liability", "damages",
        "deposition", "writ",
        "gdpr", "hipaa", "ccpa", "regulatory",
        "westlaw", "lexisnexis", "manupatra", "scc online",
        "mediation",
        "family law", "mergers", "acquisitions",
        "pro bono",
        "llb", "llm",
    ],
    "operations": [
        # Process improvement
        "supply chain", "logistics", "inventory", "warehouse", "procurement",
        "sourcing", "vendor management", "six sigma", "lean", "kaizen",
        "process improvement", "continuous improvement", "value stream",
        "standard operating procedures", "sops", "operational efficiency",
        "business process", "workflow optimization", "dmaic", "5s",
        # Project management
        "project management", "program management", "agile", "scrum", "kanban",
        "sprint", "backlog", "stakeholder management", "risk management",
        "pmp", "prince2", "milestone", "project delivery", "project planning",
        # Supply chain
        "demand planning", "demand forecasting", "inventory management",
        "warehouse management", "distribution center", "fulfillment",
        "transportation management", "freight", "last mile", "fleet management",
        "cold chain", "reverse logistics", "import export", "customs",
        "3pl", "order fulfillment",
        # Procurement
        "strategic sourcing", "category management", "spend analysis",
        "purchase order", "rfp", "rfq", "e-procurement", "contract management",
        "supplier relationship", "vendor onboarding", "cost reduction",
        # Quality
        "quality assurance", "quality control", "iso 9001", "root cause analysis",
        "capa", "fmea", "statistical process control",
        "ehs", "occupational health", "quality management system",
        # Manufacturing
        "production planning", "capacity planning", "plant operations",
        "assembly line", "shop floor", "bill of materials",
        "oee", "industrial engineering",
        # Tools
        "sap", "oracle erp", "microsoft dynamics", "netsuite", "erp",
        "wms", "tms", "manhattan associates", "blue yonder",
        "coupa", "jaggaer", "ariba", "servicenow",
    ],
    "healthcare": [
        # Clinical & Patient Care
        "patient care", "clinical assessment", "care coordination", "triage",
        "discharge planning", "medication administration", "wound care",
        "vital signs", "phlebotomy", "iv therapy", "specimen collection",
        "critical care", "emergency care", "intensive care", "ambulatory care",
        "palliative care", "hospice care", "telehealth", "telemedicine",
        "mental health", "behavioral health", "pediatric", "neonatal",
        "oncology", "surgical", "rehabilitation",
        "occupational therapy", "physical therapy", "speech therapy",
        # Medical Knowledge
        "pharmacology", "pathophysiology", "differential diagnosis",
        "evidence-based practice", "disease management",
        "radiology", "icd-10", "cpt codes",
        "medical billing", "medical coding", "revenue cycle",
        "prior authorization", "claims processing",
        # Healthcare IT
        "ehr", "emr", "epic", "cerner", "meditech", "allscripts",
        "athenahealth", "electronic health records",
        "health information management", "clinical informatics",
        "remote patient monitoring", "population health", "hl7", "fhir",
        "pacs", "lis", "clinical decision support",
        # Clinical Research
        "clinical research", "clinical trial", "good clinical practice",
        "gcp", "irb", "informed consent", "adverse event", "pharmacovigilance",
        "fda submission", "clinical data management", "biostatistics",
        "medidata", "ctms", "edc",
        # Healthcare Administration
        "healthcare administration", "hospital operations",
        "utilization management", "case management",
        "medical staff credentialing", "provider enrollment",
        "patient flow", "bed management",
        # Regulatory
        "hipaa", "joint commission", "cms regulations",
        "infection control", "infection prevention", "patient safety",
        "healthcare compliance", "osha healthcare",
        "medicare", "medicaid", "stark law", "anti-kickback",
        # Public Health
        "public health", "epidemiology", "disease surveillance",
        "community health", "contact tracing", "immunization",
        # Credentials
        "registered nurse", "nurse practitioner", "physician assistant",
        "nursing", "physician", "clinician",
    ],
}

DOMAIN_TIER_MAP: Dict[str, int] = {
    "technology": 1,
    "marketing":  2,
    "sales":      3,
    "finance":    4,
    "law":        5,
    "operations": 6,
    "healthcare": 7,
}

DEFAULT_DOMAIN = "technology"


def detect_domain(resume_text: str) -> Tuple[str, int, Dict[str, int]]:
    """
    Score resume text against each domain's keyword list and return the best match.
    Used for display/info only — extraction runs selected domain(s).

    Returns:
        Tuple of (domain_name, tier_number, scores_dict)
    """
    text_lower = resume_text.lower()

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if len(kw) < 3:
                continue  # Skip very short labels to avoid false matches
            pattern = r'\b' + re.escape(kw) + r'\b'
            count += len(re.findall(pattern, text_lower))
        scores[domain] = count

    best_domain = max(scores, key=lambda d: scores[d])

    if scores[best_domain] == 0:
        best_domain = DEFAULT_DOMAIN

    tier = DOMAIN_TIER_MAP.get(best_domain, 1)
    return best_domain, tier, scores
