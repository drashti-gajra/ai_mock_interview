"""
extractor_tier1_technology.py
==============================
Skill extractor for the Technology domain (Tier 1).

Labels cover: programming languages, frameworks, databases, cloud platforms,
CI/CD, DevOps, ML/AI tools, networking, embedded systems, and more.
These labels are EXCLUSIVE to this domain and do not appear in Tier 2 or Tier 3.
"""

import sys
import re as _re
sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Any
from extractor_utils import normalize_skill_text, passes_noise_filter

# ── Coursework sentence detector ─────────────────────────────────────────────
_COURSEWORK_HEADER = _re.compile(
    r'^\s*(relevant\s+)?course\s*work\s*[:\-]'
    r'|^\s*courses?\s+taken\s*[:\-]'
    r'|^\s*related\s+courses?\s*[:\-]'
    r'|^\s*academic\s+courses?\s*[:\-]',
    _re.IGNORECASE
)


def _is_coursework_sentence(text: str) -> bool:
    """
    Return True if the sentence is a coursework listing (e.g. 'Coursework: X, Y').
    Detects the structural pattern — not specific course names.
    """
    return bool(_COURSEWORK_HEADER.match(text))


# ── Contact/URL line detector ─────────────────────────────────────────────────
_CONTACT_LINE = _re.compile(
    r'https?://|www\.|linkedin\.com|github\.com'
    r'|\+\d{1,3}[\s\-]?\(?\d'
    r'|@[a-zA-Z0-9]',
    _re.IGNORECASE
)

def _is_contact_line(text: str) -> bool:
    """Return True if the sentence looks like a contact/header line with URLs/emails/phones."""
    return bool(_CONTACT_LINE.search(text))


# ── Education/degree sentence detector ───────────────────────────────────────
# Note: abbreviated forms (B.S., M.S.) must include literal periods to avoid
# matching tool prefixes like "MS Excel" or "BS degree" ambiguously.
_DEGREE_LINE = _re.compile(
    # Full unambiguous degree words
    r'\b(bachelor|master|phd|doctorate)\b'
    # Common degree abbreviations that include periods (b.tech, m.tech, etc.)
    r'|\b(mba|b\.tech|m\.tech|b\.com|m\.com|b\.sc|m\.sc|b\.e\.|m\.e\.)\b'
    # B.S. / M.S. only when followed by "in" or "of" to avoid "MS Excel" false match
    r'|\b(b\.s\.|m\.s\.)\s*(in|of)\b'
    r'|\bconcentration\s+in\b|\bmajor\s+in\b|\bminor\s+in\b'
    r'|\bwith\s+a\s+\w+\s+concentration\b',
    _re.IGNORECASE
)

def _is_education_sentence(text: str) -> bool:
    """Return True if the sentence describes a degree or academic concentration."""
    return bool(_DEGREE_LINE.search(text))


# ── Department listing sentence detector ─────────────────────────────────────
_DEPT_LISTING = _re.compile(
    r'\bdepartments?\b.{0,40}\bincluding\b'
    r'|\bdivisions?\b.{0,40}\bincluding\b'
    r'|\bteams?\b.{0,40}\bincluding\b',
    _re.IGNORECASE
)

def _is_department_listing(text: str) -> bool:
    """Return True if the sentence is listing departments/teams (not skills)."""
    return bool(_DEPT_LISTING.search(text))



DOMAIN     = "technology"
TIER       = 1
THRESHOLD  = 0.45
GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
SPACY_MODEL_NAME  = "en_core_web_md"


# ── Input transformation ─────────────────────────────────────────────────────
# Transforms comma-separated skill lists into sentence-like text so GLiNER
# can detect skills with sufficient context.

_SECTION_HEADERS = {
    "skills", "summary", "experience", "education", "projects",
    "certifications", "objective", "profile", "interests", "references",
    "work experience", "professional experience", "technical skills",
    "core competencies", "achievements", "awards", "publications",
    "volunteer", "activities", "hobbies", "languages",
    "additional skills", "additional strengths", "key strengths",
    "strengths", "key skills", "professional skills", "soft skills",
    "competencies", "areas of expertise", "expertise",
}

_SENTENCE_VERBS = {
    "built", "developed", "created", "designed", "managed", "led",
    "worked", "using", "used", "implemented", "maintained", "delivered",
    "conducted", "established", "collaborated", "partnered", "wrote",
    "applied", "analyzed", "coordinated", "supported", "mentored",
    "defined", "identified", "translated", "engineered", "optimized",
    "leveraged", "automated", "integrated", "resolved", "ensured",
}


def _transform_for_gliner(text: str, label_set: set = None) -> str:
    """
    Transform comma-separated skill lists into GLiNER-friendly sentences.
    Uses only string operations — no regex.
    If label_set is provided, short uppercase tokens (≤5 chars) that match
    a known label get a richer sentence for better GLiNER detection.
    """
    # Normalize bullet/pipe separators to commas before processing
    text = text.replace(' • ', ', ').replace(' | ', ', ').replace('•', ',').replace('|', ',')

    lines = text.split('\n')
    result = []
    pending_header = None  # Track headers on their own line
    pending_saw_item = False  # Whether we've seen at least one item after the header

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Only clear pending_header if we've already processed items
            # This allows empty lines between section header and first item
            if pending_saw_item:
                pending_header = None
                pending_saw_item = False
            result.append(line)
            continue

        # Section headers that contain skill listings (SKILLS, TECHNICAL SKILLS, etc.)
        header_lower = stripped.rstrip(':').strip().lower()
        if header_lower in _SECTION_HEADERS:
            # Skill-related section headers → activate pending_header
            # so subsequent one-per-line items get wrapped
            _SKILL_SECTION_WORDS = {
                "skills", "technical skills", "core competencies",
                "tools", "technologies", "soft skills",
                "strengths", "competencies", "expertise",
            }
            if any(w in header_lower for w in _SKILL_SECTION_WORDS):
                pending_header = header_lower
            else:
                pending_header = None
            result.append(line)
            continue

        # Check for sentence verbs — don't transform actual sentences
        words_lower = stripped.lower().split()
        if any(v in words_lower for v in _SENTENCE_VERBS):
            pending_header = None
            result.append(line)
            continue

        # --- Handle multi-line headers ---
        # If previous line was "Header:" with nothing after colon,
        # keep processing subsequent lines as items until we hit an empty
        # line, a new section header, or a line with verbs.
        if pending_header is not None:
            has_parens = '(' in stripped and ')' in stripped
            if not has_parens:
                if ',' in stripped:
                    items = [i.strip() for i in stripped.split(',') if i.strip()]
                    avg_len = sum(len(i) for i in items) / len(items) if items else 0
                    if avg_len < 25:
                        for item in items:
                            if (label_set and len(item) <= 5
                                    and item.isupper()
                                    and item.lower() in label_set):
                                result.append(
                                    f"Proficient in {item} technology for professional use."
                                )
                            else:
                                result.append(f"Skilled in {item}.")
                        pending_saw_item = True
                        continue
                # Single item on its own line after a header
                if stripped and len(stripped) < 50:
                    # Short uppercase tokens that match a known label get
                    # a richer sentence so GLiNER can detect them
                    if (label_set and len(stripped) <= 5
                            and stripped.isupper()
                            and stripped.lower() in label_set):
                        result.append(
                            f"Proficient in {stripped} technology for professional use."
                        )
                    else:
                        result.append(f"Skilled in {stripped}.")
                    pending_saw_item = True
                    continue
            pending_header = None

        # --- Detect header-only lines (e.g. "IDEs:" or "Cloud Technologies:") ---
        if stripped.endswith(':'):
            header_text = stripped[:-1].strip().lower()
            if header_text not in _SECTION_HEADERS:
                pending_header = header_text
                continue

        # --- "Header: item1, item2, item3" (always active — explicit skill listings) ---
        if ':' in stripped:
            colon_idx = stripped.find(':')
            after_colon = stripped[colon_idx + 1:].strip()
            if ',' in after_colon:
                items = [i.strip() for i in after_colon.split(',') if i.strip()]
                avg_len = sum(len(i) for i in items) / len(items) if items else 0
                if avg_len < 25:
                    for item in items:
                        result.append(f"Skilled in {item}.")
                    continue
            # Single item after colon (e.g., "Database: Oracle DB")
            if after_colon and len(after_colon) < 40 and ',' not in after_colon:
                result.append(f"Skilled in {after_colon}.")
                continue

        # --- Parenthesized and comma-separated lines ---
        # Only transform these when inside a skill-related section (pending_header)
        # to avoid wrapping company names, cities, dates as skills
        if pending_header is not None:
            # Parenthesized sub-lists
            if '(' in stripped and ')' in stripped:
                open_idx = stripped.find('(')
                close_idx = stripped.find(')')
                paren_content = stripped[open_idx + 1:close_idx]
                if ',' in paren_content:
                    prefix = stripped[:open_idx].rstrip().rstrip(',').strip()
                    after = stripped[close_idx + 1:].strip().lstrip(',').strip()
                    if prefix and ',' in prefix:
                        for item in prefix.split(','):
                            item = item.strip()
                            if item:
                                result.append(f"Skilled in {item}.")
                    elif prefix:
                        result.append(f"Skilled in {prefix}.")
                    for item in paren_content.split(','):
                        item = item.strip()
                        if item:
                            result.append(f"Skilled in {item}.")
                    if after and ',' in after:
                        for item in after.split(','):
                            item = item.strip()
                            if item:
                                result.append(f"Skilled in {item}.")
                    elif after:
                        result.append(f"Skilled in {after}.")
                    pending_saw_item = True
                    continue

            # Comma-separated lines
            if ',' in stripped:
                items = [i.strip() for i in stripped.split(',') if i.strip()]
                avg_len = sum(len(i) for i in items) / len(items) if items else 0
                if avg_len < 25:
                    for item in items:
                        result.append(f"Skilled in {item}.")
                    pending_saw_item = True
                    continue

        # Regular line — keep as-is
        result.append(line)

    return '\n'.join(result)


# ── GLiNER Labels: Technology Domain ─────────────────────────────────────────
# Rule: specific named tools/technologies ONLY.
# NO generic categories ("programming language", "web framework").
# NO roles ("client manager", "developer", "engineer").
# NO ambiguous acronyms (HCM, PPM, GL).

class TechLabels:

    # ── Programming Languages ──
    LANGUAGES = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
        "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Scala",
        "SQL", "PL/SQL", "Bash", "PowerShell", "HTML", "CSS", "VBA",
        "R programming", "Perl", "COBOL", "Fortran", "MATLAB",
        "Objective-C", "Dart", "Lua", "Groovy", "Shell",
        "XSLT", "OpenEdge ABL",
    ]

    # ── Data Formats & Markup ──
    DATA_FORMATS = [
        "XML", "JSON", "YAML", "CSV", "XSD", "Markdown",
        "Avro", "Parquet", "Protobuf",
    ]

    # ── Web & Backend Frameworks ──
    WEB_FRAMEWORKS = [
        "React", "Angular", "Vue.js", "Next.js", "Node.js",
        "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET",
        "Express.js", "Laravel", "Ruby on Rails", "Svelte",
        "Nuxt.js", "Gatsby", "Remix", "Ember.js",
    ]

    # ── Mobile ──
    MOBILE = [
        "React Native", "Flutter", "SwiftUI", "Xamarin", "Ionic",
    ]

    # ── Databases & Storage ──
    DATABASES = [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
        "Cassandra", "DynamoDB", "SQL Server", "Snowflake", "Redshift",
        "Oracle", "SQLite", "MariaDB", "CouchDB", "Neo4j",
        "Firebase", "Couchbase", "InfluxDB",
    ]

    # ── Cloud & Infrastructure ──
    CLOUD = [
        "AWS", "Azure", "Google Cloud Platform", "Oracle Cloud",
        "IBM Cloud", "Heroku", "DigitalOcean", "Netlify", "Vercel",
        "Docker", "Kubernetes", "Terraform", "Ansible",
        "Helm", "Istio", "OpenShift", "Rancher", "Vagrant",
    ]

    # ── DevOps & CI/CD ──
    DEVOPS = [
        "Jenkins", "GitHub Actions", "GitLab CI/CD", "ArgoCD",
        "Git", "GitHub", "GitLab", "Bitbucket",
        "CircleCI", "Travis CI", "Bamboo", "TeamCity",
        "CI/CD", "CI/CD Pipelines", "DevOps",
    ]

    # ── Data & ML ──
    DATA_ML = [
        "TensorFlow", "PyTorch", "Scikit-learn",
        "Apache Spark", "Spark", "Apache Kafka", "Kafka", "Apache Airflow",
        "Databricks", "MLflow", "Pandas", "NumPy",
        "Matplotlib", "SciPy", "Seaborn", "Keras",
        "Hadoop", "Elasticsearch", "Elastic Stack",
        "Tableau", "Power BI", "R programming", "Minitab",
        "QlikView", "Looker", "BIRT", "Crystal Reports",
        "dbt", "Talend", "Informatica",
    ]

    # ── Monitoring & Observability ──
    MONITORING = [
        "Prometheus", "Grafana", "Splunk", "Datadog",
        "New Relic", "PagerDuty", "Nagios", "Kibana",
        "ELK Stack", "Dynatrace", "AppDynamics",
    ]

    # ── APIs & Protocols ──
    APIS = [
        "REST API", "GraphQL", "SOAP API", "WebSocket",
        "gRPC", "OpenAPI", "Open API", "Swagger", "SOAP UI",
    ]

    # ── Messaging & Streaming ──
    MESSAGING = [
        "RabbitMQ", "ActiveMQ", "ZeroMQ", "Celery",
        "Amazon SQS", "Azure Service Bus",
    ]

    # ── Security & Networking ──
    SECURITY_NETWORKING = [
        "Active Directory", "SIEM", "Penetration Testing",
        "Burp Suite", "Wireshark", "Cisco", "TCP/IP", "VPN",
        "Tanium", "CrowdStrike", "Nessus", "Metasploit",
        "SonarQube", "Fortify", "OWASP",
    ]

    # ── Enterprise Apps & Platforms ──
    ENTERPRISE = [
        "SAP", "Salesforce", "ServiceNow", "Jira",
        "Workday", "Oracle EBS", "QAD ERP",
        "Linux", "UNIX", "Windows Server",
        "Microservices Architecture", "Test-Driven Development",
    ]

    # ── Collaboration & PM Tools ──
    COLLABORATION = [
        "Slack", "Microsoft Teams", "Confluence", "Trello",
        "Asana", "Monday.com", "Basecamp", "Mural", "Zenhub",
        "Notion", "Airtable",
    ]

    # ── Testing ──
    TESTING = [
        "Selenium", "Jest", "Pytest", "Cypress", "Playwright",
        "JUnit", "TestNG", "Mocha", "Robot Framework",
        "Appium", "Katalon", "Cucumber",
    ]

    # ── IDEs & Dev Tools ──
    DEV_TOOLS = [
        "Jupyter Notebook", "PyCharm", "VS Code", "Spyder",
        "R Studio", "IntelliJ IDEA", "Postman", "Eclipse",
        "Vim", "Sublime Text", "Android Studio", "Xcode",
        "Oracle SQL Developer", "Oxygen XML Developer",
    ]

    # ── Additional Tools ──
    EXTRA_TOOLS = [
        "Excel", "MS Excel", "DAX", "SharePoint", "MS Office", "Office 365",
        "Azure Data Factory", "Oracle Database", "SSIS", "SSRS", "SSAS",
        "Pivot Tables", "Stored Procedures",
        "MS Visio", "MS Project", "Outlook",
    ]

    # ── Soft Skills ──
    SOFT = [
        # Core interpersonal
        "Leadership", "Communication", "Collaboration",
        "Teamwork", "Team Leadership", "Cross-Functional Collaboration",
        # Problem solving & thinking
        "Problem Solving", "Critical Thinking", "Analytical Thinking",
        "Strategic Thinking", "Creative Thinking", "Design Thinking",
        "Logical Thinking", "Systems Thinking",
        # Management & organization
        "Project Management", "Time Management", "Priority Management",
        "Stakeholder Management", "Change Management", "Conflict Resolution",
        "Decision Making", "Risk Assessment",
        # Personal qualities
        "Creativity", "Adaptability", "Resilience", "Self-Motivation",
        "Accountability", "Integrity", "Emotional Intelligence",
        "Growth Mindset", "Continuous Learning",
        # Communication variants
        "Public Speaking", "Presentation Skills", "Written Communication",
        "Active Listening", "Storytelling", "Persuasion", "Negotiation",
        # People development
        "Mentoring", "Coaching", "Training", "Team Building",
        "Talent Development", "Performance Management",
        # Work habits
        "Attention to Detail", "Multitasking", "Organization",
        "Planning", "Resourcefulness", "Initiative",
        # Relationship
        "Relationship Building", "Client Management", "Customer Focus",
        "Networking", "Influencing", "Empathy",
    ]

    # ── CS Fundamentals ──
    CONCEPTS_CS = [
        "Data Structures", "Algorithms", "Object-Oriented Programming",
        "Functional Programming", "Design Patterns", "System Design",
        "Concurrency", "Multithreading", "Recursion", "Operating Systems",
        "Compiler Design", "Computer Networks", "Graph Theory",
        "Dynamic Programming", "Software Architecture", "Version Control",
        "API Design", "Memory Management", "Garbage Collection",
    ]

    # ── Architecture & Infrastructure Concepts ──
    CONCEPTS_ARCHITECTURE = [
        "Distributed Systems", "Event-Driven Architecture",
        "Serverless Computing", "Cloud Computing",
        "Infrastructure as Code", "High Availability",
        "Load Balancing", "Caching", "Service-Oriented Architecture",
        "Edge Computing", "System Integration", "Cloud Integration",
        "System Architecture", "Cloud Architecture",
        "API Management", "API Integration",
        "Microservices", "Monolithic Architecture",
        "Message Queue", "Containerization", "Orchestration",
        "Fault Tolerance", "Disaster Recovery", "Cloud Migration",
        "Horizontal Scaling", "Vertical Scaling",
        "Reverse Proxy", "Service Mesh", "API Gateway",
    ]

    # ── ML/AI Concepts ──
    CONCEPTS_ML = [
        "Machine Learning", "Deep Learning",
        "Natural Language Processing", "Computer Vision",
        "Large Language Models", "Reinforcement Learning",
        "Neural Networks", "Generative AI",
        "Data Science", "Statistical Modeling",
        "Transfer Learning", "Feature Engineering",
        "Predictive Analytics", "Recommendation Systems",
        "Image Recognition", "Speech Recognition",
        "Sentiment Analysis", "Text Classification",
        "Anomaly Detection", "Time Series Analysis",
        "Model Training", "Model Deployment", "MLOps",
        "Hyperparameter Tuning", "Cross Validation",
        "Supervised Learning", "Unsupervised Learning",
        "Clustering", "Regression Analysis", "Classification",
        "Big Data", "Data Mining",
    ]

    # ── Data Concepts ──
    CONCEPTS_DATA = [
        "Data Engineering", "ETL", "Data Warehousing",
        "Data Modeling", "Data Pipeline", "Data Governance",
        "Business Intelligence", "Data Lake", "Stream Processing",
        "Database Design", "Data Integration", "Data Transformation",
        "Data Migration", "Data Quality", "Data Cleansing",
        "Data Visualization", "Data Analysis",
        "Master Data Management", "Data Catalog",
        "Batch Processing", "Real-Time Processing",
        "Schema Design", "Data Architecture",
        "OLAP", "OLTP", "Dimensional Modeling",
        "Data Replication", "Data Synchronization",
        "Stored Procedures", "Database Optimization",
    ]

    # ── Security Concepts ──
    CONCEPTS_SECURITY = [
        "Cybersecurity", "Encryption", "Network Security",
        "Identity and Access Management", "Zero Trust",
        "Threat Modeling", "Vulnerability Assessment",
        "Cloud Security", "Application Security",
        "Penetration Testing", "Security Audit",
        "Incident Response", "Data Privacy", "Compliance",
        "Authentication", "Authorization", "OAuth",
        "Single Sign-On", "Multi-Factor Authentication",
        "SSL/TLS", "Firewall", "Intrusion Detection",
        "Security Operations", "Threat Intelligence",
    ]

    # ── DevOps & Engineering Practices ──
    CONCEPTS_METHODOLOGY = [
        "Agile Methodology", "Scrum", "Kanban",
        "Continuous Integration", "Continuous Deployment",
        "Code Review", "Performance Optimization", "Scalability",
        "Site Reliability Engineering", "Domain-Driven Design",
        "Automation", "Process Automation", "Design Thinking",
        "Test-Driven Development", "Behavior-Driven Development",
        "Pair Programming", "Trunk-Based Development",
        "Blue-Green Deployment", "Canary Deployment",
        "Infrastructure Monitoring", "Log Management",
        "Incident Management", "Capacity Planning",
        "Configuration Management", "Release Management",
        "Technical Documentation", "System Administration",
    ]

    # ── Software Engineering Concepts ──
    CONCEPTS_SOFTWARE = [
        "Full Stack Development", "Frontend Development",
        "Backend Development", "Mobile Development",
        "Web Development", "Desktop Development",
        "Responsive Design", "Progressive Web App",
        "Single Page Application", "Server-Side Rendering",
        "Client-Side Rendering", "State Management",
        "Dependency Injection", "Middleware",
        "Object-Relational Mapping", "Query Optimization",
        "Unit Testing", "Integration Testing",
        "End-to-End Testing", "Load Testing",
        "Performance Testing", "Regression Testing",
        "Debugging", "Profiling", "Refactoring",
        "Code Coverage", "Static Analysis",
        "Package Management", "Build Automation",
    ]

    # ── Cloud & Infra Concepts ──
    CONCEPTS_CLOUD = [
        "Cloud Native", "Multi-Cloud", "Hybrid Cloud",
        "Infrastructure Provisioning", "Auto Scaling",
        "Cost Optimization", "Cloud Governance",
        "Virtual Machine", "Container Registry",
        "Continuous Monitoring", "Observability",
        "Service Discovery", "Secret Management",
        "Network Architecture", "DNS Management",
        "CDN", "Object Storage", "Block Storage",
        "Serverless Functions", "Event-Driven Computing",
    ]

    @classmethod
    def all(cls) -> list:
        return (
            cls.LANGUAGES + cls.DATA_FORMATS +
            cls.WEB_FRAMEWORKS + cls.MOBILE +
            cls.DATABASES + cls.CLOUD + cls.DEVOPS +
            cls.DATA_ML + cls.MONITORING +
            cls.APIS + cls.MESSAGING + cls.SECURITY_NETWORKING +
            cls.ENTERPRISE + cls.COLLABORATION +
            cls.TESTING + cls.DEV_TOOLS + cls.EXTRA_TOOLS +
            cls.CONCEPTS_CS + cls.CONCEPTS_ARCHITECTURE +
            cls.CONCEPTS_ML + cls.CONCEPTS_DATA +
            cls.CONCEPTS_SECURITY + cls.CONCEPTS_METHODOLOGY +
            cls.CONCEPTS_SOFTWARE + cls.CONCEPTS_CLOUD +
            cls.SOFT
        )

    @classmethod
    def batches(cls) -> list:
        """Return labels grouped into ~7 batches for speed."""
        return [
            cls.LANGUAGES + cls.DATA_FORMATS + cls.WEB_FRAMEWORKS + cls.MOBILE,
            cls.DATABASES + cls.CLOUD + cls.DEVOPS + cls.MESSAGING,
            cls.DATA_ML + cls.MONITORING + cls.APIS + cls.SECURITY_NETWORKING,
            cls.ENTERPRISE + cls.COLLABORATION + cls.TESTING + cls.DEV_TOOLS + cls.EXTRA_TOOLS,
            cls.CONCEPTS_CS + cls.CONCEPTS_ARCHITECTURE + cls.CONCEPTS_CLOUD + cls.SOFT,
            cls.CONCEPTS_ML + cls.CONCEPTS_DATA + cls.CONCEPTS_SECURITY,
            cls.CONCEPTS_METHODOLOGY + cls.CONCEPTS_SOFTWARE,
        ]


GLINER_LABELS: List[str] = TechLabels.all()


# ── Skill Classification ─────────────────────────────────────────────────────
# Build a name → category lookup from ALL label lists.
# Skill name match takes priority over GLiNER label (which can be wrong).

_SKILL_CATEGORY_MAP = {}

for _label in (TechLabels.LANGUAGES + TechLabels.DATA_FORMATS +
               TechLabels.WEB_FRAMEWORKS + TechLabels.MOBILE +
               TechLabels.DATABASES + TechLabels.CLOUD + TechLabels.DEVOPS +
               TechLabels.DATA_ML + TechLabels.MONITORING + TechLabels.APIS +
               TechLabels.MESSAGING + TechLabels.SECURITY_NETWORKING +
               TechLabels.ENTERPRISE + TechLabels.COLLABORATION +
               TechLabels.TESTING + TechLabels.DEV_TOOLS + TechLabels.EXTRA_TOOLS):
    _SKILL_CATEGORY_MAP[_label.lower()] = "hard_skill"

for _label in (TechLabels.CONCEPTS_CS + TechLabels.CONCEPTS_ARCHITECTURE +
               TechLabels.CONCEPTS_ML + TechLabels.CONCEPTS_DATA +
               TechLabels.CONCEPTS_SECURITY + TechLabels.CONCEPTS_METHODOLOGY +
               TechLabels.CONCEPTS_SOFTWARE + TechLabels.CONCEPTS_CLOUD):
    _SKILL_CATEGORY_MAP[_label.lower()] = "concept"

for _label in TechLabels.SOFT:
    _SKILL_CATEGORY_MAP[_label.lower()] = "soft_skill"

# Sets for label-based fallback
_CONCEPT_LABELS = {k for k, v in _SKILL_CATEGORY_MAP.items() if v == "concept"}
_SOFT_LABELS = {k for k, v in _SKILL_CATEGORY_MAP.items() if v == "soft_skill"}


def _classify_skill(skill_name: str, skill_labels: list) -> str:
    """Classify a skill as 'hard_skill', 'concept', or 'soft_skill'.
    Checks skill name against curated lists FIRST (reliable),
    falls back to GLiNER label (which can be noisy)."""
    name_lower = skill_name.lower()
    # Priority: skill name match against our curated label lists
    if name_lower in _SKILL_CATEGORY_MAP:
        return _SKILL_CATEGORY_MAP[name_lower]
    # Fallback: check GLiNER-assigned labels
    for label in skill_labels:
        label_lower = label.lower()
        if label_lower in _SOFT_LABELS:
            return "soft_skill"
        if label_lower in _CONCEPT_LABELS:
            return "concept"
    return "hard_skill"


# ── Legacy stubs (kept so tier2/tier3 import calls don't break) ───────────────
# These are empty — all keyword-list logic has been removed.
NON_SKILL_BLOCKLIST: set = set()
SEED_SKILLS: Dict[str, Any] = {}
SKILL_LABEL_OVERRIDES: Dict[str, str] = {}



class TechnologySkillExtractor:
    """
    Extracts technology skills from resume text using GLiNER.
    Uses the Technology (Tier 1) label set exclusively.
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
            domain_name="technology",
            label_batches=TechLabels.batches(),
            original_doc=original_doc,
        )


# ── Shared extraction pipeline (reused by all tier extractors) ───────────────

def _run_extraction_pipeline(
    resume_text: str,
    nlp,
    gliner_model,
    labels: list,
    threshold: float,
    min_freq: int,
    domain_name: str,
    label_batches: list = None,
    original_doc=None,
    # Legacy params kept for backward compatibility — ignored
    non_skill_blocklist: set = None,
    skill_label_overrides: dict = None,
    seed_skills: dict = None,
) -> List[Dict[str, Any]]:
    """
    Extraction pipeline shared across all domain extractors.
    Uses only GLiNER + GLINER_LABELS. No keyword lists.
    Transforms list-like text into sentences before GLiNER for better recall.
    When label_batches is provided, runs GLiNER with small label batches
    for significantly better accuracy.
    """
    import time as _time
    _t0 = _time.time()

    # ── Step 1: Preprocess ──
    label_set_lower = {l.lower() for l in labels}
    transformed_text = _transform_for_gliner(resume_text, label_set=label_set_lower)

    # Precompute label patterns for long sentence gating (skip sentences with no skills)
    _label_patterns = [
        _re.compile(r'\b' + _re.escape(label.lower()) + r'\b')
        for label in labels
        if len(label) > 3
    ]
    gliner_doc = nlp(transformed_text)
    if original_doc is None:
        original_doc = nlp(resume_text)
    raw_entities = []

    _t1_ = _time.time()
    print(f"   [TIMING] spaCy + preprocess: {_t1_ - _t0:.2f}s")

    label_allowlist = {l.lower() for l in labels}

    if label_batches is None:
        label_batches = [labels]

    # ── Step 2: Extract skills ──
    # For transformed sentences ("Experience with X." / "Proficient in X..."),
    # if X matches a known label, add directly — no GLiNER call needed.
    # Only run GLiNER on natural/contextual sentences.
    gliner_sentences = []
    for sent in gliner_doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 10:
            continue
        if _is_coursework_sentence(sent_text):
            continue
        if len(sent_text) <= 300:
            if _is_contact_line(sent_text):
                continue
            if _is_education_sentence(sent_text):
                continue
            if _is_department_listing(sent_text):
                continue
        else:
            if _is_contact_line(sent_text):
                continue

        # Skip long narrative sentences with no skill signals
        if len(sent_text) > 180:
            sent_lower = sent_text.lower()
            if not any(p.search(sent_lower) for p in _label_patterns):
                continue

        # Direct extraction for transformed skill-listing sentences
        # Check each line within the sentence (spaCy may group multiple lines)
        _direct_matches = 0
        for line in sent_text.split('\n'):
            line = line.strip()
            for prefix in ("Skilled in ", "Proficient in "):
                if line.startswith(prefix):
                    skill_text = line[len(prefix):].strip()
                    for sfx in (" technology for professional use.",):
                        if skill_text.endswith(sfx):
                            skill_text = skill_text[:-len(sfx)].strip()
                            break
                    skill_text = skill_text.rstrip('.')
                    if skill_text.lower() in label_allowlist:
                        raw_entities.append((
                            skill_text, skill_text, 0.90,
                            sent_text, skill_text,
                        ))
                        _direct_matches += 1
                    break
        if _direct_matches == 0:
            gliner_sentences.append(sent_text)

    valid_sentences = gliner_sentences

    # Track already-extracted skills — skip them in GLiNER processing
    already_extracted = {e[0].lower() for e in raw_entities}

    # Group sentences into chunks by character budget to reduce GLiNER calls.
    # Sentences under 750 chars get merged together until the chunk reaches ~750 chars.
    # Very long sentences (>= 750 chars) become their own chunk.
    _CHUNK_BUDGET = 750
    chunks = []
    _buffer = []
    _buffer_len = 0
    for sent_text in valid_sentences:
        if len(sent_text) >= _CHUNK_BUDGET:
            # Flush buffer first, then add long sentence as its own chunk
            if _buffer:
                chunks.append(' '.join(_buffer))
                _buffer = []
                _buffer_len = 0
            chunks.append(sent_text)
        else:
            _buffer.append(sent_text)
            _buffer_len += len(sent_text)
            if _buffer_len >= _CHUNK_BUDGET:
                chunks.append(' '.join(_buffer))
                _buffer = []
                _buffer_len = 0
    if _buffer:
        chunks.append(' '.join(_buffer))

    _t2_ = _time.time()
    print(f"   [TIMING] Sentence prep: {_t2_ - _t1_:.2f}s")
    print(f"   [TIMING] Chunks to process: {len(chunks)}, Batches per chunk: {len(label_batches)}, Total GLiNER calls: {len(chunks) * len(label_batches)}")

    for chunk_text in chunks:
        chunk_threshold = threshold
        if "Proficient in" in chunk_text and "technology for professional use" in chunk_text:
            chunk_threshold = 0.25

        all_entities = []
        for batch in label_batches:
            try:
                entities = gliner_model.inference(
                    chunk_text, batch, threshold=chunk_threshold, relations=[]
                )
                if entities:
                    if isinstance(entities[0], list):
                        entities = entities[0]
                    if not isinstance(entities[0], dict):
                        entities = []
                    all_entities.extend(entities)
            except Exception:
                pass

        best_by_text = {}
        for ent in all_entities:
            if not isinstance(ent, dict):
                continue
            text_key = ent.get("text", "").strip().lower()
            if text_key not in best_by_text or ent.get("score", 0) > best_by_text[text_key].get("score", 0):
                best_by_text[text_key] = ent
        entities = list(best_by_text.values())

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            ent_text  = ent.get("text", "").strip()
            ent_score = ent.get("score", 0.5)
            ent_label = ent.get("label", "")

            if len(ent_text) < 2 or not any(c.isalpha() for c in ent_text):
                continue

            normalized = normalize_skill_text(ent_text)
            if not normalized or len(normalized) < 2:
                continue

            # Skip if already extracted from skill section
            if normalized.lower() in already_extracted:
                continue

            if not passes_noise_filter(normalized, original_doc, label_allowlist=label_allowlist):
                continue

            raw_entities.append((normalized, ent_text, ent_score, chunk_text, ent_label))

    _t3_ = _time.time()
    print(f"   [TIMING] GLiNER inference: {_t3_ - _t2_:.2f}s")

    # Build label casing map: "networking" → "Networking", "hubspot" → "HubSpot"
    label_casing = {l.lower(): l for l in labels}
    skills = _build_skill_list(raw_entities, resume_text, min_freq, label_casing=label_casing)

    # ── Step 3: Hard filters (ORG removal, context validation) ──
    skills = _apply_hard_filters(skills, original_doc, label_allowlist)

    # ── Step 4: Merge & dedup ──
    skills = _deduplicate_skills(skills)

    # ── Step 5: Final scoring ──
    skills = _apply_final_scoring(skills)

    # ── Step 6: Remove concept noise ──
    _CONCEPT_BLOCKLIST = {"esp", "organization", "operations", "strategy"}
    skills = [
        s for s in skills
        if not (s.get("category") == "concept" and s["skill"].lower() in _CONCEPT_BLOCKLIST)
    ]

    _t4_ = _time.time()
    print(f"   [TIMING] Post-processing: {_t4_ - _t3_:.2f}s")
    print(f"   [TIMING] TOTAL pipeline: {_t4_ - _t0:.2f}s")

    skills.sort(key=lambda x: x["first_occurrence_pos"])
    print(f"   Extracted {len(skills)} {domain_name} skills (threshold={threshold})")
    return skills


# ── Step 3: Hard filters ─────────────────────────────────────────────────────

def _is_org_entity(span_text: str, doc) -> bool:
    """Return True if the span overlaps with a spaCy ORG entity in the doc."""
    span_lower = span_text.lower()
    for ent in doc.ents:
        if ent.label_ == "ORG" and span_lower in ent.text.lower():
            return True
    return False


def _has_tech_context(context: str) -> bool:
    """Return True if the context sentence contains technology-related words."""
    ctx = context.lower()
    _TECH_CONTEXT_WORDS = (
        "developed", "built", "implemented", "used", "using",
        "experience", "working", "designed", "integrated",
        "platform", "system", "tool", "framework", "api",
        "proficient", "skilled", "expertise", "technologies",
        "programming", "database", "cloud", "server", "data",
        "deployed", "configured", "automated", "managed",
        "languages", "stack", "environment", "infrastructure",
        "software", "application", "library", "sdk", "ide",
    )
    return any(w in ctx for w in _TECH_CONTEXT_WORDS)


def _apply_hard_filters(
    skills: list,
    original_doc,
    label_allowlist: set,
) -> list:
    """Remove ORG leakage and validate context for non-label skills."""
    filtered = []
    for s in skills:
        skill_name = s["skill"]
        skill_lower = skill_name.lower()

        # ── ORG filter ──
        # If spaCy thinks it's an ORG AND it's not in our label list, skip it.
        # (Salesforce IS a label → keep it; random company names → remove)
        if _is_org_entity(skill_name, original_doc) and skill_lower not in label_allowlist:
            continue

        # ── Context validation for non-label skills ──
        # Skills in our label list are trusted. Unlisted skills need tech context.
        if skill_lower not in label_allowlist:
            if not _has_tech_context(s.get("context", "")):
                continue

        filtered.append(s)
    return filtered


# ── Step 4: Deduplication ─────────────────────────────────────────────────────

def _normalize_skill_name(name: str) -> str:
    """Normalize skill name for dedup comparison."""
    n = name.lower()
    n = n.replace("microsoft ", "").replace("ms ", "")
    return n.strip()


def _deduplicate_skills(skills: list) -> list:
    """Plural/possessive normalization + bidirectional substring dedup."""

    def _base_form(s: str) -> str:
        if s.endswith("'s"):
            return s[:-2]
        if s.endswith("s") and len(s) > 3:
            return s[:-1]
        return s

    # Merge plural variants
    base_map: Dict[str, int] = {}
    for i, s in enumerate(skills):
        b = _base_form(_normalize_skill_name(s["skill"]))
        if b in base_map:
            canonical_idx = base_map[b]
            canonical = skills[canonical_idx]
            if s["confidence"] > canonical["confidence"]:
                skills[canonical_idx] = {**s, "skill": canonical["skill"]}
            skills[i] = None  # type: ignore
        else:
            base_map[b] = i
    skills = [s for s in skills if s is not None]

    # Bidirectional whole-word substring dedup
    def _skill_substring(needle: str, haystack: str) -> bool:
        pat = r'(?<![/\-._\w])' + _re.escape(needle) + r'(?![/\-._\w])'
        return bool(_re.search(pat, haystack))

    to_remove: set = set()
    for i, s in enumerate(skills):
        if i in to_remove:
            continue
        for j, other in enumerate(skills):
            if i == j or j in to_remove:
                continue
            si = s["skill"].lower()
            sj = other["skill"].lower()
            if si == sj:
                continue
            i_in_j = _skill_substring(si, sj)
            j_in_i = _skill_substring(sj, si)
            if i_in_j or j_in_i:
                if s["confidence"] > other["confidence"]:
                    to_remove.add(j)
                elif other["confidence"] > s["confidence"]:
                    to_remove.add(i)
                    break
                else:
                    if len(si) >= len(sj):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
    skills = [s for i, s in enumerate(skills) if i not in to_remove]

    # Fuzzy merge (case variants like Javascript/JavaScript)
    try:
        from rapidfuzz import fuzz
        merged = []
        for s in skills:
            found = False
            for m in merged:
                if fuzz.ratio(s["skill"].lower(), m["skill"].lower()) > 90:
                    if s["confidence"] > m["confidence"]:
                        m["skill"] = s["skill"]
                        m["confidence"] = s["confidence"]
                    m["frequency"] = m.get("frequency", 1) + s.get("frequency", 1)
                    found = True
                    break
            if not found:
                merged.append(s)
        skills = merged
    except ImportError:
        pass  # rapidfuzz not installed — skip fuzzy merge

    # Length-based dominance dedup: keep the most specific skill
    # among overlapping ones (e.g., "SAP FICO" > "SAP", "Microsoft Excel" > "Excel")
    skills = sorted(skills, key=lambda x: len(x["skill"]), reverse=True)
    selected = []
    for s in skills:
        name = s["skill"].lower()
        if not any(name in existing["skill"].lower() for existing in selected):
            selected.append(s)
    skills = selected

    return skills


# ── Step 5: Final scoring ─────────────────────────────────────────────────────

def _final_score(skill: dict) -> float:
    """Compute a composite score from confidence, frequency, and phrase shape.
    Single-word skills get a penalty (more ambiguous than multi-word)."""
    confidence = skill.get("confidence", 0)
    frequency = skill.get("frequency", 1)
    word_count = len(skill.get("skill", "").split())

    score = (
        confidence * 0.6
        + min(frequency, 3) / 3 * 0.3
        + (0.1 if word_count > 1 else 0)
    )

    # Single-word penalty: "Progress", "Oracle", "Platform" are ambiguous
    # But short uppercase acronyms (SQL, SEO, ETL) are specific — no penalty
    skill_text = skill.get("skill", "")
    if word_count == 1 and not (len(skill_text) <= 5 and skill_text.isupper()):
        score -= 0.1

    return score


def _apply_final_scoring(
    skills: list,
    min_score: float = 0.40,
    hard_threshold: float = 0.65,
    concept_threshold: float = 0.50,
    soft_threshold: float = 0.60,
    context_threshold: float = 0.70,
) -> list:
    """Remove weak signals with category-aware thresholds.
    Thresholds are configurable per domain."""
    result = []
    for s in skills:
        confidence = s.get("confidence", 0)
        frequency = s.get("frequency", 1)
        ctx = s.get("context", "").lower()
        category = s.get("category", "hard_skill")
        from_skill_section = "skilled in" in ctx or "proficient in" in ctx

        # Section-aware boost: skills from "Skills" section get a confidence bump
        if from_skill_section:
            s["confidence"] = min(s["confidence"] * 1.15, 1.0)
            confidence = s["confidence"]

        # Multi-word skills are more specific — boost confidence
        word_count = len(s.get("skill", "").split())
        if word_count >= 2:
            s["confidence"] = min(s["confidence"] * 1.1, 1.0)
            confidence = s["confidence"]

        # ── Category-specific thresholds ──
        effective_hard = hard_threshold if word_count == 1 else 0.50
        effective_context = context_threshold if word_count == 1 else 0.55

        if category == "hard_skill":
            if confidence < effective_hard and frequency == 1:
                continue
            if not from_skill_section and confidence < effective_context:
                continue

        elif category == "concept":
            if confidence < concept_threshold:
                continue

        elif category == "soft_skill":
            if confidence < soft_threshold:
                continue

        score = _final_score(s)
        if score >= min_score:
            result.append(s)
    return result


# ── Shared builder (reused by all tier extractors) ────────────────────────────

def _build_skill_list(
    raw_entities: list,
    source_text: str,
    min_freq: int,
    label_casing: dict = None,
) -> List[Dict[str, Any]]:
    if not raw_entities:
        return []

    skill_map: Dict[str, Dict[str, Any]] = {}
    for normalized, original, score, context, label in raw_entities:
        key = normalized.lower()
        # Use canonical label casing if available (e.g., "networking" → "Networking")
        if label_casing and key in label_casing:
            normalized = label_casing[key]
        if key not in skill_map:
            pos = source_text.lower().find(normalized.lower())
            if pos == -1:
                pos = source_text.lower().find(original.lower())
            if pos == -1:
                pos = len(source_text)

            skill_map[key] = {
                "skill":               normalized,
                "original_forms":      set(),
                "scores":              [],
                "contexts":            [],
                "labels":              set(),
                "count":               0,
                "first_occurrence_pos": pos,
            }

        skill_map[key]["original_forms"].add(original)
        skill_map[key]["scores"].append(score)
        if context not in skill_map[key]["contexts"]:
            skill_map[key]["contexts"].append(context)
        skill_map[key]["labels"].add(label)
        skill_map[key]["count"] += 1

    result = []
    for key, info in skill_map.items():
        freq = info["count"]
        if freq < min_freq:
            continue
        max_score  = max(info["scores"])
        confidence = min(max_score + 0.05 * (freq - 1), 1.0)
        best_ctx   = max(info["contexts"], key=len) if info["contexts"] else ""

        labels = list(info["labels"])

        category = _classify_skill(info["skill"], labels)

        result.append({
            "skill":               info["skill"],
            "frequency":           freq,
            "confidence":          round(confidence, 4),
            "context":             best_ctx,
            "labels":              labels,
            "category":            category,
            "first_occurrence_pos": info["first_occurrence_pos"],
        })

    result.sort(key=lambda x: x["first_occurrence_pos"])
    return result
