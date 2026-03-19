"""Build functional cluster groupings from parsed taxonomy roles."""

from src.taxonomy import get_categories


def build_clusters(roles: list[dict]) -> list[dict]:
    """Build functional subclusters from taxonomy roles.

    Splitting rules:
    - Categories with <10 roles: single cluster (label = category name)
    - Categories with 10-15 roles: 2 subclusters
    - Categories with 16-25 roles: 2-3 subclusters
    - Categories with 26+ roles: 3-4 subclusters

    Args:
        roles: List of {"role": str, "category": str} dicts.

    Returns:
        List of {"cluster_label": str, "category": str, "roles": [str, ...]} dicts.

    Raises:
        ValueError: If any role is missing from cluster assignments or
                    cluster definitions reference roles not in the taxonomy.
    """
    categories = get_categories(roles)

    # Hardcoded subclusters keyed by category name.
    # Each value is a list of (cluster_label, [role_names]) tuples.
    _CLUSTER_DEFS: dict[str, list[tuple[str, list[str]]]] = {
        # ── 5 largest categories (spec-provided) ──────────────────────
        "Information Technology": [
            ("IT Operations & Support", [
                "IT Administrator", "IT Support Specialist",
                "IT Service Desk Manager", "Desktop Engineer",
                "IT Professional", "IT Operations Analyst",
            ]),
            ("IT Infrastructure & Engineering", [
                "Cloud Administrator", "Network Administrator",
                "Network Engineer", "DevOps Engineer",
                "Site Reliability Engineer (SRE)", "Systems Administrator",
                "Configuration Manager",
            ]),
            ("IT Security & Governance", [
                "GRC Analyst", "Identity & Access Management Analyst",
                "Incident Response Coordinator", "IT Compliance Analyst",
                "IT Governance Analyst", "SOC Manager", "IT Asset Manager",
                "IT Financial Management (FinOps) Analyst", "IT Vendor Manager",
                "IT Procurement Analyst",
            ]),
            ("IT Product & Platform", [
                "Solutions Architect", "Business Systems Analyst",
                "QA Analyst (Software)", "Release Train Engineer",
                "Scrum Program Lead", "Service Delivery Manager",
                "Technical Program Manager", "SharePoint Administrator",
                "M365 Administrator", "Power Platform Admin",
                "CMDB Administrator", "Launch Infrastructure Manager",
                "Observability/Monitoring Analyst", "Field IT Manager",
                "IT Product Manager", "Platform Product Manager",
            ]),
        ],
        "Finance": [
            ("Financial Accounting & Reporting", [
                "Accountant", "Senior Accountant", "Revenue Accountant",
                "Cost Accountant", "Fixed Assets Accountant",
                "Financial Accounting", "Financial Reporting Analyst",
                "Controller", "Bookkeeper", "Hedge Accounting Specialist",
            ]),
            ("Financial Planning & Analysis", [
                "Financial Analyst", "FP&A Manager",
                "Financial Planning and Analysis", "Financial Operations",
                "Capital Planning Analyst", "Billing Analyst", "M&A Analyst",
                "Valuation Analyst", "Finance Business Partner",
                "Finance Manager", "ALM Analyst", "Vendor Engagement Manager",
            ]),
            ("Tax, Compliance & Treasury", [
                "Tax Manager", "Tax Specialist",
                "Income Tax Compliance Manager", "SOX Compliance Analyst",
                "Internal Auditor",
                "Audit, Risk and Compliance (ARC) Data Solution Manager",
                "Treasury Analyst", "Treasury Operations Analyst",
                "AP Specialist", "AR Specialist", "Payroll Specialist",
                "Credit Analyst",
            ]),
        ],
        "Design": [
            ("UX Design & Research", [
                "User Experience Designer", "User Experience Researcher",
                "UX Researcher", "User Experience Writer", "UX Writer",
                "Interaction Designer", "Service Designer",
                "Experience Designer", "Design Researcher",
                "Information Architect",
            ]),
            ("Visual & Product Design", [
                "Graphic Designer", "Brand Designer", "Visual Designer",
                "Motion Designer", "Packaging Designer", "Industrial Designer",
                "Material Designer", "Environmental Designer",
                "Product Designer", "Content Designer",
                "Information Designer", "Design Engineer",
            ]),
            ("Design Leadership & Operations", [
                "Creative Director", "Design Director", "Experience Director",
                "Design Operations Manager", "Design Program Manager",
                "Design Strategist", "Design Systems Manager",
                "Brand Strategist", "Research Operations Manager",
                "Video Producer (Admin)",
            ]),
        ],
        "Software Engineering": [
            ("Application Development", [
                "Frontend Software Engineer", "Backend Software Engineer",
                "Full-Stack Software Engineer",
                "Mobile Software Engineer (Android)",
                "Mobile Software Engineer (iOS)", "Web Developer",
                "Game Developer", "Software Engineer",
                "AR/VR Software Engineer",
            ]),
            ("Platform & Infrastructure", [
                "Cloud Engineer", "Platform Engineer",
                "Systems Software Engineer", "DevSecOps Engineer",
                "Build/Release Engineer", "Embedded Software Engineer",
                "Firmware Engineer", "Data Platform Engineer",
                "Graphics/Rendering Engineer", "Robotics Software Engineer",
            ]),
            ("Engineering Management & Quality", [
                "Engineering Manager", "Software Development Manager",
                "Principal Software Engineer", "Staff Software Engineer",
                "Quality Engineer (Software)",
                "SDET (Software Development Engineer in Test)",
                "Test Automation Engineer", "Tools Engineer", "MLOps Engineer",
                "Solutions Engineer (Pre-Sales)",
                "Application Security Engineer",
            ]),
        ],
        "Education": [
            ("Teaching & Academic Programs", [
                "Professor", "Teacher", "Academic Advisor",
                "Curriculum Developer", "Student Success Advisor",
                "Career Services Coordinator", "Corporate Trainer",
            ]),
            ("Student Services & Enrollment", [
                "Admissions Counselor", "Financial Aid Officer",
                "Enrollment Manager", "Enrollment Marketing Manager",
                "International Student Services Coordinator", "Registrar",
                "Student Records Coordinator", "Bursar",
                "Bursar Operations Analyst", "Scheduling Officer",
            ]),
            ("Educational Technology & Administration", [
                "Instructional Designer", "Instructional Technologist",
                "eLearning Developer", "Learning Experience Designer",
                "Institutional Research Analyst",
                "Registrar Systems Analyst",
                "Assessment & Accreditation Coordinator",
                "Educational Program Coordinator",
                "Department Coordinator", "Dean's Office Administrator",
                "School Administrator", "Alumni Relations Manager",
            ]),
        ],

        # ── Remaining 37 categories ───────────────────────────────────

        # Human Resources (26 → 3 clusters)
        "Human Resources": [
            ("HR Strategy & Business Partnership", [
                "HR Business Partner", "HR Manager", "HR Generalist",
                "HR Consultant", "People Operations Specialist",
                "DEI Program Manager", "Employee Experience Manager",
                "Organizational Development Specialist",
            ]),
            ("Talent Acquisition & Development", [
                "Recruiter", "HR Recruiting Manager",
                "Talent Acquisition Coordinator", "Talent Operations Manager",
                "Learning & Development Specialist", "People Analytics Analyst",
                "HRIS Analyst",
            ]),
            ("HR Operations & Compliance", [
                "HR Administrator", "HR Compliance Officer",
                "HR Data Privacy Specialist", "HR Senior Service Manager",
                "HR Service Manager", "HR Support Advisor",
                "Benefits Administrator", "Compensation & Benefits Analyst",
                "Compensation Manager", "Payroll Manager",
                "Employee Relations Specialist",
            ]),
        ],

        # Sales (17 → 2 clusters)
        "Sales": [
            ("Enterprise & Strategic Sales", [
                "Account Executive", "Enterprise Account Executive",
                "Cloud Solution Architect", "Commercial Executive",
                "Enterprise Solution Specialist",
                "Digital Sales Specialist at Microsoft",
                "Senior Sales Specialist at Microsoft",
                "Digital Account Executive",
            ]),
            ("Account Management & Sales Operations", [
                "Account Manager", "Channel Account Manager",
                "Customer Success Account Manager", "Partner Account Manager",
                "Sales Administrator", "Sales Enablement Manager",
                "Sales Manager", "Sales Operations Analyst",
                "Sales Representative",
            ]),
        ],

        # Legal (20 → 2 clusters)
        "Legal": [
            ("Corporate & Transactional Law", [
                "Corporate Counsel", "Principal Corporate Counsel",
                "Product Counsel", "Contract Administrator",
                "Contracts Manager", "Legal Assistant", "Paralegal",
                "IP Paralegal", "Legal Operations Manager",
                "Trademark Specialist",
            ]),
            ("Compliance, Privacy & Regulatory", [
                "Compliance Officer", "AML Compliance Analyst",
                "Anti-Corruption Compliance Analyst", "Data Privacy Analyst",
                "Data Protection Officer", "Regulatory Affairs Specialist",
                "Policy Governance Manager", "eDiscovery Specialist",
                "Litigation Support Specialist",
                "Records & eDiscovery Manager",
            ]),
        ],

        # Marketing (26 → 3 clusters)
        "Marketing": [
            ("Digital & Growth Marketing", [
                "Digital Marketing Specialist", "Growth Marketing Manager",
                "SEO/SEM Specialist", "Social Media Manager",
                "Demand Generation Manager", "ABM Manager",
                "Lifecycle Marketing Manager", "Affiliate Marketing Manager",
                "Influencer Marketing Manager",
            ]),
            ("Brand & Content Marketing", [
                "Brand Manager", "Content Marketer", "Copy Chief",
                "Product Marketing Manager", "Customer Marketing Manager",
                "Integrated Marketing", "Partner Marketing Manager",
                "Field Marketing Manager", "Events Manager",
            ]),
            ("Marketing Operations & Media", [
                "Marketing Coordinator", "Marketing Manager",
                "Marketing Operations Manager", "Campaign Operations Manager",
                "CRM Administrator", "Media Buyer", "Media Planner",
                "Traffic Manager",
            ]),
        ],

        # Communications (17 → 2 clusters)
        "Communications": [
            ("External Communications & PR", [
                "Communications Manager", "Communications VP",
                "Public Relations Manager", "Crisis Communications Manager",
                "Issues Management Manager", "Brand Social Communications Manager",
                "Newsroom Assignment Editor", "Rights & Clearances Manager",
                "Communications Specialist",
            ]),
            ("Content & Editorial Operations", [
                "Content Strategist", "Copywriter", "Editor",
                "Technical Writer", "Editorial Operations Manager",
                "Internal Communications Director",
                "Internal Communications Manager",
                "Podcast Producer (Admin)",
            ]),
        ],

        # Administrative Support (21 → 2 clusters)
        "Administrative Support": [
            ("Executive & Office Administration", [
                "Executive Assistant", "Executive Scheduler",
                "Executive Communications Coordinator", "Office Administrator",
                "Office Manager", "Office Policies Administrator",
                "Front Desk Coordinator", "Receptionist",
                "Facilities Coordinator", "Corporate Event Coordinator",
            ]),
            ("Document Management & Coordination", [
                "Administrative Assistant", "Administrative Coordinator",
                "Administrative Specialist", "Administrative Support",
                "Corporate Archivist", "Document Control Specialist",
                "Document Specialist", "Procurement Coordinator",
                "Project Coordinator", "Records Manager",
                "Travel Coordinator",
            ]),
        ],

        # Engineering (15 → 2 clusters)
        "Engineering": [
            ("Design & Systems Engineering", [
                "BIM Coordinator", "CAD Manager",
                "Computer Systems Engineers", "Configuration Engineer",
                "Automation Engineer (Office)", "Software Developers",
                "Technical Writer (Engineering)",
            ]),
            ("Manufacturing & Quality Engineering", [
                "Industrial Engineer", "Process Engineer",
                "Quality Assurance Manager", "Quality Engineer",
                "Reliability Engineer",
                "Manufacturing Quality Systems Manager",
                "New Product Introduction (NPI) Program Manager",
                "PLM Analyst",
            ]),
        ],

        # Executive Offices (19 → 2 clusters)
        "Executive Offices": [
            ("C-Suite & Strategy", [
                "CEO", "Chief Data Officer", "Chief Information Officer",
                "Chief Technology Officer", "Chief of Staff",
                "Corporate Development Manager", "Corporate Strategy Manager",
                "Strategy Analyst", "Strategy Associate",
                "Innovation Program Manager",
            ]),
            ("Governance, Risk & Compliance", [
                "Enterprise Risk Manager", "ESG Reporting Manager",
                "Ethics & Compliance Manager", "Executive (VP/Director)",
                "Knowledge Manager", "Policy & Governance Manager",
                "Portfolio Governance Lead", "Records & Information Manager",
                "Sustainability Program Manager",
            ]),
        ],

        # Customer Service (23 → 3 clusters)
        "Customer Service": [
            ("Customer Support & Service Delivery", [
                "Customer Service Agent", "Customer Service Representative",
                "Customer Support Representative", "Customer Service Manager",
                "Customer Support Manager", "Customer Support Director",
                "Field Support Coordinator",
            ]),
            ("Customer Success & Engagement", [
                "Customer Success Manager", "Customer Education Manager",
                "Customer Insights Program Manager (CS)",
                "Community Manager", "Onboarding Specialist (CS)",
                "Professional Services Coordinator",
                "Renewals Manager", "Technical Account Manager",
            ]),
            ("Technical Support & Operations", [
                "Support Engineer", "Technical Advisor",
                "Escalation Engineer", "Escalation Manager",
                "Escalations Manager", "Knowledge Base Manager",
                "Support Operations Analyst",
                "Support Readiness Manager",
            ]),
        ],

        # Operations (22 → 2 clusters)
        "Operations": [
            ("Business Analysis & Consulting", [
                "Business Analyst", "Business Manager",
                "Management Consultant", "Operations Analyst",
                "Operations Specialist", "Inventory Analyst",
                "Production Planner", "S&OP Planner",
                "Supply Chain Analyst", "Logistics Coordinator",
                "Procurement Manager", "Procurement Specialist",
            ]),
            ("Program & Change Management", [
                "Operations Manager", "Operations PM",
                "Operations Program Manager", "Project Manager",
                "Change Manager", "Business Continuity Manager",
                "Internal Manager", "Quality Assurance (Operations)",
                "Shift Manager", "Vendor Manager",
            ]),
        ],

        # Project Management (16 → 2 clusters)
        "Project Management": [
            ("Product & Portfolio Management", [
                "Product Manager", "Product Owner",
                "Product Operations Manager", "Technical Product Manager",
                "Portfolio Manager", "OKR Program Manager",
                "Business Relationship Manager",
            ]),
            ("Project Execution & Agile", [
                "Program Manager", "Agile Coach", "Scrum Master",
                "Release Manager", "Business Project Analyst",
                "Change Enablement Manager", "PMO Analyst",
                "Project Controls Analyst", "Project Scheduler",
            ]),
        ],

        # Data & Analytics (24 → 3 clusters)
        "Data & Analytics": [
            ("Data Engineering & Management", [
                "Data Engineer", "Data Curator", "Data Steward",
                "Data Governance Analyst", "Data Product Manager",
                "BI Developer",
            ]),
            ("Data Science & Machine Learning", [
                "Data Scientist", "Machine Learning Engineer",
                "NLP Analyst", "Operations Research Analyst",
                "Forecasting Analyst", "Experimentation (A/B) Analyst",
                "GeoSpatial (GIS) Analyst",
            ]),
            ("Analytics & Insights", [
                "Data Analyst", "Analytics Manager",
                "Product Analytics Manager", "Customer Insights Analyst",
                "Market Research Analyst", "Research Analyst",
                "Reporting Analyst", "Web Analytics Specialist",
                "Pricing Analyst", "Fraud Analyst",
                "Risk Analytics Analyst",
            ]),
        ],

        # Security (1 → 1 cluster)
        "Security": [
            ("Security", [
                "Security Analyst",
            ]),
        ],

        # Supply Chain & Logistics (24 → 3 clusters)
        "Supply Chain & Logistics": [
            ("Demand & Supply Planning", [
                "Demand Planner", "Master Scheduler",
                "Capacity Planner (Logistics)", "Route Planner",
                "Transportation Planner", "Network Planning Analyst (Air/Rail)",
                "Order Management Specialist",
            ]),
            ("Procurement & Supplier Management", [
                "Strategic Sourcing Manager", "Commodity Manager",
                "Supplier Performance Analyst", "Supplier Quality Engineer",
            ]),
            ("Logistics & Trade Compliance", [
                "Warehouse Manager", "Warehouse Operations Analyst",
                "Fleet Coordinator", "Fleet Manager",
                "Dispatcher (Office)", "Freight Analyst",
                "Intermodal Operations Analyst",
                "Last-Mile Operations Coordinator",
                "Returns/Reverse Logistics Analyst",
                "Customs Compliance Analyst",
                "Customs Compliance Specialist",
                "Trade Compliance Analyst", "Trade Compliance Manager",
            ]),
        ],

        # Manufacturing (5 → 1 cluster)
        "Manufacturing": [
            ("Manufacturing", [
                "EHS Specialist", "Field Operations Coordinator",
                "Maintenance Planner", "Maintenance Planner/Scheduler",
                "Manufacturing Analyst",
            ]),
        ],

        # Healthcare (17 → 2 clusters)
        "Healthcare": [
            ("Clinical Operations & Patient Care", [
                "Case Management Coordinator",
                "Clinical Documentation Improvement (CDI) Specialist",
                "Clinical Operations Coordinator",
                "Credentialing Specialist",
                "Patient Access Supervisor",
                "Patient Services Coordinator",
                "Practice Manager",
                "Provider Enrollment Specialist",
                "Quality & Patient Safety Coordinator (Admin)",
                "Utilization Review Coordinator",
            ]),
            ("Health Information & Revenue", [
                "Health Information Manager", "HIM Data Analyst",
                "Medical Billing Specialist", "Medical Coder",
                "Medical Office Administrator",
                "Healthcare Compliance Specialist",
                "Revenue Cycle Analyst",
            ]),
        ],

        # Government & Public Sector (16 → 2 clusters)
        "Government & Public Sector": [
            ("Policy, Regulation & Legislation", [
                "Policy Analyst", "Regulatory Policy Analyst",
                "Legislative Aide", "City Planner",
                "Transportation Planning Analyst",
                "Emergency Management Planner",
                "Public Health Program Analyst",
                "Public Information Officer",
            ]),
            ("Public Administration & Compliance", [
                "Budget Analyst (Public)", "Clerk of Court (Admin)",
                "Elections Coordinator", "Grant Writer",
                "Grants Compliance Manager", "Procurement Officer (Public)",
                "Public Records Coordinator", "Records Officer",
            ]),
        ],

        # Nonprofit & NGO (19 → 2 clusters)
        "Nonprofit & NGO": [
            ("Fundraising & Donor Engagement", [
                "Fundraising Manager", "Major Gifts Officer",
                "Planned Giving Officer", "Donor Relations Manager",
                "Development Coordinator", "Grant Accountant",
                "Grants Manager", "Membership Manager",
            ]),
            ("Programs & Community Impact", [
                "Program Coordinator", "Program Evaluator",
                "Program Manager (Nonprofit)", "NGO Operations Manager",
                "Community Development Coordinator",
                "Community Outreach Manager",
                "Advocacy Campaigns Manager", "Coalition Coordinator",
                "Economic Development Analyst",
                "Monitoring & Evaluation Specialist",
                "Volunteer Coordinator",
            ]),
        ],

        # Real Estate & Construction (19 → 2 clusters)
        "Real Estate & Construction": [
            ("Real Estate Transactions & Management", [
                "Acquisitions Analyst", "Asset Manager (Real Estate)",
                "Development Coordinator (Real Estate)",
                "Escrow Officer (Office)", "Facilities Manager",
                "Leasing Consultant", "Property Manager",
                "Real Estate Agent", "Real Estate Financial Analyst",
                "Title Officer (Office)", "Transaction Coordinator",
            ]),
            ("Construction Project Delivery", [
                "Construction Project Manager",
                "Document Control Coordinator (Construction)",
                "Estimator", "Owner's Rep (Office)",
                "Preconstruction Manager",
                "Safety Compliance Coordinator (Office)",
                "Scheduler (Construction)", "Site Administrator",
            ]),
        ],

        # Retail & E-commerce (14 → 2 clusters)
        "Retail & E-commerce": [
            ("Merchandising & Category Management", [
                "Assortment Planner", "Category Manager",
                "Merchandising Analyst", "Space Planner",
                "Visual Merchandising Coordinator (Office)",
                "Retail Finance Analyst", "Loss Prevention Analyst (Office)",
            ]),
            ("E-commerce & Retail Operations", [
                "E-commerce Manager", "Marketplace Operations Manager",
                "Omnichannel Operations Manager", "Store Operations Manager",
                "CRM & Loyalty Program Manager",
                "Returns & Refunds Analyst",
                "Vendor Operations Manager (Retail)",
            ]),
        ],

        # Insurance (11 → 2 clusters)
        "Insurance": [
            ("Underwriting & Risk Assessment", [
                "Actuarial Analyst", "Catastrophe Modeling Analyst",
                "Exposure Management Analyst", "Premium Audit Analyst",
                "Reinsurance Analyst", "Underwriter",
            ]),
            ("Claims & Policy Operations", [
                "Broker Account Manager (Office)",
                "Claims Adjuster (Office)",
                "Claims Operations Manager (Office)",
                "Compliance Analyst (Insurance)",
                "Policy Administration Specialist",
            ]),
        ],

        # Banking & Financial Services (17 → 2 clusters)
        "Banking & Financial Services": [
            ("Investment & Research", [
                "Buy-Side Analyst", "Equity Research Associate",
                "Investment Analyst", "Portfolio Analyst",
                "Sell-Side Analyst", "Trading Operations Analyst",
                "Wealth Management Associate",
                "Wealth Operations Coordinator",
            ]),
            ("Risk, Compliance & Banking Operations", [
                "AML Investigator", "Compliance Analyst",
                "Financial Crime Analyst", "Loan Officer",
                "Payments Operations Analyst", "Relationship Manager",
                "Risk Analyst", "Sanctions Analyst",
                "Trust Officer (Office)",
            ]),
        ],

        # Hospitality & Travel (12 → 2 clusters)
        "Hospitality & Travel": [
            ("Hotel & Event Sales", [
                "Catering Sales Manager", "Event Sales Coordinator",
                "Group Sales Coordinator", "Hotel Sales Manager",
                "Banquet Administration Coordinator",
                "Guest Relations Manager (Office)",
            ]),
            ("Travel Operations & Revenue", [
                "Airline Scheduling Analyst (Office)",
                "Crew Scheduler (Office)", "Revenue Analyst",
                "Tour Operations Coordinator", "Travel Product Manager",
                "Yield/Revenue Management Analyst",
            ]),
        ],

        # Energy & Utilities (11 → 2 clusters)
        "Energy & Utilities": [
            ("Energy Trading & Market Operations", [
                "Energy Analyst", "Gas Nominations/Scheduling Analyst",
                "Power Trading Operations Analyst (Office)",
                "Rates Analyst", "Settlements Analyst (Energy)",
            ]),
            ("Utility Infrastructure & Compliance", [
                "Asset Management Analyst",
                "Environmental Compliance Specialist",
                "Outage Planning Coordinator (Office)",
                "Regulatory Affairs Manager (Office)",
                "Scheduling Coordinator",
                "Transmission Planning Analyst (Office)",
            ]),
        ],

        # Pharma & Life Sciences (12 → 2 clusters)
        "Pharma & Life Sciences": [
            ("Clinical & Regulatory Affairs", [
                "Clinical Study Start-Up Specialist",
                "Clinical Trial Coordinator",
                "Medical Affairs Coordinator (Admin)",
                "Medical Writer", "Pharmacovigilance Specialist",
                "Regulatory Affairs Associate",
                "Regulatory Submissions Coordinator",
                "Safety Case Processing Specialist",
            ]),
            ("Quality & Documentation", [
                "Biostatistics Operations Coordinator (Admin)",
                "CMC Documentation Specialist (Office)",
                "Quality Assurance Specialist",
                "Quality Systems Coordinator (GxP)",
            ]),
        ],

        # Research Administration (11 → 2 clusters)
        "Research Administration": [
            ("Grant & Research Finance", [
                "Grant Administrator", "Grant Compliance Officer",
                "Research Finance/Grant Accountant",
                "Sponsored Programs Specialist",
                "Research Administrator",
            ]),
            ("Research Operations & Compliance", [
                "Core Facilities Coordinator (Admin)",
                "IRB/IACUC Coordinator (Admin)",
                "Lab Operations Coordinator (Admin)",
                "Publication Manager (Admin)",
                "Research Coordinator", "Technical Editor",
            ]),
        ],

        # Agency & Advertising (11 → 2 clusters)
        "Agency & Advertising": [
            ("Account Strategy & Media", [
                "Account Manager (Agency)", "Account Planner",
                "Ad Operations Manager", "Media Operations Manager",
                "Media Strategist", "Strategist (Agency)",
            ]),
            ("Creative & Production Operations", [
                "Creative Operations Manager",
                "Post-Production Coordinator (Office)",
                "Production Coordinator (Agency)",
                "Rights & Licensing Coordinator",
                "Traffic Coordinator (Agency)",
            ]),
        ],

        # Agriculture (10 → 2 clusters)
        "Agriculture": [
            ("Agribusiness & Commodity Operations", [
                "Agribusiness Analyst",
                "Commodity Trading Operations Analyst",
                "Co-op Operations Coordinator",
                "Farm Records Manager (Office)",
                "Seed/Inputs Product Manager (Office)",
            ]),
            ("Agricultural Compliance & Sustainability", [
                "Food Safety Compliance Coordinator (Office)",
                "Land Management Coordinator",
                "Regulatory Affairs Coordinator (Ag)",
                "Supply Planner (Ag)",
                "Sustainability Reporting Analyst (Ag)",
            ]),
        ],

        # Architecture, Engineering & Construction (AEC) (10 → 2 clusters)
        "Architecture, Engineering & Construction (AEC)": [
            ("Design & Engineering (AEC)", [
                "CAD/BIM Manager", "Civil Engineer (Office)",
                "MEP Engineer (Office)", "Project Architect (Office)",
                "Structural Engineer (Office)", "Specification Writer",
            ]),
            ("Project Controls & Estimation (AEC)", [
                "Cost Estimator (AEC)", "Design Project Manager (AEC)",
                "Permitting Coordinator (AEC)",
                "Project Controls Manager (AEC)",
            ]),
        ],

        # Automotive (10 → 2 clusters)
        "Automotive": [
            ("Automotive Product & Program Management", [
                "Engineering Change Coordinator (Auto)",
                "Homologation/Regulatory Specialist (Auto)",
                "Manufacturing Program Manager (Auto)",
                "Product Planning Analyst (Auto)",
                "Quality Systems Engineer (Office)",
            ]),
            ("Automotive Sales & Service Operations", [
                "After-Sales Operations Analyst",
                "Dealer Network Development Analyst",
                "Service Technical Writer (Auto)",
                "Telematics Data Analyst (Auto)",
                "Warranty Claims Analyst",
            ]),
        ],

        # Consumer Packaged Goods (CPG) (10 → 2 clusters)
        "Consumer Packaged Goods (CPG)": [
            ("CPG Product & Innovation", [
                "Innovation Project Manager (CPG)",
                "Packaging Engineer (Office)",
                "R&D Program Coordinator (Office)",
                "Regulatory Labeling Specialist",
                "Quality Systems Coordinator (CPG)",
            ]),
            ("CPG Commercial & Trade", [
                "Brand Finance Analyst (CPG)",
                "Category Insights Manager (CPG)",
                "Field Marketing Operations Coordinator (CPG)",
                "Sales Finance Analyst (CPG)",
                "Trade Promotion Analyst",
            ]),
        ],

        # Media & Entertainment (9 → 1 cluster)
        "Media & Entertainment": [
            ("Media & Entertainment", [
                "Content Operations Manager",
                "Distribution Operations Analyst",
                "Music Licensing Coordinator",
                "Post Scheduling Coordinator (Office)",
                "Production Finance Analyst",
                "Programming Scheduler (Office)",
                "Rights Management Analyst",
                "Script Coordinator (Office)",
                "Talent Relations Coordinator (Office)",
            ]),
        ],

        # Mining (10 → 2 clusters)
        "Mining": [
            ("Mine Planning & Operations", [
                "Exploration Program Coordinator (Office)",
                "Geospatial Data Analyst (Mining)",
                "Mine Planning Analyst (Office)",
                "Commodity Logistics Coordinator (Mining)",
                "Royalty/Lease Analyst (Mining)",
            ]),
            ("Mining Compliance & Stakeholder Relations", [
                "Environmental Analyst (Mining)",
                "Permitting Specialist (Mining)",
                "Reclamation Program Coordinator (Office)",
                "Safety & Compliance Analyst (Mining)",
                "Stakeholder Relations Coordinator (Mining)",
            ]),
        ],

        # Sports & Entertainment (10 → 2 clusters)
        "Sports & Entertainment": [
            ("Sports Revenue & Sponsorship", [
                "Merchandise Planning Analyst (Sports)",
                "Season Ticket Operations Manager",
                "Sponsorship Coordinator",
                "Ticketing Analyst",
                "Event Finance Analyst (Sports)",
            ]),
            ("Sports Operations & Community", [
                "Athletics Compliance Coordinator (Office)",
                "Community Relations Coordinator (Sports)",
                "Game Day Operations Planner (Office)",
                "Team Operations Coordinator (Office)",
                "Venue Scheduling Coordinator",
            ]),
        ],

        # Telecom (10 → 2 clusters)
        "Telecom": [
            ("Telecom Network & Capacity", [
                "Capacity Planning Analyst (Telecom)",
                "Network Planning Analyst (Telecom)",
                "OSS/BSS Analyst",
                "Spectrum Manager (Office)",
                "Service Assurance Coordinator (Office)",
            ]),
            ("Telecom Products & Operations", [
                "Customer Provisioning Specialist (Office)",
                "Field Operations Coordinator (Telecom)",
                "Partner Operations Manager (Telecom)",
                "Regulatory Compliance Analyst (Telecom)",
                "Telecom Product Manager",
            ]),
        ],

        # Proposal & Bid Management (2 → 1 cluster)
        "Proposal & Bid Management": [
            ("Proposal & Bid Management", [
                "Proposal/Bid Manager", "RFP Manager",
            ]),
        ],

        # Frontline Management (1 → 1 cluster)
        "Frontline Management": [
            ("Frontline Management", [
                "Small Business Owner",
            ]),
        ],
    }

    # Fix the Communications cluster that has an inline conditional
    # (the code above uses a ternary for readability but let's just
    # define it correctly in the dict literal above — it's already correct)

    # Build cluster list
    clusters: list[dict] = []
    all_clustered_roles: set[str] = set()
    all_taxonomy_roles: set[str] = {r["role"] for r in roles}

    for category_name, category_roles in categories.items():
        if category_name not in _CLUSTER_DEFS:
            raise ValueError(
                f"No cluster definition for category: {category_name}"
            )

        defs = _CLUSTER_DEFS[category_name]
        category_role_set = set(category_roles)
        cluster_role_set: set[str] = set()

        for cluster_label, cluster_roles in defs:
            if not cluster_roles:
                raise ValueError(
                    f"Empty cluster '{cluster_label}' in category "
                    f"'{category_name}'"
                )

            # Validate all cluster roles exist in this category
            for role in cluster_roles:
                if role not in category_role_set:
                    raise ValueError(
                        f"Cluster '{cluster_label}' references role "
                        f"'{role}' not found in category '{category_name}'"
                    )
                if role in cluster_role_set:
                    raise ValueError(
                        f"Role '{role}' appears in multiple clusters "
                        f"within '{category_name}'"
                    )
                cluster_role_set.add(role)

            clusters.append({
                "cluster_label": cluster_label,
                "category": category_name,
                "roles": list(cluster_roles),
            })
            all_clustered_roles.update(cluster_roles)

        # Verify every role in this category is covered
        missing = category_role_set - cluster_role_set
        if missing:
            raise ValueError(
                f"Roles in '{category_name}' not assigned to any cluster: "
                f"{sorted(missing)}"
            )

    # Final global validation
    unclustered = all_taxonomy_roles - all_clustered_roles
    if unclustered:
        raise ValueError(
            f"Roles not assigned to any cluster: {sorted(unclustered)}"
        )

    extra = all_clustered_roles - all_taxonomy_roles
    if extra:
        raise ValueError(
            f"Cluster roles not in taxonomy: {sorted(extra)}"
        )

    return clusters
