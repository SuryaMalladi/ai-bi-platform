import streamlit as st
import pandas as pd
import io
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the default Streamlit menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALISATION
# All variables the app needs to remember across screens
# ============================================================

def init_session_state():
    defaults = {
        "screen": "api_key",
        "api_key": None,
        "api_validated": False,
        "privacy_accepted": False,
        "uploaded_df": None,
        "uploaded_filename": None,
        "quality_report": None,
        "proceed_to_role": False,
        "detected_industry": None,
        "confirmed_industry": None,
        "user_role": None,
        "ready_to_analyse": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================
# SCREEN ROUTER
# Decides which screen to show based on session state
# ============================================================

def get_current_screen():
    if not st.session_state.api_validated:
        return "api_key"
    if not st.session_state.privacy_accepted:
        return "privacy"
    if st.session_state.uploaded_df is None:
        return "data_source"
    if not st.session_state.proceed_to_role:
        return "data_quality"
    if not st.session_state.ready_to_analyse:
        return "role_input"
    return "dashboard_placeholder"


# ============================================================
# PROGRESS BAR
# Shows the user where they are in the journey
# ============================================================

def show_progress(current_step):
    steps = [
        "API Setup",
        "Privacy",
        "Data Upload",
        "Quality Check",
        "Role & Industry",
        "Dashboard"
    ]
    step_map = {
        "api_key": 1,
        "privacy": 2,
        "data_source": 3,
        "data_quality": 4,
        "role_input": 5,
        "dashboard_placeholder": 6
    }
    current = step_map.get(current_step, 1)
    total = len(steps)
    progress_value = current / total

    st.progress(progress_value)
    cols = st.columns(total)
    for i, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.caption(f"✅ {step}")
            elif i == current:
                st.caption(f"**🔵 {step}**")
            else:
                st.caption(f"⬜ {step}")
    st.divider()


# ============================================================
# SCREEN 1 — API KEY ENTRY
# ============================================================

def screen_api_key():
    st.markdown("## 🔐 Secure Configuration")
    st.markdown("Enter your OpenAI API key to begin. "
                "This key is used only for this session and is never stored anywhere.")

    st.info("🔒 Your key is stored only in memory. "
            "It is automatically cleared when you close this tab.")

    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )
        submitted = st.form_submit_button(
            "Validate and Continue →",
            use_container_width=True
        )

    if submitted:
        if not api_key_input:
            st.error("Please enter your API key.")
        elif not api_key_input.startswith("sk-"):
            st.error("This does not look like a valid OpenAI key. "
                     "It should start with sk-")
        else:
            with st.spinner("Validating your API key..."):
                import openai
                try:
                    client = openai.OpenAI(api_key=api_key_input)
                    client.models.list()
                    st.session_state.api_key = api_key_input
                    st.session_state.api_validated = True
                    st.success("✅ API key validated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error("This API key does not appear to be valid. "
                             "Please check and try again.")


# ============================================================
# SCREEN 2 — DATA PRIVACY NOTICE
# ============================================================

def screen_privacy():
    show_progress("privacy")
    st.markdown("## 📋 Data Privacy Notice")

    st.warning("""
**Before you upload any data, please read and confirm the following:**

- Your data is processed in this session only
- Nothing is stored permanently on any server
- Analysis is powered by OpenAI — please ensure your data complies
  with your organisation's data sharing and privacy policy before uploading
- All data is cleared automatically when you close this tab
- OpenAI is named as a data subprocessor — your data is sent to OpenAI
  via their secure API for analysis only
""")

    st.markdown("---")

    if st.button("✅ I Understand — Continue to Data Upload",
                 use_container_width=True,
                 type="primary"):
        st.session_state.privacy_accepted = True
        st.rerun()


# ============================================================
# SCREEN 3 — DATA SOURCE SELECTION
# ============================================================

def screen_data_source():
    show_progress("data_source")
    st.markdown("## 📂 Upload Your Dataset")
    st.markdown("Upload a CSV or Excel file to begin your analysis.")

    st.info("📌 Database connections (MySQL, PostgreSQL, SQL Server) "
            "will be available in Phase 4.")

    uploaded_file = st.file_uploader(
        "Choose your file",
        type=["csv", "xlsx", "xls"],
        help="Maximum file size: 50MB"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_df = df
            st.session_state.uploaded_filename = uploaded_file.name

            st.success(f"✅ File uploaded successfully: "
                       f"**{uploaded_file.name}** "
                       f"({len(df):,} rows, {len(df.columns)} columns)")

            st.markdown("**Preview — first 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Continue to Data Quality Check →",
                         use_container_width=True,
                         type="primary"):
                st.rerun()

        except Exception as e:
            st.error(f"Could not read this file. "
                     f"Please check it is a valid CSV or Excel file.")


# ============================================================
# DATA QUALITY FUNCTIONS
# Used by Screen 4
# ============================================================

def detect_pii(df):
    """
    Scans every cell in the dataframe for common PII patterns.
    Returns a list of columns where PII was found.
    """
    pii_found = []

    email_pattern = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    phone_pattern = re.compile(
        r'\b(\+?\d[\d\s\-().]{7,}\d)\b'
    )
    card_pattern = re.compile(
        r'\b(?:\d[ -]?){13,16}\b'
    )
    nid_pattern = re.compile(
        r'\b\d{3}-\d{2}-\d{4}\b'
    )

    for col in df.columns:
        col_str = df[col].astype(str)
        for pattern, label in [
            (email_pattern, "email addresses"),
            (phone_pattern, "phone numbers"),
            (card_pattern, "possible card numbers"),
            (nid_pattern, "national ID numbers"),
        ]:
            if col_str.str.contains(pattern, regex=True).any():
                pii_found.append((col, label))
                break

    return pii_found


def calculate_quality_score(df, missing_count, duplicate_count, pii_found):
    """
    Calculates an overall data quality score out of 100.
    Deducts points for missing values, duplicates, and PII.
    """
    score = 100
    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:
        missing_pct = (missing_count / total_cells) * 100
        score -= min(missing_pct * 2, 30)

    if duplicate_count > 0:
        dup_pct = (duplicate_count / len(df)) * 100
        score -= min(dup_pct * 2, 20)

    if pii_found:
        score -= 15

    return max(0, round(score))


def run_quality_report(df):
    """
    Runs all quality checks on the dataframe and returns
    a dictionary of results.
    """
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())
    duplicate_count = int(df.duplicated().sum())
    pii_found = detect_pii(df)
    quality_score = calculate_quality_score(
        df, total_missing, duplicate_count, pii_found
    )

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    date_cols = [
        col for col in df.columns
        if 'date' in col.lower() or 'time' in col.lower()
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing": total_missing,
        "missing_per_col": missing_per_col[missing_per_col > 0].to_dict(),
        "duplicate_count": duplicate_count,
        "pii_found": pii_found,
        "quality_score": quality_score,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
    }


# ============================================================
# SCREEN 4 — DATA QUALITY REPORT
# ============================================================

def screen_data_quality():
    show_progress("data_quality")
    df = st.session_state.uploaded_df
    filename = st.session_state.uploaded_filename

    st.markdown("## 📋 Data Quality Report")
    st.markdown(f"**Dataset:** {filename}")

    if st.session_state.quality_report is None:
        with st.spinner("Scanning your dataset..."):
            st.session_state.quality_report = run_quality_report(df)

    report = st.session_state.quality_report
    score = report["quality_score"]

    # Quality score display
    if score >= 80:
        score_color = "🟢"
        score_label = "Good"
    elif score >= 60:
        score_color = "🟡"
        score_label = "Fair — some issues to note"
    else:
        score_color = "🔴"
        score_label = "Poor — review recommended"

    st.markdown(
        f"### Overall Quality Score: "
        f"{score_color} **{score}/100** — {score_label}"
    )
    st.progress(score / 100)
    st.markdown("---")

    # Basic counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{report['rows']:,}")
    with col2:
        st.metric("Total Columns", report['columns'])
    with col3:
        st.metric("Missing Values", report['total_missing'])
    with col4:
        st.metric("Duplicate Rows", report['duplicate_count'])

    st.markdown("---")

    # Column type breakdown
    st.markdown("**Column Breakdown:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📊 **{len(report['numeric_cols'])}** Numeric columns\n\n"
                + ", ".join(report['numeric_cols'][:5])
                + ("..." if len(report['numeric_cols']) > 5 else ""))
    with col2:
        st.info(f"🏷️ **{len(report['categorical_cols'])}** Text/Category columns\n\n"
                + ", ".join(report['categorical_cols'][:5])
                + ("..." if len(report['categorical_cols']) > 5 else ""))
    with col3:
        if report['date_cols']:
            st.info(f"📅 **{len(report['date_cols'])}** Date columns\n\n"
                    + ", ".join(report['date_cols']))
        else:
            st.info("📅 **0** Date columns detected")

    # Missing values detail
    if report['missing_per_col']:
        st.markdown("---")
        st.warning("⚠️ **Missing Values Detected:**")
        for col, count in report['missing_per_col'].items():
            pct = round((count / report['rows']) * 100, 1)
            st.markdown(f"- **{col}**: {count} missing values ({pct}% of rows)")

    # Duplicates
    if report['duplicate_count'] > 0:
        st.markdown("---")
        st.warning(
            f"⚠️ **{report['duplicate_count']} duplicate rows detected.** "
            "These will not affect your analysis — the AI will work "
            "with the full dataset as uploaded."
        )

    # PII warning
    if report['pii_found']:
        st.markdown("---")
        st.error("🔴 **Data Protection Notice — PII Detected**")
        st.markdown(
            "This dataset appears to contain personal information in "
            "the following columns:"
        )
        for col, label in report['pii_found']:
            st.markdown(f"- **{col}**: {label} detected")
        st.markdown(
            "> Please ensure you have the legal right to process this data "
            "and that it complies with your organisation's data protection "
            "policy before proceeding."
        )

    # Recommendations
    st.markdown("---")
    st.markdown("**Recommended Actions:**")
    if report['total_missing'] == 0 and report['duplicate_count'] == 0:
        st.success("✅ No data quality issues detected. "
                   "Your dataset is ready for analysis.")
    else:
        if report['total_missing'] > 0:
            st.markdown(
                "- Missing values are present. The AI will note "
                "where data gaps may affect confidence."
            )
        if report['duplicate_count'] > 0:
            st.markdown(
                "- Duplicate rows detected. Consider removing them "
                "for cleaner analysis, though the AI will still "
                "produce meaningful results."
            )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Upload a Different File",
                     use_container_width=True):
            st.session_state.uploaded_df = None
            st.session_state.uploaded_filename = None
            st.session_state.quality_report = None
            st.rerun()
    with col2:
        proceed_label = (
            "⚠️ Proceed with Analysis (PII present — confirm above)"
            if report['pii_found']
            else "Proceed to Role & Industry Setup →"
        )
        if st.button(proceed_label,
                     use_container_width=True,
                     type="primary"):
            st.session_state.proceed_to_role = True
            st.rerun()


# ============================================================
# INDUSTRY DETECTION — KEYWORD ENGINE
# Zero API cost — runs entirely in Python
# ============================================================

def detect_industry_from_columns(df):
    """
    Scans column names for business domain keywords.
    Returns the most likely industry as a string.
    No API call needed — pure Python keyword matching.
    """
    columns_lower = [col.lower() for col in df.columns]
    col_string = " ".join(columns_lower)

    domain_keywords = {
        "Retail & Sales": [
            "sales", "revenue", "store", "product", "category",
            "units sold", "returns", "customer", "basket",
            "transaction", "sku", "margin", "target", "region"
        ],
        "HR & People": [
            "employee", "attrition", "salary", "department",
            "performance", "tenure", "satisfaction", "headcount",
            "absence", "gender", "age", "hire", "leave",
            "engagement", "band", "role"
        ],
        "Finance & Budget": [
            "budget", "actual", "variance", "forecast", "cost",
            "expenditure", "spend", "p&l", "ebitda", "revenue",
            "quarter", "ytd", "centre", "center", "approved",
            "invoice", "payment", "balance"
        ],
        "Operations & Manufacturing": [
            "units produced", "defects", "downtime", "efficiency",
            "cycle time", "shift", "line", "throughput", "yield",
            "operator", "machine", "output", "target output", "oee"
        ],
        "Marketing": [
            "campaign", "impressions", "clicks", "ctr", "cpl",
            "cac", "roas", "conversion", "channel", "leads",
            "funnel", "roi", "email open", "bounce"
        ],
    }

    scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(
            1 for keyword in keywords
            if keyword in col_string
        )
        scores[domain] = score

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    # Find any secondary domains with meaningful scores
    secondary = [
        domain for domain, score in scores.items()
        if score > 0 and domain != best_domain
    ]

    if best_score == 0:
        return "General Business", []

    return best_domain, secondary


# ============================================================
# SCREEN 5 — ROLE INPUT AND INDUSTRY CONFIRMATION
# ============================================================

def screen_role_input():
    show_progress("role_input")
    df = st.session_state.uploaded_df

    st.markdown("## 🏭 Industry & Role Setup")
    st.markdown(
        "We have scanned your dataset. Confirm the industry below "
        "and tell us your role — then we will generate your dashboard."
    )

    # --- INDUSTRY DETECTION ---
    if st.session_state.detected_industry is None:
        detected, secondary = detect_industry_from_columns(df)
        st.session_state.detected_industry = detected
        st.session_state.detected_secondary = secondary

    detected = st.session_state.detected_industry
    secondary = st.session_state.get("detected_secondary", [])

    st.markdown("---")
    st.markdown("### Step 1 — Confirm Your Industry")

    if secondary:
        detection_message = (
            f"We detected this dataset is primarily **{detected}** "
            f"with elements of: **{', '.join(secondary)}**."
        )
    else:
        detection_message = (
            f"We detected this dataset is: **{detected}**."
        )

    st.info(f"🔍 {detection_message}\n\nIs this correct?")

    industry_choice = st.radio(
        "Select an option:",
        options=[
            f"✅ Yes — {detected} is correct",
            "✏️ No — I want to describe my industry myself"
        ],
        index=0,
        label_visibility="collapsed"
    )

    confirmed_industry = detected

    if "No" in industry_choice:
        custom_industry = st.text_input(
            "Describe your industry or business domain:",
            placeholder="e.g. E-commerce retail, NHS healthcare, "
                        "logistics and supply chain..."
        )
        if custom_industry:
            confirmed_industry = custom_industry
            st.success(f"✅ Industry set to: **{custom_industry}**")
    else:
        st.success(f"✅ Industry confirmed: **{detected}**")

    # --- ROLE INPUT ---
    st.markdown("---")
    st.markdown("### Step 2 — Enter Your Role")
    st.markdown(
        "Type your exact role or job title. "
        "The AI will interpret your seniority, function, "
        "and what insights matter most to you."
    )

    role_examples = (
        "e.g. CFO, Regional Sales Manager North, "
        "Store Manager, Head of HR, Data Scientist, "
        "Operations Supervisor, Board Member..."
    )

    user_role = st.text_input(
        "Your role or job title:",
        placeholder=role_examples,
        value=st.session_state.user_role or ""
    )

    # Prompt injection check
    injection_keywords = [
        "ignore", "reveal", "system prompt", "you are now",
        "forget", "override", "jailbreak", "disregard",
        "pretend", "act as", "drop table", "select *",
        "<script", "javascript:"
    ]

    role_is_safe = True
    if user_role:
        role_lower = user_role.lower()
        if any(kw in role_lower for kw in injection_keywords):
            st.error(
                "⚠️ This input cannot be processed. "
                "Please enter a valid role description."
            )
            role_is_safe = False

    # --- DATASET SUMMARY FOR AI ---
    st.markdown("---")
    st.markdown("### What will be sent to the AI")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", st.session_state.uploaded_filename)
    with col2:
        st.metric("Industry", confirmed_industry)
    with col3:
        st.metric("Your Role", user_role if user_role else "Not entered yet")

    st.markdown("---")

    # --- GENERATE DASHBOARD BUTTON ---
    button_ready = (
        user_role
        and role_is_safe
        and confirmed_industry
    )

    if st.button(
        "🚀 Generate My Dashboard →",
        use_container_width=True,
        type="primary",
        disabled=not button_ready
    ):
        if not user_role:
            st.error("Please enter your role before continuing.")
        else:
            st.session_state.confirmed_industry = confirmed_industry
            st.session_state.user_role = user_role
            st.session_state.ready_to_analyse = True
            st.rerun()

    if not button_ready and user_role and role_is_safe:
        st.caption("Please confirm your industry above before generating.")


# ============================================================
# SCREEN 6 — DASHBOARD PLACEHOLDER
# Full dashboard built in Step 6
# ============================================================

def screen_dashboard_placeholder():
    show_progress("dashboard_placeholder")

    st.markdown("## 📊 Your Dashboard")

    st.success(
        f"✅ Everything is ready. Generating dashboard for: "
        f"**{st.session_state.user_role}** "
        f"| Industry: **{st.session_state.confirmed_industry}**"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Role", st.session_state.user_role)
    with col2:
        st.metric("Industry", st.session_state.confirmed_industry)
    with col3:
        st.metric(
            "Dataset",
            f"{len(st.session_state.uploaded_df):,} rows"
        )

    st.info(
        "🔧 **Dashboard engine coming in Step 6.**\n\n"
        "In the next step, this screen will be replaced with your "
        "full AI-generated dashboard — executive summary, traffic lights, "
        "charts, anomaly alerts, recommendations, and narrative report."
    )

    st.markdown("---")
    st.markdown("**What will appear here in Step 6:**")

    features = [
        "🎯 Executive Summary Card — 3 sentences tailored to your role",
        "🚦 Traffic Light System — key metrics with Green / Amber / Red status",
        "⚠️ Anomaly Alerts Panel — unusual findings flagged automatically",
        "📊 AI-Generated Charts — selected dynamically for your role",
        "💡 Business Insight Captions — one per chart, role-specific language",
        "📋 Recommendations — specific actions for your role",
        "📝 Narrative Report — professional business document, copy-paste ready",
    ]

    for feature in features:
        st.markdown(f"- {feature}")

    st.markdown("---")

    if st.button("← Change Role or Industry",
                 use_container_width=False):
        st.session_state.ready_to_analyse = False
        st.session_state.confirmed_industry = None
        st.session_state.user_role = None
        st.rerun()


# ============================================================
# MAIN APP ROUTER
# Reads current screen and calls the right function
# ============================================================

def main():
    current_screen = get_current_screen()

    if current_screen == "api_key":
        screen_api_key()
    elif current_screen == "privacy":
        screen_privacy()
    elif current_screen == "data_source":
        screen_data_source()
    elif current_screen == "data_quality":
        screen_data_quality()
    elif current_screen == "role_input":
        screen_role_input()
    elif current_screen == "dashboard_placeholder":
        screen_dashboard_placeholder()


if __name__ == "__main__":
    main()