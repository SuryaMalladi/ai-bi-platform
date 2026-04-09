# app.py — AI Intelligence Platform — Complete with NL Chat Interface

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import openai
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "api_key": None,
        "api_validated": False,
        "privacy_accepted": False,
        "uploaded_df": None,
        "uploaded_filename": None,
        "comparison_df": None,
        "comparison_filename": None,
        "comparison_mode": False,
        "quality_report": None,
        "proceed_to_role": False,
        "detected_industry": None,
        "detected_secondary": [],
        "confirmed_industry": None,
        "user_role": None,
        "ready_to_analyse": False,
        "dashboard_result": None,
        "analysis_error": None,
        "precomputed_stats": None,
        "role_profile": None,
        "show_narrative": False,
        "analysis_count": 0,
        "chat_history": [],
        "suggested_questions": [],
        "chat_input_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================
# SCREEN ROUTER
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
    return "dashboard"


# ============================================================
# PROGRESS BAR
# ============================================================

def show_progress(current_step):
    steps = [
        "API Setup", "Privacy", "Data Upload",
        "Quality Check", "Role & Industry", "Dashboard"
    ]
    step_map = {
        "api_key": 1, "privacy": 2, "data_source": 3,
        "data_quality": 4, "role_input": 5, "dashboard": 6
    }
    current = step_map.get(current_step, 1)
    total = len(steps)
    st.progress(current / total)
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
# SCREEN 1 — API KEY
# ============================================================

def screen_api_key():
    st.markdown("## 🔐 Secure Configuration")
    st.markdown(
        "Enter your OpenAI API key to begin. "
        "This key is used only for this session and is never stored anywhere."
    )
    st.info(
        "🔒 Your key is stored only in memory. "
        "Cleared automatically when you close this tab."
    )
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
            st.error("This does not look like a valid OpenAI key.")
        else:
            with st.spinner("Validating..."):
                try:
                    client = openai.OpenAI(api_key=api_key_input)
                    client.models.list()
                    st.session_state.api_key = api_key_input
                    st.session_state.api_validated = True
                    st.success("✅ API key validated.")
                    st.rerun()
                except Exception:
                    st.error("Invalid API key. Please check and try again.")


# ============================================================
# SCREEN 2 — PRIVACY
# ============================================================

def screen_privacy():
    show_progress("privacy")
    st.markdown("## 📋 Data Privacy Notice")
    st.warning("""
**Before you upload any data, please confirm:**

- Your data is processed in this session only
- Nothing is stored permanently on any server
- Analysis is powered by OpenAI — ensure your data complies with your
  organisation's data sharing policy before uploading
- All data is cleared automatically when you close this tab
- OpenAI is named as a data subprocessor
""")
    st.markdown("---")
    if st.button(
        "✅ I Understand — Continue",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.privacy_accepted = True
        st.rerun()


# ============================================================
# SCREEN 3 — DATA SOURCE
# ============================================================

def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        st.error(
            "Could not read this file. "
            "Please check it is a valid CSV or Excel file."
        )
        return None


def screen_data_source():
    show_progress("data_source")
    st.markdown("## 📂 Upload Your Dataset")

    comparison_toggle = st.toggle(
        "Enable Comparison Mode — upload two datasets to compare"
    )
    st.session_state.comparison_mode = comparison_toggle
    st.markdown("---")

    if not comparison_toggle:
        uploaded_file = st.file_uploader(
            "Choose your file",
            type=["csv", "xlsx", "xls"],
            key="main_upload"
        )
        if uploaded_file:
            df = load_file(uploaded_file)
            if df is not None:
                st.session_state.uploaded_df = df
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(
                    f"✅ {uploaded_file.name} — "
                    f"{len(df):,} rows, {len(df.columns)} columns"
                )
                st.dataframe(df.head(), use_container_width=True)
                if st.button(
                    "Continue to Data Quality Check →",
                    type="primary",
                    use_container_width=True
                ):
                    st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Dataset 1 — Primary")
            file1 = st.file_uploader(
                "Upload first file",
                type=["csv", "xlsx", "xls"],
                key="file1"
            )
            if file1:
                df1 = load_file(file1)
                if df1 is not None:
                    st.session_state.uploaded_df = df1
                    st.session_state.uploaded_filename = file1.name
                    st.success(f"✅ {file1.name} — {len(df1):,} rows")
                    st.dataframe(df1.head(3), use_container_width=True)
        with col2:
            st.markdown("### Dataset 2 — Comparison")
            file2 = st.file_uploader(
                "Upload second file",
                type=["csv", "xlsx", "xls"],
                key="file2"
            )
            if file2:
                df2 = load_file(file2)
                if df2 is not None:
                    st.session_state.comparison_df = df2
                    st.session_state.comparison_filename = file2.name
                    st.success(f"✅ {file2.name} — {len(df2):,} rows")
                    st.dataframe(df2.head(3), use_container_width=True)

        if (
            st.session_state.uploaded_df is not None and
            st.session_state.comparison_df is not None
        ):
            if st.button(
                "Continue to Data Quality Check →",
                type="primary",
                use_container_width=True
            ):
                st.rerun()


# ============================================================
# DATA QUALITY
# ============================================================

def detect_pii(df):
    pii_found = []
    patterns = [
        (re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ), "email addresses"),
        (re.compile(
            r'\b(\+?\d[\d\s\-().]{7,}\d)\b'
        ), "phone numbers"),
        (re.compile(
            r'\b(?:\d[ -]?){13,16}\b'
        ), "possible card numbers"),
        (re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        ), "national ID numbers"),
    ]
    for col in df.columns:
        col_str = df[col].astype(str)
        for pattern, label in patterns:
            if col_str.str.contains(pattern, regex=True).any():
                pii_found.append((col, label))
                break
    return pii_found


def run_quality_report(df):
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())
    duplicate_count = int(df.duplicated().sum())
    pii_found = detect_pii(df)
    score = 100
    total_cells = df.shape[0] * df.shape[1]
    if total_cells > 0:
        score -= min((total_missing / total_cells) * 200, 30)
    if duplicate_count > 0:
        score -= min((duplicate_count / len(df)) * 200, 20)
    if pii_found:
        score -= 15
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing": total_missing,
        "missing_per_col": missing_per_col[missing_per_col > 0].to_dict(),
        "duplicate_count": duplicate_count,
        "pii_found": pii_found,
        "quality_score": max(0, round(score)),
        "numeric_cols": df.select_dtypes(
            include='number'
        ).columns.tolist(),
        "categorical_cols": df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist(),
        "date_cols": [
            c for c in df.columns
            if 'date' in c.lower() or 'time' in c.lower()
        ],
    }


def screen_data_quality():
    show_progress("data_quality")
    df = st.session_state.uploaded_df
    st.markdown("## 📋 Data Quality Report")
    st.markdown(f"**Dataset:** {st.session_state.uploaded_filename}")

    if st.session_state.quality_report is None:
        with st.spinner("Scanning dataset..."):
            st.session_state.quality_report = run_quality_report(df)

    report = st.session_state.quality_report
    score = report["quality_score"]
    score_color = (
        "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    )
    score_label = (
        "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"
    )

    st.markdown(
        f"### {score_color} Quality Score: **{score}/100** — {score_label}"
    )
    st.progress(score / 100)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{report['rows']:,}")
    with col2:
        st.metric("Columns", report['columns'])
    with col3:
        st.metric("Missing Values", report['total_missing'])
    with col4:
        st.metric("Duplicates", report['duplicate_count'])

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            f"📊 **{len(report['numeric_cols'])}** Numeric\n\n" +
            ", ".join(report['numeric_cols'][:5]) +
            ("..." if len(report['numeric_cols']) > 5 else "")
        )
    with c2:
        st.info(
            f"🏷️ **{len(report['categorical_cols'])}** Categorical\n\n" +
            ", ".join(report['categorical_cols'][:5]) +
            ("..." if len(report['categorical_cols']) > 5 else "")
        )
    with c3:
        st.info(
            f"📅 **{len(report['date_cols'])}** Date\n\n" +
            (", ".join(report['date_cols'])
             if report['date_cols'] else "None detected")
        )

    if report['missing_per_col']:
        st.markdown("---")
        st.warning("⚠️ **Missing Values:**")
        for col, count in report['missing_per_col'].items():
            pct = round((count / report['rows']) * 100, 1)
            st.markdown(f"- **{col}**: {count} missing ({pct}%)")

    if report['duplicate_count'] > 0:
        st.markdown("---")
        st.warning(
            f"⚠️ **{report['duplicate_count']} duplicate rows detected.**"
        )

    if report['pii_found']:
        st.markdown("---")
        st.error("🔴 **PII Detected — Data Protection Notice**")
        for col, label in report['pii_found']:
            st.markdown(f"- **{col}**: {label}")
        st.markdown(
            "> Ensure you have legal right to process "
            "this data before proceeding."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Upload Different File", use_container_width=True):
            st.session_state.uploaded_df = None
            st.session_state.uploaded_filename = None
            st.session_state.quality_report = None
            st.rerun()
    with col2:
        label = (
            "⚠️ Proceed Anyway (PII present)"
            if report['pii_found']
            else "Proceed to Role & Industry →"
        )
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.proceed_to_role = True
            st.rerun()


# ============================================================
# INDUSTRY DETECTION
# ============================================================

def detect_industry_from_columns(df):
    col_string = " ".join([c.lower() for c in df.columns])
    domain_keywords = {
        "Retail & Sales": [
            "sales", "revenue", "store", "product", "category",
            "units", "returns", "customer", "margin", "target",
            "region", "order"
        ],
        "HR & People": [
            "employee", "attrition", "salary", "department",
            "performance", "tenure", "satisfaction", "headcount",
            "absence", "gender", "engagement", "band"
        ],
        "Finance & Budget": [
            "budget", "actual", "variance", "forecast", "cost",
            "expenditure", "spend", "quarter", "centre", "center",
            "approved", "invoice", "payment"
        ],
        "Operations & Manufacturing": [
            "defects", "downtime", "efficiency", "cycle",
            "shift", "line", "throughput", "yield",
            "operator", "output", "oee", "produced"
        ],
        "Marketing": [
            "campaign", "impressions", "clicks", "ctr", "cpl",
            "cac", "roas", "conversion", "channel", "leads", "funnel"
        ],
    }
    scores = {
        domain: sum(1 for kw in keywords if kw in col_string)
        for domain, keywords in domain_keywords.items()
    }
    best = max(scores, key=scores.get)
    secondary = [d for d, s in scores.items() if s > 0 and d != best]
    return (best, secondary) if scores[best] > 0 else ("General Business", [])


# ============================================================
# ROLE PROFILE BUILDER
# ============================================================

def build_role_profile(user_role, industry, df):
    role_lower = user_role.lower()

    level = "L2"
    l1_keywords = [
        "ceo", "cfo", "coo", "cmo", "cto", "chro", "cpo", "cco",
        "chief", "md", "managing director", "board", "founder",
        "chairman", "president", "group director"
    ]
    l2_keywords = [
        "director", "vp", "svp", "evp", "vice president",
        "head of", "general manager", "gm", "bdm",
        "business development manager", "finance manager",
        "sales director", "hr director", "marketing director"
    ]
    l3_keywords = [
        "manager", "hrbp", "hr business partner", "supervisor",
        "branch", "store", "team lead", "section lead"
    ]
    l4a_keywords = [
        "analyst", "scientist", "engineer", "statistician",
        "data", "bi ", "business intelligence"
    ]
    l4f_keywords = [
        "shift supervisor", "floor supervisor",
        "section supervisor", "frontline"
    ]

    if any(kw in role_lower for kw in l1_keywords):
        level = "L1"
    elif any(kw in role_lower for kw in l2_keywords):
        level = "L2"
    elif any(kw in role_lower for kw in l4a_keywords):
        level = "L4A"
    elif any(kw in role_lower for kw in l4f_keywords):
        level = "L4F"
    elif any(kw in role_lower for kw in l3_keywords):
        level = "L3"

    function = "General Management"
    if any(k in role_lower for k in [
        "cfo", "finance", "financial", "fd", "treasurer",
        "budget", "accounting"
    ]):
        function = "Finance"
    elif any(k in role_lower for k in [
        "sales", "commercial", "revenue", "account manager",
        "rsm", "regional sales", "business development"
    ]):
        function = "Sales"
    elif any(k in role_lower for k in [
        "hr", "human resource", "people", "talent",
        "hrbp", "recruitment", "workforce", "chro", "cpo"
    ]):
        function = "HR"
    elif any(k in role_lower for k in [
        "operations", "ops", "manufacturing", "supply chain",
        "logistics", "production", "plant", "coo"
    ]):
        function = "Operations"
    elif any(k in role_lower for k in [
        "marketing", "brand", "digital", "growth", "campaign", "cmo"
    ]):
        function = "Marketing"
    elif any(k in role_lower for k in [
        "ceo", "md", "board", "founder", "chairman",
        "president", "chief executive", "group director"
    ]):
        function = "Executive"

    governing_questions = {
        "L1": "What does this mean for the organisation as a whole and what strategic decisions must I make now?",
        "L2": "What should I focus on this month, what is at risk, and what do I need to escalate or delegate?",
        "L3": "What do I do next, what is underperforming in my area, and what does my team need to know right now?",
        "L4F": "What is the number and what specific action do I take today?",
        "L4A": "What does this data show statistically and what are the methodological caveats?",
    }
    governing_question = governing_questions.get(level, governing_questions["L2"])

    language_rules = {
        "L1": "Max 20 words per sentence. Strategic headlines only. No operational detail. No hedging. Decisive. Translate numbers to business impact.",
        "L2": "Max 25 words per sentence. Performance vs target focus. Always state variance with baseline. Businesslike and direct.",
        "L3": "Max 30 words per sentence. Plain English. Action-oriented. Team-focused. Explain what numbers mean.",
        "L4F": "Max 15 words per sentence. Number then action only.",
        "L4A": "Flexible length. Full statistical methodology. All assumptions explicit. Technical peer-level.",
    }
    language_rule = language_rules.get(level, language_rules["L2"])

    universal_exclusions = [
        "row id", "order id", "customer id", "customer name",
        "postal code", "product id", "country"
    ]

    if level == "L1" and function == "Executive":
        inclusions = ["sales", "profit", "margin", "cost", "shipping", "discount", "revenue"]
        exclusions = universal_exclusions
        kpi_focus = ["Total Revenue", "Total Profit", "Profit Margin %", "Shipping Cost Total"]
        chart_focus = ["Profit by category", "Sales trend over time", "Sales vs profit by segment", "Shipping cost by category"]
    elif function == "Finance" and level in ["L1", "L2"]:
        inclusions = ["sales", "profit", "margin", "discount", "shipping", "cost", "revenue"]
        exclusions = universal_exclusions
        kpi_focus = ["Total Revenue", "Total Profit", "Profit Margin %", "Total Discount Impact", "Shipping Cost Total"]
        chart_focus = ["Profit margin by category", "Discount impact on profit", "Shipping cost by category", "Revenue vs profit over time"]
    elif function == "Sales" or ("sales" in role_lower and level in ["L2", "L3"]):
        inclusions = ["sales", "profit", "quantity", "discount", "shipping", "revenue", "target"]
        exclusions = universal_exclusions
        kpi_focus = ["Total Sales", "Average Order Value", "Total Orders", "Profit per Order"]
        chart_focus = ["Sales by region", "Sales by category", "Sales trend over time", "Profit by segment"]
    elif function == "HR":
        inclusions = ["attrition", "salary", "performance", "tenure", "satisfaction", "headcount", "absence", "engagement"]
        exclusions = universal_exclusions + ["sales", "profit", "shipping", "product"]
        kpi_focus = ["Attrition Rate", "Average Tenure", "Headcount", "Average Satisfaction"]
        chart_focus = ["Attrition by department", "Satisfaction distribution", "Tenure vs performance", "Headcount trend"]
    elif function == "Operations":
        inclusions = ["units", "defects", "downtime", "efficiency", "cycle", "throughput", "yield", "output"]
        exclusions = universal_exclusions + ["customer", "revenue", "marketing"]
        kpi_focus = ["Overall Efficiency %", "Total Defects", "Total Downtime", "Output vs Target"]
        chart_focus = ["Efficiency trend over time", "Defect rate by line", "Downtime breakdown", "Output vs target"]
    else:
        inclusions = ["sales", "revenue", "profit", "cost", "target", "performance", "margin"]
        exclusions = universal_exclusions
        kpi_focus = ["Total Revenue", "Total Profit", "Key Variance"]
        chart_focus = ["Performance trend", "Key metric breakdown"]

    stats_depth = {
        "L1": "headline", "L2": "moderate", "L3": "moderate",
        "L4F": "outputs_only", "L4A": "full",
    }.get(level, "moderate")

    return {
        "role_raw": user_role,
        "level": level,
        "function": function,
        "governing_question": governing_question,
        "language_rule": language_rule,
        "inclusions": inclusions,
        "exclusions": exclusions,
        "kpi_focus": kpi_focus,
        "chart_focus": chart_focus,
        "stats_depth": stats_depth,
    }


# ============================================================
# SCREEN 5 — ROLE INPUT
# ============================================================

def screen_role_input():
    show_progress("role_input")
    df = st.session_state.uploaded_df

    st.markdown("## 🏭 Industry & Role Setup")

    if st.session_state.detected_industry is None:
        detected, secondary = detect_industry_from_columns(df)
        st.session_state.detected_industry = detected
        st.session_state.detected_secondary = secondary

    detected = st.session_state.detected_industry
    secondary = st.session_state.detected_secondary

    st.markdown("---")
    st.markdown("### Step 1 — Confirm Your Industry")

    msg = (
        f"We detected: **{detected}**" +
        (f" with elements of: **{', '.join(secondary)}**" if secondary else "")
    )
    st.info(f"🔍 {msg}")

    industry_choice = st.radio(
        "Is this correct?",
        options=[f"✅ Yes — {detected} is correct", "✏️ No — I will describe it myself"],
        index=0
    )

    confirmed_industry = detected
    if "No" in industry_choice:
        custom = st.text_input(
            "Describe your industry:",
            placeholder="e.g. E-commerce, NHS healthcare, logistics..."
        )
        if custom:
            confirmed_industry = custom
            st.success(f"✅ Set to: **{custom}**")
    else:
        st.success(f"✅ Confirmed: **{detected}**")

    st.markdown("---")
    st.markdown("### Step 2 — Enter Your Role")

    user_role = st.text_input(
        "Your role or job title:",
        placeholder="e.g. CFO, Regional Sales Manager, Store Manager...",
        value=st.session_state.user_role or ""
    )

    injection_keywords = [
        "ignore", "reveal", "system prompt", "you are now",
        "forget", "override", "jailbreak", "pretend",
        "act as", "drop table", "select *", "<script"
    ]
    role_is_safe = True
    if user_role:
        if any(kw in user_role.lower() for kw in injection_keywords):
            st.error("⚠️ This input cannot be processed. Please enter a valid role.")
            role_is_safe = False

    if user_role and role_is_safe and len(user_role) > 2:
        profile = build_role_profile(user_role, confirmed_industry, df)
        st.markdown("---")
        st.markdown("**How we will interpret your role:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Level:** {profile['level']}")
        with col2:
            st.info(f"**Function:** {profile['function']}")
        with col3:
            st.info(f"**Depth:** {profile['stats_depth']}")
        st.caption(f"🎯 Governing question: *{profile['governing_question']}*")

    st.markdown("---")
    button_ready = user_role and role_is_safe and confirmed_industry

    if st.button(
        "🚀 Generate My Dashboard →",
        use_container_width=True,
        type="primary",
        disabled=not button_ready
    ):
        st.session_state.confirmed_industry = confirmed_industry
        st.session_state.user_role = user_role
        st.session_state.ready_to_analyse = True
        st.session_state.dashboard_result = None
        st.session_state.precomputed_stats = None
        st.session_state.role_profile = None
        st.session_state.show_narrative = False
        st.session_state.chat_history = []
        st.session_state.suggested_questions = []
        st.session_state.chat_input_key = 0
        st.rerun()


# ============================================================
# COLUMN RELEVANCE
# ============================================================

def get_relevant_columns(df, role_profile):
    inclusions = role_profile["inclusions"]
    exclusions = role_profile["exclusions"]
    level = role_profile["level"]

    meaningful = []
    for col in df.columns:
        col_lower = col.lower()
        if any(ex in col_lower for ex in exclusions):
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0 and non_null.std() > 0:
                meaningful.append(col)
        elif pd.api.types.is_object_dtype(series):
            n_unique = series.nunique()
            if 2 <= n_unique <= 25:
                meaningful.append(col)
        elif 'date' in col_lower or 'time' in col_lower:
            meaningful.append(col)

    if level == "L1":
        priority = [col for col in meaningful if any(inc in col.lower() for inc in inclusions)]
        others = [col for col in meaningful if col not in priority]
        role_relevant = priority + others
    else:
        priority = [col for col in meaningful if any(inc in col.lower() for inc in inclusions)]
        others = [col for col in meaningful if col not in priority]
        role_relevant = priority + others[:5]

    return role_relevant if role_relevant else meaningful


# ============================================================
# PRE-COMPUTATION ENGINE
# ============================================================

def precompute_statistics(df, relevant_cols, role_profile, df_name="Dataset"):
    level = role_profile["level"]

    stats = {
        "dataset_name": df_name,
        "total_rows": len(df),
        "relevant_cols": relevant_cols,
        "role_level": level,
        "role_function": role_profile["function"],
        "numeric_stats": {},
        "categorical_stats": {},
        "time_series": {},
        "correlations": {},
        "target_vs_actual": {},
        "anomalies": [],
        "period_info": None,
    }

    numeric_cols = [c for c in relevant_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in relevant_cols if pd.api.types.is_object_dtype(df[c])]
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        mean_val = float(series.mean())
        std_val = float(series.std()) if len(series) > 1 else 0
        stats["numeric_stats"][col] = {
            "mean": round(mean_val, 2),
            "median": round(float(series.median()), 2),
            "std": round(std_val, 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "total": round(float(series.sum()), 2),
            "count": int(len(series)),
            "missing": int(df[col].isnull().sum()),
            "outlier_count": int(((series - mean_val).abs() > 2 * std_val).sum()) if std_val > 0 else 0,
        }

    target_keywords = ['target', 'budget', 'plan', 'quota', 'forecast']
    actual_keywords = ['actual', 'sales', 'revenue', 'spend', 'achieved']
    target_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in target_keywords)]
    actual_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in actual_keywords)]
    for t_col in target_cols:
        for a_col in actual_cols:
            if t_col != a_col:
                t_total = df[t_col].sum()
                a_total = df[a_col].sum()
                if t_total > 0:
                    variance_pct = round(((a_total - t_total) / t_total) * 100, 2)
                    key = f"{a_col}_vs_{t_col}"
                    stats["target_vs_actual"][key] = {
                        "actual_col": a_col,
                        "target_col": t_col,
                        "actual_total": round(float(a_total), 2),
                        "target_total": round(float(t_total), 2),
                        "variance_pct": variance_pct,
                        "variance_abs": round(float(a_total - t_total), 2),
                        "status": ("GREEN" if variance_pct >= -5 else "AMBER" if variance_pct >= -15 else "RED")
                    }

    max_categories = 5 if level == "L1" else 15
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        breakdown = {}
        for numeric_col in numeric_cols[:4]:
            try:
                group = df.groupby(col)[numeric_col].agg(['sum', 'mean', 'count'])
                breakdown[numeric_col] = {
                    str(k): {"sum": round(float(v['sum']), 2), "mean": round(float(v['mean']), 2), "count": int(v['count'])}
                    for k, v in list(group.iterrows())[:max_categories]
                }
            except Exception:
                pass
        stats["categorical_stats"][col] = {
            "unique_values": int(value_counts.nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.head(max_categories).items()},
            "breakdown": breakdown
        }

    if date_cols:
        date_col = date_cols[0]
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            if len(df_copy) > 0:
                min_date = df_copy[date_col].min()
                max_date = df_copy[date_col].max()
                stats["period_info"] = {
                    "date_column": date_col,
                    "from": str(min_date.date()),
                    "to": str(max_date.date()),
                    "days": (max_date - min_date).days
                }
                df_copy['_period'] = df_copy[date_col].dt.to_period('M')
                for col in numeric_cols[:4]:
                    try:
                        monthly = df_copy.groupby('_period')[col].sum()
                        stats["time_series"][col] = {str(k): round(float(v), 2) for k, v in monthly.items()}
                    except Exception:
                        pass
        except Exception:
            pass

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            continue
        outliers = series[(series - mean_val).abs() > 2.5 * std_val]
        if len(outliers) > 0:
            stats["anomalies"].append({
                "column": col,
                "outlier_count": len(outliers),
                "outlier_values": [round(float(v), 2) for v in outliers.head(3)],
                "mean": round(float(mean_val), 2),
                "std": round(float(std_val), 2),
            })

    if level == "L4A" or len(numeric_cols) <= 6:
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                strong_corrs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) >= 0.6 and not np.isnan(corr_val):
                            strong_corrs.append({
                                "col1": numeric_cols[i],
                                "col2": numeric_cols[j],
                                "correlation": round(float(corr_val), 3)
                            })
                stats["correlations"] = strong_corrs
            except Exception:
                pass

    return stats


# ============================================================
# COMPARISON PRE-COMPUTATION
# ============================================================

def precompute_comparison(df1, df2, role_profile, name1, name2):
    relevant1 = get_relevant_columns(df1, role_profile)
    common_cols = [c for c in relevant1 if c in df2.columns]
    stats1 = precompute_statistics(df1, common_cols, role_profile, name1)
    stats2 = precompute_statistics(df2, common_cols, role_profile, name2)
    comparison = {
        "dataset1_name": name1,
        "dataset2_name": name2,
        "numeric_comparison": {},
        "improved": [],
        "declined": [],
        "stable": [],
    }
    numeric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(df1[c]) and c in df2.columns and pd.api.types.is_numeric_dtype(df2[c])
    ]
    for col in numeric_cols:
        val1 = stats1["numeric_stats"].get(col, {}).get("total", 0)
        val2 = stats2["numeric_stats"].get(col, {}).get("total", 0)
        if val1 == 0:
            continue
        change_pct = round(((val2 - val1) / abs(val1)) * 100, 2)
        direction = ("improved" if change_pct > 2 else "declined" if change_pct < -2 else "stable")
        comparison["numeric_comparison"][col] = {
            "dataset1_total": val1,
            "dataset2_total": val2,
            "change_pct": change_pct,
            "direction": direction
        }
        if direction == "improved":
            comparison["improved"].append(col)
        elif direction == "declined":
            comparison["declined"].append(col)
        else:
            comparison["stable"].append(col)
    return {"stats1": stats1, "stats2": stats2, "comparison": comparison}


# ============================================================
# AI PROMPT BUILDER
# ============================================================

def build_analysis_prompt(stats, role_profile, industry, is_comparison=False, comparison_data=None):
    numeric_cols_available = list(stats.get("numeric_stats", {}).keys())
    categorical_cols_available = list(stats.get("categorical_stats", {}).keys())
    time_series_cols_available = list(stats.get("time_series", {}).keys())
    target_vs_actual_available = list(stats.get("target_vs_actual", {}).keys())

    columns_manifest = f"""
EXACT AVAILABLE COLUMNS — USE ONLY THESE IN CHART FIELDS:

Numeric columns (y_field options):
{json.dumps(numeric_cols_available)}

Categorical columns (x_field options for breakdowns):
{json.dumps(categorical_cols_available)}

Time series columns (y_field when data_source is time_series):
{json.dumps(time_series_cols_available)}

Target vs Actual keys (only use if non-empty):
{json.dumps(target_vs_actual_available)}

CHART FIELD RULES — NON NEGOTIABLE:
- x_field MUST be an exact name from the lists above
- y_field MUST be an exact name from the lists above
- data_source categorical_stats → x_field from categorical list
- data_source time_series → y_field from time series list
- data_source target_vs_actual → only if list above is non-empty
- NEVER invent column names not in these lists
- If a chart idea cannot use available columns, choose a different chart
"""

    stats_json = json.dumps(stats, indent=2, default=str)

    comparison_section = ""
    if is_comparison and comparison_data:
        comparison_section = f"""
COMPARISON MODE:
Dataset 1: {comparison_data['stats1']['dataset_name']}
Dataset 2: {comparison_data['stats2']['dataset_name']}
Changes: {json.dumps(comparison_data['comparison'], indent=2)}
Improved: {comparison_data['comparison']['improved']}
Declined: {comparison_data['comparison']['declined']}
Stable: {comparison_data['comparison']['stable']}
Generate comparative insights.
Root cause hypothesis required for any decline above 10%.
"""

    exclusion_text = (f"\nEXPLICIT EXCLUSIONS — never reference: {', '.join(role_profile['exclusions'])}" if role_profile["exclusions"] else "")
    inclusion_text = f"\nPRIORITY FOCUS: {', '.join(role_profile['inclusions'])}"
    kpi_text = f"\nKPI CARDS FOCUS ON: {', '.join(role_profile['kpi_focus'])}"
    chart_text = f"\nCHART BUSINESS QUESTIONS (use available columns above): {', '.join(role_profile['chart_focus'])}"

    prompt = f"""You are an enterprise-grade business intelligence analyst.
You receive pre-computed statistics from a full dataset.
Generate a complete, accurate, role-specific business intelligence dashboard.

ROLE: {role_profile['role_raw']}
ASSIGNED LEVEL: {role_profile['level']}
ASSIGNED FUNCTION: {role_profile['function']}
INDUSTRY: {industry}

GOVERNING QUESTION — answer this and only this:
{role_profile['governing_question']}

Every insight, KPI, chart, and recommendation must directly answer
this governing question. Exclude anything that does not.

LANGUAGE RULES — NON NEGOTIABLE:
{role_profile['language_rule']}
{exclusion_text}
{inclusion_text}
{kpi_text}
{chart_text}

{columns_manifest}

ABSOLUTE RULES:
- Every number must come from pre-computed statistics only
- Never invent data, trends, or patterns not in statistics
- Never present uncertain findings as facts
- Traffic lights ONLY if target_vs_actual list is non-empty — else []
- Never fabricate thresholds
- x_field and y_field must be exact names from column manifest
- CEO and Store Manager analysing same data must differ completely:
  different framing, different language, different actions

CONFIDENCE:
HIGH: 100+ rows, consistent pattern across 3+ periods or categories
MEDIUM: 30-99 rows, directional
INDICATIVE: below 30 rows
CONFIDENCE_OVERALL = lowest individual confidence

PRE-COMPUTED STATISTICS:
{stats_json}

{comparison_section}

Return ONLY valid JSON. No markdown. No explanation. Exact schema:

{{
  "role_interpreted": "string",
  "level": "{role_profile['level']}",
  "function": "{role_profile['function']}",
  "executive_summary": {{
    "sentence_1": "string",
    "sentence_2": "string",
    "sentence_3": "string"
  }},
  "kpi_cards": [
    {{
      "label": "string",
      "value": "string",
      "delta": "string or null",
      "delta_positive": true,
      "context": "string"
    }}
  ],
  "traffic_lights": [],
  "anomalies": [
    {{
      "severity": "HIGH|MEDIUM|LOW",
      "description": "string",
      "metric": "string",
      "value": "string",
      "expected": "string"
    }}
  ],
  "charts": [
    {{
      "type": "bar|line|pie|scatter|waterfall|ranked_list",
      "title": "string",
      "data_source": "categorical_stats|time_series|target_vs_actual|numeric_stats",
      "x_field": "string — exact name from manifest",
      "y_field": "string — exact name from manifest",
      "group_by": "string or null",
      "caption": "string",
      "sentiment": "POSITIVE|NEGATIVE|NEUTRAL|URGENT",
      "confidence": "HIGH|MEDIUM|INDICATIVE",
      "verified": true
    }}
  ],
  "recommendations": [
    {{
      "priority": 1,
      "action": "string",
      "rationale": "string",
      "owner": "string",
      "timeframe": "IMMEDIATE|SHORT_TERM|STRATEGIC",
      "data_evidence": "string"
    }}
  ],
  "narrative": {{
    "opening": "string",
    "body": ["string"],
    "close": "string"
  }},
  "evaluation": {{
    "relevance_score": 8,
    "accuracy_validated": "YES|PARTIAL|NO",
    "coverage": "string",
    "confidence_overall": "HIGH|MEDIUM|INDICATIVE",
    "bias_check": "BALANCED|IMBALANCED|NOT_APPLICABLE",
    "bias_detail": "string or null",
    "evaluation_status": "COMPLETE|PARTIAL|FAILED"
  }},
  "comparison_insights": []
}}"""

    return prompt


# ============================================================
# OPENAI API CALL
# ============================================================

def call_openai(prompt, api_key, progress_placeholder):
    import time
    steps = [
        "📂 Reading dataset and role profile...",
        "🎯 Applying governing question for your role...",
        "📊 Matching available columns to chart types...",
        "🧠 Generating role-specific insights...",
        "📈 Building charts from available data...",
        "🚦 Evaluating traffic light evidence...",
        "⚠️ Running anomaly detection...",
        "✅ Applying quality and accuracy gates...",
        "📝 Finalising recommendations and narrative...",
    ]
    for i, step in enumerate(steps):
        progress_placeholder.info(f"**Step {i+1} of {len(steps)}:** {step}")
        time.sleep(0.35)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4000,
        temperature=0.15,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise BI analyst. "
                    "Return only valid JSON matching the exact schema. "
                    "No markdown. No explanation. "
                    "x_field and y_field must be exact column names from the manifest provided. "
                    "Every insight must answer the governing question."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI response could not be parsed. Details: {str(e)} | Preview: {raw[:300]}")


# ============================================================
# KPI SMART ROUNDING
# ============================================================

def format_kpi_value(value_str):
    if not value_str:
        return value_str
    currency = ""
    clean = value_str.strip()
    for symbol in ["$", "£", "€"]:
        if clean.startswith(symbol):
            currency = symbol
            clean = clean[1:]
            break
    clean = clean.replace(",", "").strip()
    try:
        num = float(clean)
    except ValueError:
        return value_str
    if abs(num) >= 1_000_000:
        rounded = f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        rounded = f"{num / 1_000:.1f}K"
    else:
        rounded = str(int(num)) if num == int(num) else f"{num:.2f}"
    return f"{currency}{rounded}"


# ============================================================
# CHART RENDERER
# ============================================================

def render_chart(chart_spec, stats, df):
    chart_type = chart_spec.get("type", "bar")
    title = chart_spec.get("title", "Chart")
    data_source = chart_spec.get("data_source", "")
    x_field = chart_spec.get("x_field", "")
    y_field = chart_spec.get("y_field", "")

    COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#6366f1", "#8b5cf6", "#06b6d4", "#ec4899"]
    fig = None

    try:
        if data_source == "time_series":
            ts_data = stats.get("time_series", {})
            matched_col = None
            if y_field in ts_data:
                matched_col = y_field
            else:
                for col in ts_data:
                    if col.lower() in y_field.lower() or y_field.lower() in col.lower():
                        matched_col = col
                        break
                if not matched_col and ts_data:
                    matched_col = list(ts_data.keys())[0]
            if matched_col:
                labels = list(ts_data[matched_col].keys())
                values = list(ts_data[matched_col].values())
                fig = px.line(x=labels, y=values, title=title, labels={"x": "Period", "y": matched_col}, color_discrete_sequence=COLORS)
                fig.update_traces(line=dict(width=2.5), mode='lines+markers')

        elif data_source == "categorical_stats":
            cat_stats = stats.get("categorical_stats", {})
            numeric_cols = list(stats.get("numeric_stats", {}).keys())
            matched_cat = None
            if x_field in cat_stats:
                matched_cat = x_field
            else:
                for col in cat_stats:
                    if col.lower() in x_field.lower() or x_field.lower() in col.lower():
                        matched_cat = col
                        break
                if not matched_cat and cat_stats:
                    matched_cat = list(cat_stats.keys())[0]
            matched_num = None
            if y_field in numeric_cols:
                matched_num = y_field
            else:
                for col in numeric_cols:
                    if col.lower() in y_field.lower() or y_field.lower() in col.lower():
                        matched_num = col
                        break
                if not matched_num and numeric_cols:
                    matched_num = numeric_cols[0]
            if matched_cat and matched_num:
                breakdown = cat_stats[matched_cat].get("breakdown", {})
                if matched_num in breakdown:
                    segments = list(breakdown[matched_num].keys())
                    values = [breakdown[matched_num][s]["sum"] for s in segments]
                else:
                    top_vals = cat_stats[matched_cat].get("top_values", {})
                    segments = list(top_vals.keys())[:10]
                    values = list(top_vals.values())[:10]
                    matched_num = "Count"
                df_plot = pd.DataFrame({matched_cat: segments, matched_num: values}).sort_values(matched_num, ascending=False)
                if chart_type == "pie":
                    fig = px.pie(df_plot, names=matched_cat, values=matched_num, title=title, color_discrete_sequence=COLORS)
                elif chart_type == "ranked_list":
                    df_plot = df_plot.sort_values(matched_num, ascending=True).tail(10)
                    fig = px.bar(df_plot, x=matched_num, y=matched_cat, orientation='h', title=title, color_discrete_sequence=COLORS)
                else:
                    fig = px.bar(df_plot, x=matched_cat, y=matched_num, title=title, color_discrete_sequence=COLORS)

        elif data_source == "target_vs_actual":
            tva = stats.get("target_vs_actual", {})
            if tva:
                key = list(tva.keys())[0]
                item = tva[key]
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=["Target", "Variance", "Actual"],
                    y=[item["target_total"], item["variance_abs"], item["actual_total"]],
                    connector={"line": {"color": "#1e2330"}},
                    increasing={"marker": {"color": "#10b981"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    totals={"marker": {"color": "#3b82f6"}}
                ))
                fig.update_layout(title=title)

        elif data_source == "numeric_stats":
            numeric_stats = stats.get("numeric_stats", {})
            numeric_cols = list(numeric_stats.keys())
            if chart_type == "scatter" and len(numeric_cols) >= 2:
                col_x = (x_field if x_field in df.columns else numeric_cols[0])
                col_y = (y_field if y_field in df.columns else numeric_cols[1] if len(numeric_cols) > 1 else None)
                if col_x and col_y:
                    try:
                        fig = px.scatter(df, x=col_x, y=col_y, title=title, color_discrete_sequence=COLORS, trendline="ols")
                    except Exception:
                        fig = px.scatter(df, x=col_x, y=col_y, title=title, color_discrete_sequence=COLORS)
            else:
                labels = list(numeric_stats.keys())
                values = [numeric_stats[c]["total"] for c in labels]
                if labels:
                    fig = px.bar(x=labels, y=values, title=title, labels={"x": "Metric", "y": "Total"}, color_discrete_sequence=COLORS)

        if fig is None:
            ts_data = stats.get("time_series", {})
            cat_stats = stats.get("categorical_stats", {})
            numeric_stats = stats.get("numeric_stats", {})
            if ts_data:
                first_col = list(ts_data.keys())[0]
                labels = list(ts_data[first_col].keys())
                values = list(ts_data[first_col].values())
                fig = px.line(x=labels, y=values, title=title, labels={"x": "Period", "y": first_col}, color_discrete_sequence=COLORS)
                fig.update_traces(line=dict(width=2.5), mode='lines+markers')
            elif cat_stats and numeric_stats:
                first_cat = list(cat_stats.keys())[0]
                first_num = list(numeric_stats.keys())[0]
                breakdown = cat_stats[first_cat].get("breakdown", {})
                if first_num in breakdown:
                    segments = list(breakdown[first_num].keys())
                    values = [breakdown[first_num][s]["sum"] for s in segments]
                else:
                    top_vals = cat_stats[first_cat].get("top_values", {})
                    segments = list(top_vals.keys())[:10]
                    values = list(top_vals.values())[:10]
                    first_num = "Count"
                df_plot = pd.DataFrame({first_cat: segments, first_num: values}).sort_values(first_num, ascending=False)
                fig = px.bar(df_plot, x=first_cat, y=first_num, title=title, color_discrete_sequence=COLORS)
            elif numeric_stats:
                labels = list(numeric_stats.keys())
                values = [numeric_stats[c]["total"] for c in labels]
                fig = px.bar(x=labels, y=values, title=title, labels={"x": "Metric", "y": "Total"}, color_discrete_sequence=COLORS)

        if fig:
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                title=dict(font=dict(size=14)),
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")

        return fig

    except Exception:
        return None


# ============================================================
# DASHBOARD RENDERING SECTIONS
# ============================================================

def render_executive_summary(result):
    st.markdown("### 🎯 Executive Summary")
    role = result.get("role_interpreted", "")
    level = result.get("level", "")
    function = result.get("function", "")
    st.info(f"**Role:** {role} &nbsp;|&nbsp; **Level:** {level} &nbsp;|&nbsp; **Function:** {function}")
    summary = result.get("executive_summary", {})
    for i, key in enumerate(["sentence_1", "sentence_2", "sentence_3"], 1):
        sentence = summary.get(key, "")
        if sentence:
            st.markdown(f"**{i}.** {sentence}")


def render_kpi_cards(kpi_cards):
    if not kpi_cards:
        return
    st.markdown("### 📊 Key Performance Indicators")
    cols = st.columns(min(len(kpi_cards), 5))
    for i, kpi in enumerate(kpi_cards[:5]):
        with cols[i % len(cols)]:
            raw_value = kpi.get("value", "")
            formatted_value = format_kpi_value(raw_value)
            delta_val = kpi.get("delta")
            delta_color = "normal"
            if delta_val:
                is_positive = kpi.get("delta_positive", True)
                delta_color = "normal" if is_positive else "inverse"
            st.metric(
                label=kpi.get("label", ""),
                value=formatted_value,
                delta=delta_val,
                delta_color=delta_color,
                help=kpi.get("context", "")
            )


def render_traffic_lights(traffic_lights):
    if not traffic_lights:
        return
    st.markdown("### 🚦 Traffic Light Status")
    status_icons = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
    cols = st.columns(min(len(traffic_lights), 4))
    for i, tl in enumerate(traffic_lights):
        with cols[i % len(cols)]:
            status = tl.get("status", "")
            icon = status_icons.get(status, "⬜")
            st.metric(
                label=f"{icon} {tl.get('metric', '')}",
                value=tl.get("actual_value", ""),
                delta=(f"vs target: {tl.get('target_value', '')} ({tl.get('variance_pct', '')}%)"),
                delta_color=("normal" if status == "GREEN" else "inverse"),
                help=tl.get("reason", "")
            )


def render_anomalies(anomalies):
    if not anomalies:
        st.success("✅ No anomalies detected.")
        return
    st.markdown("### ⚠️ Anomaly Alerts")
    severity_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
    for anomaly in anomalies:
        severity = anomaly.get("severity", "LOW")
        icon = severity_icons.get(severity, "🔵")
        with st.expander(f"{icon} {severity} — {anomaly.get('metric', '')}", expanded=(severity == "HIGH")):
            st.markdown(f"**Finding:** {anomaly.get('description', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Observed:** {anomaly.get('value', '')}")
            with col2:
                st.markdown(f"**Expected:** {anomaly.get('expected', '')}")


def render_charts(charts, stats, df):
    if not charts:
        return
    st.markdown("### 📈 AI-Generated Charts")
    st.caption("Selected dynamically based on your role and governing question")
    sentiment_icons = {"POSITIVE": "📈", "NEGATIVE": "📉", "NEUTRAL": "➡️", "URGENT": "⚡"}
    confidence_icons = {"HIGH": "🔵", "MEDIUM": "🟡", "INDICATIVE": "⚪"}
    i = 0
    while i < len(charts):
        if i + 1 < len(charts):
            col1, col2 = st.columns(2)
            for col, chart_spec in zip([col1, col2], [charts[i], charts[i + 1]]):
                with col:
                    _render_single_chart(chart_spec, stats, df, sentiment_icons, confidence_icons)
            i += 2
        else:
            _render_single_chart(charts[i], stats, df, sentiment_icons, confidence_icons)
            i += 1


def _render_single_chart(chart_spec, stats, df, sentiment_icons, confidence_icons):
    title = chart_spec.get("title", "Chart")
    caption = chart_spec.get("caption", "")
    sentiment = chart_spec.get("sentiment", "NEUTRAL")
    confidence = chart_spec.get("confidence", "MEDIUM")
    verified = chart_spec.get("verified", False)
    sent_icon = sentiment_icons.get(sentiment, "➡️")
    conf_icon = confidence_icons.get(confidence, "⚪")
    st.markdown(f"**{title}**")
    st.caption(
        f"{sent_icon} {sentiment} &nbsp;|&nbsp; {conf_icon} {confidence} Confidence"
        + (" &nbsp;|&nbsp; 🔵 ✓ Verified" if verified else "")
    )
    fig = render_chart(chart_spec, stats, df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"💡 {caption}" if caption else "")
    if caption and fig:
        st.info(f"💡 {caption}")
    st.markdown("---")


def render_recommendations(recommendations):
    if not recommendations:
        return
    st.markdown("### 💡 Recommendations")
    priority_icons = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵", 5: "⚪"}
    timeframe_labels = {"IMMEDIATE": "⚡ Immediate", "SHORT_TERM": "📅 Short Term", "STRATEGIC": "🗺️ Strategic"}
    for rec in sorted(recommendations, key=lambda x: x.get("priority", 5)):
        priority = rec.get("priority", 3)
        icon = priority_icons.get(priority, "⚪")
        timeframe = timeframe_labels.get(rec.get("timeframe", ""), rec.get("timeframe", ""))
        with st.expander(f"{icon} Priority {priority} — {rec.get('action', '')}", expanded=(priority == 1)):
            st.markdown(f"**Rationale:** {rec.get('rationale', '')}")
            st.markdown(f"**Data Evidence:** {rec.get('data_evidence', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Owner:** {rec.get('owner', '')}")
            with col2:
                st.markdown(f"**Timeframe:** {timeframe}")


def render_comparison_insights(comparison_insights):
    if not comparison_insights:
        return
    st.markdown("### 🔄 Comparison Analysis")
    direction_icons = {"improved": "📈", "declined": "📉", "stable": "➡️"}
    improved = [c for c in comparison_insights if c.get("direction") == "improved"]
    declined = [c for c in comparison_insights if c.get("direction") == "declined"]
    stable = [c for c in comparison_insights if c.get("direction") == "stable"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Improved", len(improved))
    with col2:
        st.metric("Declined", len(declined))
    with col3:
        st.metric("Stable", len(stable))
    st.markdown("---")
    for insight in comparison_insights:
        direction = insight.get("direction", "stable")
        icon = direction_icons.get(direction, "➡️")
        with st.expander(f"{icon} {insight.get('metric', '')} — {insight.get('change_pct', '')}% change", expanded=(direction == "declined")):
            st.markdown(f"**Implication:** {insight.get('business_implication', '')}")
            st.markdown(f"**Root Cause:** {insight.get('root_cause_hypothesis', '')}")


def render_narrative(narrative):
    st.markdown("### 📝 Narrative Report")
    st.caption("Professional business document — written for your role")
    opening = narrative.get("opening", "")
    body = narrative.get("body", [])
    close = narrative.get("close", "")
    full_text = ""
    if opening:
        full_text += opening + "\n\n"
    for para in body:
        full_text += para + "\n\n"
    if close:
        full_text += close
    st.markdown(full_text)
    if full_text:
        st.download_button(
            label="📥 Download as Text File",
            data=full_text,
            file_name=f"narrative_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )


def render_evaluation(evaluation):
    st.markdown("### 🔬 Evaluation Metadata")
    st.caption("6 quality methods applied automatically")
    score = evaluation.get("relevance_score", 0)
    acc = evaluation.get("accuracy_validated", "")
    conf = evaluation.get("confidence_overall", "")
    bias = evaluation.get("bias_check", "")
    status = evaluation.get("evaluation_status", "")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Relevance", f"{score}/10")
    with col2:
        acc_icon = ("✅" if acc == "YES" else "⚠️" if acc == "PARTIAL" else "❌")
        st.metric("Accuracy", f"{acc_icon} {acc}")
    with col3:
        st.metric("Coverage", evaluation.get("coverage", "—"))
    with col4:
        conf_icon = {"HIGH": "🔵", "MEDIUM": "🟡", "INDICATIVE": "⚪"}.get(conf, "")
        st.metric("Confidence", f"{conf_icon} {conf}")
    with col5:
        bias_icon = ("✅" if bias == "BALANCED" else "⚠️" if bias == "IMBALANCED" else "➡️")
        st.metric("Bias", f"{bias_icon} {bias}")
    with col6:
        status_icon = "✅" if status == "COMPLETE" else "⚠️"
        st.metric("Status", f"{status_icon} {status}")
    if evaluation.get("bias_detail"):
        st.caption(f"Bias detail: {evaluation['bias_detail']}")


def render_precomputed_stats(stats):
    if not stats:
        return
    st.markdown("**Pre-Computed Statistics — Full Dataset**")
    st.caption("Every number the AI references validated against these values")
    if stats.get("numeric_stats"):
        rows = []
        for col, s in stats["numeric_stats"].items():
            rows.append({
                "Column": col,
                "Total": f"{s['total']:,.2f}",
                "Mean": f"{s['mean']:,.2f}",
                "Min": f"{s['min']:,.2f}",
                "Max": f"{s['max']:,.2f}",
                "Outliers": s['outlier_count']
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if stats.get("period_info"):
        p = stats["period_info"]
        st.info(f"📅 Date range: **{p['from']}** to **{p['to']}** ({p['days']} days)")
    if stats.get("correlations"):
        st.markdown("**Strong Correlations:**")
        for corr in stats["correlations"]:
            direction = ("positive" if corr["correlation"] > 0 else "negative")
            st.markdown(f"- **{corr['col1']}** ↔ **{corr['col2']}**: {corr['correlation']} ({direction})")


# ============================================================
# CHAT FUNCTIONS — NATURAL LANGUAGE QUERY (FEATURE 12)
# ============================================================

def build_stats_summary_for_chat(stats):
    """
    Builds a compact text summary of pre-computed stats.
    This is what gets sent to the AI for chat — not the raw data.
    Keeps token cost low.
    """
    lines = []
    lines.append(f"Dataset: {stats.get('dataset_name', 'Unknown')}")
    lines.append(f"Total rows: {stats.get('total_rows', 0)}")

    if stats.get("numeric_stats"):
        lines.append("\nNumeric column totals and averages:")
        for col, s in stats["numeric_stats"].items():
            lines.append(f"  {col}: total={s['total']}, mean={s['mean']}, min={s['min']}, max={s['max']}")

    if stats.get("categorical_stats"):
        lines.append("\nCategorical breakdowns (top values by count):")
        for col, s in stats["categorical_stats"].items():
            top = list(s.get("top_values", {}).items())[:5]
            top_str = ", ".join([f"{k}={v}" for k, v in top])
            lines.append(f"  {col}: {top_str}")
            # Include numeric breakdowns if available
            if s.get("breakdown"):
                for num_col, breakdown in list(s["breakdown"].items())[:2]:
                    sums = {k: v["sum"] for k, v in list(breakdown.items())[:5]}
                    sums_str = ", ".join([f"{k}={v}" for k, v in sums.items()])
                    lines.append(f"    {col} by {num_col} (sum): {sums_str}")

    if stats.get("time_series"):
        lines.append("\nTime series (monthly totals):")
        for col, ts in stats["time_series"].items():
            periods = list(ts.items())
            lines.append(f"  {col}: {len(periods)} periods from {periods[0][0]} to {periods[-1][0]}")
            # Show last 3 periods
            for period, val in periods[-3:]:
                lines.append(f"    {period}: {val}")

    if stats.get("target_vs_actual"):
        lines.append("\nTarget vs Actual:")
        for key, item in stats["target_vs_actual"].items():
            lines.append(f"  {item['actual_col']} vs {item['target_col']}: actual={item['actual_total']}, target={item['target_total']}, variance={item['variance_pct']}% ({item['status']})")

    if stats.get("anomalies"):
        lines.append("\nAnomalies detected:")
        for a in stats["anomalies"]:
            lines.append(f"  {a['column']}: {a['outlier_count']} outliers (mean={a['mean']}, std={a['std']})")

    if stats.get("period_info"):
        p = stats["period_info"]
        lines.append(f"\nDate range: {p['from']} to {p['to']} ({p['days']} days)")

    return "\n".join(lines)


def generate_suggested_questions(role, industry, stats, api_key):
    """
    One small API call to generate 4-5 role-specific questions.
    Uses the stats summary — not raw data.
    Returns a list of question strings.
    """
    stats_summary = build_stats_summary_for_chat(stats)

    prompt = f"""You are a business intelligence assistant.
Generate exactly 4 short, specific questions that a {role} in {industry} would want to ask about their data.
Base questions on actual patterns in the statistics provided.
Questions must be answerable from the statistics provided.
Return ONLY a JSON array of 4 strings. No markdown. No explanation.

Role: {role}
Industry: {industry}

Statistics summary:
{stats_summary}

Return format:
["Question 1?", "Question 2?", "Question 3?", "Question 4?"]"""

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Return only a valid JSON array of 4 question strings. No markdown. No explanation."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q) for q in questions[:5]]
        return []
    except Exception:
        return [
            "Which metric had the highest value overall?",
            "What was the trend over time?",
            "Which category performed best?",
            "Were there any unusual patterns in the data?"
        ]


def answer_chat_question(question, role, industry, stats, api_key):
    """
    One small API call to answer a single user question.
    Uses stats summary only — never re-sends raw data.
    Returns a dict with answer text, optional chart spec, and follow-up.
    """
    stats_summary = build_stats_summary_for_chat(stats)

    # Build list of available column names for chart spec validation
    numeric_cols = list(stats.get("numeric_stats", {}).keys())
    categorical_cols = list(stats.get("categorical_stats", {}).keys())
    time_series_cols = list(stats.get("time_series", {}).keys())

    prompt = f"""You are a business intelligence assistant answering a question for a {role} in {industry}.
Answer using ONLY the statistics provided. Do not invent data.
If the data cannot answer the question, say so clearly and suggest a related question you CAN answer.

Role: {role}
Industry: {industry}

Available column names:
- Numeric: {json.dumps(numeric_cols)}
- Categorical: {json.dumps(categorical_cols)}
- Time series: {json.dumps(time_series_cols)}

Statistics summary:
{stats_summary}

User question: {question}

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{{
  "answer": "string — plain English answer written for a {role}. Use actual numbers from statistics.",
  "data_available": true,
  "suggested_followup": "string — one follow-up question the data CAN answer, or null",
  "chart_needed": false,
  "chart": null
}}

If a chart genuinely helps (e.g. comparison across categories, trend over time):
Set chart_needed to true and provide:
{{
  "answer": "string",
  "data_available": true,
  "suggested_followup": "string or null",
  "chart_needed": true,
  "chart": {{
    "type": "bar|line|pie|ranked_list",
    "title": "string",
    "data_source": "categorical_stats|time_series|numeric_stats",
    "x_field": "exact column name from available columns above",
    "y_field": "exact column name from available columns above",
    "caption": "string — one sentence insight"
  }}
}}

Rules:
- x_field and y_field must be exact names from the available columns lists above
- chart_needed must be false for single-number answers
- chart_needed must be false if no suitable columns exist for the chart
- answer must use actual numbers from the statistics
- If data_available is false, explain what is missing"""

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.15,
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the exact schema. No markdown. No explanation."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "answer": f"I could not process that question. Please try rephrasing it. (Error: {str(e)[:100]})",
            "data_available": False,
            "suggested_followup": None,
            "chart_needed": False,
            "chart": None
        }


def render_chat_panel(stats, role, industry):
    """
    Renders the full chat interface at the bottom of the dashboard.
    Uses only native Streamlit components.
    """
    st.markdown("### 💬 Ask a Question About Your Data")
    st.caption("Ask anything about your dataset in plain English")
    st.divider()

    # --- SUGGESTED QUESTIONS SECTION ---
    if not st.session_state.suggested_questions:
        if st.button("💡 Show Suggested Questions", use_container_width=False):
            with st.spinner("Generating questions for your role..."):
                questions = generate_suggested_questions(
                    role, industry, stats, st.session_state.api_key
                )
                st.session_state.suggested_questions = questions
                st.rerun()
    else:
        st.markdown("**Suggested questions for your role:**")
        # Render suggested questions as buttons in a row
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.suggested_questions)):
            with col:
                if st.button(
                    question,
                    key=f"suggested_q_{i}",
                    use_container_width=True
                ):
                    # Add question to chat and get answer
                    _process_chat_question(question, stats, role, industry)

    st.markdown("---")

    # --- CHAT HISTORY DISPLAY ---
    if st.session_state.chat_history:
        st.markdown("**Conversation:**")
        for entry in st.session_state.chat_history:
            # User question — shown in a styled info box
            st.markdown(f"**You:** {entry['question']}")

            # Assistant answer
            answer_data = entry.get("answer_data", {})
            answer_text = answer_data.get("answer", "")
            data_available = answer_data.get("data_available", True)

            if data_available:
                st.success(f"**Assistant:** {answer_text}")
            else:
                st.warning(f"**Assistant:** {answer_text}")

            # Optional chart
            if answer_data.get("chart_needed") and answer_data.get("chart"):
                chart_spec = answer_data["chart"]
                fig = render_chart(chart_spec, stats, st.session_state.uploaded_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if chart_spec.get("caption"):
                        st.caption(f"💡 {chart_spec['caption']}")

            # Follow-up suggestion
            if answer_data.get("suggested_followup"):
                st.caption(f"🔗 You could also ask: *{answer_data['suggested_followup']}*")

            st.markdown("---")

    # --- TEXT INPUT FOR NEW QUESTION ---
    st.markdown("**Ask your own question:**")

    # Use a form to handle submission cleanly
    with st.form(key=f"chat_form_{st.session_state.chat_input_key}"):
        user_question = st.text_input(
            "Type your question here...",
            placeholder="e.g. Which region had the highest sales? What was the trend over time?",
            label_visibility="collapsed"
        )
        submit_question = st.form_submit_button(
            "Submit →",
            use_container_width=False
        )

    if submit_question and user_question.strip():
        # Injection check on chat input
        injection_keywords = [
            "ignore", "reveal", "system prompt", "you are now",
            "forget", "override", "jailbreak", "drop table", "select *", "<script"
        ]
        if any(kw in user_question.lower() for kw in injection_keywords):
            st.error("⚠️ This input cannot be processed. Please ask a question about your data.")
        else:
            _process_chat_question(user_question.strip(), stats, role, industry)

    st.caption(
        f"Chat uses pre-computed statistics only. "
        f"Raw data is never re-sent. "
        f"{len(st.session_state.chat_history)} question(s) asked this session."
    )


def _process_chat_question(question, stats, role, industry):
    """
    Internal helper: calls the answer API, stores result in chat history, reruns.
    """
    with st.spinner("Answering your question..."):
        answer_data = answer_chat_question(
            question, role, industry, stats, st.session_state.api_key
        )

    # Store in chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer_data": answer_data,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    # Increment key to reset text input
    st.session_state.chat_input_key += 1
    st.rerun()


# ============================================================
# SCREEN 6 — DASHBOARD
# ============================================================

def screen_dashboard():
    show_progress("dashboard")

    df = st.session_state.uploaded_df
    role = st.session_state.user_role
    industry = st.session_state.confirmed_industry
    filename = st.session_state.uploaded_filename
    is_comparison = st.session_state.comparison_mode
    comparison_df = st.session_state.comparison_df
    comparison_filename = st.session_state.comparison_filename

    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        st.markdown(f"## 📊 {filename}")
    with col2:
        st.markdown(f"**Role:** {role}")
    with col3:
        st.markdown(f"**Industry:** {industry}")
    with col4:
        if st.button("🔄 Switch Role"):
            st.session_state.ready_to_analyse = False
            st.session_state.dashboard_result = None
            st.session_state.precomputed_stats = None
            st.session_state.role_profile = None
            st.session_state.show_narrative = False
            st.session_state.user_role = None
            st.session_state.confirmed_industry = None
            st.session_state.chat_history = []
            st.session_state.suggested_questions = []
            st.session_state.chat_input_key = 0
            st.rerun()

    st.divider()

    if st.session_state.role_profile is None:
        st.session_state.role_profile = build_role_profile(role, industry, df)
    role_profile = st.session_state.role_profile

    if st.session_state.dashboard_result is None:
        progress_placeholder = st.empty()
        try:
            progress_placeholder.info("**Step 1 of 9:** 📂 Building role profile...")
            relevant_cols = get_relevant_columns(df, role_profile)

            progress_placeholder.info("**Step 2 of 9:** 📊 Pre-computing statistics...")
            if is_comparison and comparison_df is not None:
                comp_data = precompute_comparison(df, comparison_df, role_profile, filename, comparison_filename)
                stats = comp_data["stats1"]
            else:
                comp_data = None
                stats = precompute_statistics(df, relevant_cols, role_profile, filename)

            st.session_state.precomputed_stats = stats

            progress_placeholder.info("**Step 3 of 9:** 🧠 Building analysis prompt...")
            prompt = build_analysis_prompt(stats, role_profile, industry, is_comparison=is_comparison, comparison_data=comp_data)

            result = call_openai(prompt, st.session_state.api_key, progress_placeholder)

            st.session_state.dashboard_result = result
            st.session_state.analysis_error = None
            st.session_state.analysis_count += 1
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.session_state.analysis_error = str(e)

    if st.session_state.analysis_error:
        st.error("### ⚠️ Analysis Error")
        st.markdown(f"**What went wrong:** {st.session_state.analysis_error}")
        st.markdown("**Your pre-computed statistics are still available below.**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Analysis", type="primary", use_container_width=True):
                st.session_state.dashboard_result = None
                st.session_state.analysis_error = None
                st.rerun()
        with col2:
            if st.button("← Change Role", use_container_width=True):
                st.session_state.ready_to_analyse = False
                st.session_state.dashboard_result = None
                st.session_state.analysis_error = None
                st.session_state.user_role = None
                st.rerun()
        if st.session_state.precomputed_stats:
            st.divider()
            render_precomputed_stats(st.session_state.precomputed_stats)
        return

    result = st.session_state.dashboard_result
    stats = st.session_state.precomputed_stats

    if not result:
        return

    render_executive_summary(result)
    st.divider()

    render_kpi_cards(result.get("kpi_cards", []))
    st.divider()

    render_traffic_lights(result.get("traffic_lights", []))
    if result.get("traffic_lights"):
        st.divider()

    render_anomalies(result.get("anomalies", []))
    st.divider()

    render_charts(result.get("charts", []), stats, df)
    st.divider()

    if is_comparison and result.get("comparison_insights"):
        render_comparison_insights(result["comparison_insights"])
        st.divider()

    render_recommendations(result.get("recommendations", []))
    st.divider()

    st.markdown("### 📝 Narrative Report")
    if not st.session_state.show_narrative:
        if st.button("📝 Generate Narrative Report", type="primary"):
            st.session_state.show_narrative = True
            st.rerun()
        st.caption("Generates a professional business document written in the voice appropriate for your role.")
    else:
        if result.get("narrative"):
            render_narrative(result["narrative"])
        st.divider()

    if result.get("evaluation"):
        render_evaluation(result["evaluation"])
    st.divider()

    with st.expander("📋 View Pre-Computed Statistics", expanded=False):
        render_precomputed_stats(stats)

    st.divider()

    # ---- CHAT PANEL — NATURAL LANGUAGE QUERY ----
    render_chat_panel(stats, role, industry)

    st.divider()

    # ---- FEEDBACK ----
    st.markdown("### 💬 Was this analysis useful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes — helpful", use_container_width=True):
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No — something was off", use_container_width=True):
            feedback = st.text_area(
                "What was missing or incorrect?",
                placeholder="Tell us what the analysis missed..."
            )
            if feedback:
                st.info("Feedback noted.")

    st.caption(
        f"Analysis {st.session_state.analysis_count} of 20 | "
        "Session active | Data cleared on tab close"
    )


# ============================================================
# MAIN ROUTER
# ============================================================

def main():
    screen = get_current_screen()
    if screen == "api_key":
        screen_api_key()
    elif screen == "privacy":
        screen_privacy()
    elif screen == "data_source":
        screen_data_source()
    elif screen == "data_quality":
        screen_data_quality()
    elif screen == "role_input":
        screen_role_input()
    elif screen == "dashboard":
        screen_dashboard()


if __name__ == "__main__":
    main()