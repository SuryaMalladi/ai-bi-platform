"""
AI-Powered Data Analysis & Intelligence Platform
Phase 3, Step 4 -- Data Quality Report
Author: Surya (built with Claude AI)
"""

import streamlit as st
import pandas as pd
import re

st.set_page_config(
    page_title="AI Intelligence Platform",
    page_icon="SS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "screen": "api_key",
        "api_key": "",
        "api_key_validated": False,
        "model_id": "gpt-4o-mini",
        "privacy_accepted": False,
        "dataframe": None,
        "file_name": "",
        "data_rows": 0,
        "data_columns": 0,
        "data_source_type": "",
        "analysis_count": 0,
        "quality_acknowledged": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def screen_api_key():
    st.title("Secure Configuration")
    st.caption("Enter your OpenAI API key to begin. Your key is used only for this session and is never stored anywhere.")
    st.divider()

    with st.container(border=True):
        st.markdown("**How your API key is protected**")
        st.markdown("""
- Stored only in memory -- never written to any file
- Masked in the input field at all times
- Validated with a minimal test call before proceeding
- Automatically cleared when your browser tab closes
- Never pushed to GitHub or logged anywhere
        """)

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key. Starts with sk-."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox(
            "Model",
            options=["gpt-4o-mini (Testing)", "gpt-4o (Demo)"]
        )

    model_id = "gpt-4o-mini" if "mini" in model_choice else "gpt-4o"
    st.markdown("")

    if st.button("Validate and Continue", type="primary"):
        if not api_key_input:
            st.warning("Please enter your OpenAI API key before continuing.")
        elif not api_key_input.startswith("sk-"):
            st.warning("This does not look like a valid OpenAI key. Keys begin with sk-")
        else:
            with st.spinner("Validating your API key with OpenAI..."):
                is_valid, error_message = validate_openai_key(api_key_input)
            if is_valid:
                st.session_state["api_key"] = api_key_input
                st.session_state["api_key_validated"] = True
                st.session_state["model_id"] = model_id
                st.session_state["screen"] = "privacy"
                st.rerun()
            else:
                st.error(f"API key validation failed: {error_message}")


def validate_openai_key(key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        client.models.list()
        return True, ""
    except Exception as e:
        return False, str(e)


def screen_privacy():
    st.title("Data Privacy Notice")
    st.caption("Please read the following before uploading any data to this platform.")
    st.divider()

    with st.container(border=True):
        st.markdown("**How your data is handled**")
        st.markdown("""
**Session only.** Your data is processed in this session only. When you close this tab, all data is automatically and permanently cleared.

**Never stored.** Nothing is written to any server, database, or file. Your data exists only in memory while you use the app.

**OpenAI processing.** Analysis is powered by OpenAI. Your data is sent to OpenAI's API over encrypted HTTPS only.

**Minimal data principle.** Only the data columns necessary for analysis are sent to the AI.

**Your responsibility.** Please ensure you have the legal right to process any data you upload, and that it complies with your organisation's data protection policy.
        """)
        st.caption("This platform acts as a data processor under GDPR. You, the user, act as the data controller.")

    st.markdown("")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("I Understand -- Continue", type="primary"):
            st.session_state["privacy_accepted"] = True
            st.session_state["screen"] = "data"
            st.rerun()
    with col2:
        if st.button("Change API Key", type="secondary"):
            st.session_state["screen"] = "api_key"
            st.rerun()


def screen_data_source():
    st.title("How would you like to provide your data?")
    st.caption("Upload a CSV or Excel file, or connect directly to a database.")
    st.divider()

    col_file, col_db = st.columns(2, gap="medium")

    with col_file:
        with st.container(border=True):
            st.markdown("### Upload a File")
            st.markdown("- CSV files (.csv)\n- Excel files (.xlsx, .xls)\n- Up to 50MB per file")
            st.session_state["data_source_type"] = "file"

    with col_db:
        with st.container(border=True):
            st.markdown("### Connect to Database")
            st.markdown("- MySQL / PostgreSQL\n- Microsoft SQL Server\n- Supabase / Snowflake")
            st.info("Phase 4 -- Coming Soon")
            st.button("Database Connection", type="secondary", disabled=True)

    st.divider()

    if st.session_state.get("data_source_type") == "file":
        render_file_upload_section()

    if st.session_state.get("dataframe") is not None:
        render_data_preview()
        render_continue_button()


def render_file_upload_section():
    st.markdown("**Upload your data file**")
    uploaded_file = st.file_uploader(
        "Drag and drop your file here, or click to browse",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.warning(f"This file is {uploaded_file.size/1024/1024:.1f}MB. Maximum allowed is 50MB.")
            return

        with st.spinner("Reading your file..."):
            df, parse_error = parse_uploaded_file(uploaded_file)

        if parse_error:
            st.error(f"Could not read this file: {parse_error}")
            return

        if df is None or df.empty:
            st.warning("This file appears to be empty.")
            return

        st.session_state["dataframe"] = df
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["data_rows"] = len(df)
        st.session_state["data_columns"] = len(df.columns)
        st.rerun()


def parse_uploaded_file(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif file_name.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format."

        df.columns = [str(col).strip() for col in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)


def render_data_preview():
    df = st.session_state["dataframe"]
    file_name = st.session_state["file_name"]
    rows = st.session_state["data_rows"]
    cols = st.session_state["data_columns"]

    st.success(f"Data loaded successfully -- {rows:,} rows, {cols} columns, {file_name}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    total_missing = int(df.isnull().sum().sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{rows:,}")
    m2.metric("Total Columns", cols)
    m3.metric("Numeric Columns", len(numeric_cols))
    m4.metric("Missing Values", f"{total_missing:,}" if total_missing > 0 else "None")

    st.markdown("**Data Preview -- First 5 Rows**")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

    if st.button("Upload a different file", type="secondary"):
        st.session_state["dataframe"] = None
        st.session_state["file_name"] = ""
        st.session_state["data_rows"] = 0
        st.session_state["data_columns"] = 0
        st.rerun()


def render_continue_button():
    st.divider()
    st.info("Your data is ready. Click Continue to run the Data Quality Report before analysis begins.")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Continue to Data Quality Report", type="primary"):
            st.session_state["screen"] = "quality"
            st.rerun()
    with col2:
        if st.button("Back to Privacy Notice", type="secondary"):
            st.session_state["screen"] = "privacy"
            st.rerun()


def run_data_quality_scan(df):
    report = {}
    report["row_count"] = len(df)
    report["col_count"] = len(df.columns)

    missing_counts = df.isnull().sum()
    missing_list = []
    for col in df.columns:
        count = int(missing_counts[col])
        if count > 0:
            pct = round((count / len(df)) * 100, 1)
            missing_list.append({
                "column": col,
                "count": count,
                "pct": pct,
                "severity": "HIGH" if pct > 20 else ("MEDIUM" if pct > 5 else "LOW")
            })
    report["missing"] = missing_list
    report["total_missing_cells"] = int(missing_counts.sum())

    duplicate_count = int(df.duplicated().sum())
    report["duplicates"] = {
        "count": duplicate_count,
        "pct": round((duplicate_count / len(df)) * 100, 1) if len(df) > 0 else 0
    }

    type_issues = []
    for col in df.columns:
        if df[col].dtype == object:
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            numeric_count = numeric_attempt.notna().sum()
            non_numeric_count = numeric_attempt.isna().sum() - df[col].isna().sum()
            if numeric_count > 0 and non_numeric_count > 0:
                type_issues.append({
                    "column": col,
                    "detail": f"{numeric_count} numeric values mixed with {non_numeric_count} text values"
                })
    report["type_issues"] = type_issues

    date_issues = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",
                r"\d{2}/\d{2}/\d{4}",
                r"\d{2}-\d{2}-\d{4}",
            ]
            patterns_found = set()
            for val in sample.astype(str):
                for pattern in date_patterns:
                    if re.search(pattern, val):
                        patterns_found.add(pattern)
            if len(patterns_found) > 1:
                date_issues.append({
                    "column": col,
                    "detail": f"{len(patterns_found)} different date formats detected"
                })
    report["date_issues"] = date_issues

    pii_findings = []
    pii_patterns = {
        "Email addresses": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "Phone numbers": r"(\+?\d[\d\s\-().]{7,}\d)",
        "National ID / SSN": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    }
    for col in df.columns:
        if df[col].dtype == object:
            sample_text = " ".join(df[col].dropna().head(100).astype(str))
            for pii_type, pattern in pii_patterns.items():
                matches = re.findall(pattern, sample_text)
                if matches:
                    pii_findings.append({"column": col, "type": pii_type})
                    break
    report["pii_findings"] = pii_findings

    problem_cols = set(
        [m["column"] for m in missing_list if m["severity"] == "HIGH"] +
        [t["column"] for t in type_issues] +
        [p["column"] for p in pii_findings]
    )
    report["analysis_ready_cols"] = len(df.columns) - len(problem_cols)

    score = 100
    missing_pct = (report["total_missing_cells"] / max(df.size, 1)) * 100
    score -= min(30, int(missing_pct * 3))
    score -= min(20, int(report["duplicates"]["pct"] * 2))
    score -= min(20, len(type_issues) * 5)
    score -= min(20, len(pii_findings) * 10)
    score -= min(10, len(date_issues) * 5)
    report["quality_score"] = max(0, score)
    return report


def screen_data_quality():
    st.title("Data Quality Report")
    st.caption("Your dataset has been automatically scanned. Review the findings below before analysis begins.")

    df = st.session_state["dataframe"]
    file_name = st.session_state["file_name"]

    if "quality_report" not in st.session_state:
        with st.spinner("Scanning your dataset for quality issues..."):
            st.session_state["quality_report"] = run_data_quality_scan(df)

    report = st.session_state["quality_report"]
    score = report["quality_score"]

    st.divider()

    col_info, col_score = st.columns([3, 1])
    with col_info:
        st.markdown(f"**Dataset:** {file_name}")
        st.markdown(f"**Size:** {report['row_count']:,} rows, {report['col_count']} columns")
        if score >= 80:
            st.success("This dataset is in good shape for analysis.")
        elif score >= 55:
            st.warning("Some issues were found. Review them below before proceeding.")
        else:
            st.error("Several issues detected. Consider cleaning the data first.")
    with col_score:
        st.metric("Quality Score", f"{score}/100")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing Cells", f"{report['total_missing_cells']:,}")
    m2.metric("Duplicate Rows", f"{report['duplicates']['count']:,}")
    m3.metric("Type Issues", f"{len(report['type_issues'])}")
    m4.metric("PII Detected", f"{len(report['pii_findings'])} column(s)")

    st.divider()

    if report["pii_findings"]:
        st.error("Data Protection Notice -- Personal Data Detected")
        st.markdown("This dataset appears to contain personal information. Please ensure you have the legal right to process this data before proceeding.")
        for p in report["pii_findings"]:
            st.markdown(f"- **{p['column']}** -- {p['type']}")
        st.divider()

    st.markdown("### Missing Values")
    if not report["missing"]:
        st.success("No missing values detected. All columns are complete.")
    else:
        for m in report["missing"]:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{m['column']}** -- {m['count']:,} missing ({m['pct']}%)")
                st.progress(min(m["pct"] / 100, 1.0))
            with col_b:
                if m["severity"] == "HIGH":
                    st.error(m["severity"])
                elif m["severity"] == "MEDIUM":
                    st.warning(m["severity"])
                else:
                    st.info(m["severity"])

    st.divider()

    st.markdown("### Duplicate Rows")
    dup = report["duplicates"]
    if dup["count"] == 0:
        st.success("No duplicate rows detected.")
    else:
        st.warning(f"{dup['count']:,} duplicate rows detected ({dup['pct']}% of dataset).")

    if report["type_issues"]:
        st.divider()
        st.markdown("### Data Type Issues")
        for t in report["type_issues"]:
            st.warning(f"Column '{t['column']}' has mixed data types -- {t['detail']}")

    if report["date_issues"]:
        st.divider()
        st.markdown("### Date Format Issues")
        for d in report["date_issues"]:
            st.warning(f"Column '{d['column']}' -- {d['detail']}")

    st.divider()

    ready = report["analysis_ready_cols"]
    total = report["col_count"]
    coverage_pct = int((ready / total) * 100) if total > 0 else 0
    st.markdown("### Analysis Coverage")
    st.markdown(f"**{ready} of {total} columns are analysis-ready**")
    st.progress(coverage_pct / 100)
    st.caption("Columns with high missing values, type issues, or PII are excluded from the analysis-ready count.")

    actions = []
    if report["total_missing_cells"] > 0:
        high_missing = [m for m in report["missing"] if m["severity"] == "HIGH"]
        if high_missing:
            cols_str = ", ".join([m["column"] for m in high_missing[:3]])
            actions.append(f"Consider filling or removing high-missing columns: {cols_str}")
    if dup["count"] > 0:
        actions.append(f"Review and remove {dup['count']:,} duplicate rows if they are data entry errors")
    if report["type_issues"]:
        actions.append("Standardise mixed-type columns so numeric columns contain only numbers")
    if report["pii_findings"]:
        actions.append("Confirm you have the legal right to process personal data, or remove PII columns")

    if actions:
        st.divider()
        st.markdown("### Recommended Actions")
        for a in actions:
            st.markdown(f"- {a}")

    st.divider()
    st.info("You can proceed with analysis even if issues were found. The AI will flag any limitations it encounters.")

    proceed_label = "I Acknowledge the PII Warning -- Proceed to Analysis" if report["pii_findings"] else "Proceed to Analysis"

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button(proceed_label, type="primary"):
            st.session_state["quality_acknowledged"] = True
            st.session_state["screen"] = "role_input"
            st.rerun()
    with col2:
        if st.button("Back to Data Source", type="secondary"):
            if "quality_report" in st.session_state:
                del st.session_state["quality_report"]
            st.session_state["screen"] = "data"
            st.rerun()


def screen_role_input_placeholder():
    st.title("Tell Us About Your Role")
    st.caption("The AI tailors every insight specifically to the role you enter.")
    st.divider()
    st.info("Role input, industry auto-detection, and the AI analysis engine are built in Phase 3, Step 5.")

    if st.session_state.get("dataframe") is not None:
        rows = st.session_state["data_rows"]
        cols = st.session_state["data_columns"]
        fname = st.session_state["file_name"]
        score = st.session_state.get("quality_report", {}).get("quality_score", "--")
        st.success(f"Dataset ready: {fname} -- {rows:,} rows, {cols} columns, Quality score: {score}/100")

    if st.button("Back to Data Quality Report", type="secondary"):
        st.session_state["screen"] = "quality"
        st.rerun()


def main():
    init_session_state()
    screen = st.session_state.get("screen", "api_key")

    if screen == "api_key":
        screen_api_key()
    elif screen == "privacy":
        if not st.session_state.get("api_key_validated"):
            st.session_state["screen"] = "api_key"
            st.rerun()
        else:
            screen_privacy()
    elif screen == "data":
        if not st.session_state.get("privacy_accepted"):
            st.session_state["screen"] = "privacy"
            st.rerun()
        else:
            screen_data_source()
    elif screen == "quality":
        if st.session_state.get("dataframe") is None:
            st.session_state["screen"] = "data"
            st.rerun()
        else:
            screen_data_quality()
    elif screen == "role_input":
        if not st.session_state.get("quality_acknowledged"):
            st.session_state["screen"] = "quality"
            st.rerun()
        else:
            screen_role_input_placeholder()
    else:
        st.session_state["screen"] = "api_key"
        st.rerun()


if __name__ == "__main__":
    main()
