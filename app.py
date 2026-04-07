"""
AI-Powered Data Analysis & Intelligence Platform
Phase 3, Step 4 — Data Quality Report
Author: Surya (built with Claude AI)
"""

import streamlit as st
import pandas as pd
import io
import re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# This must be the very first Streamlit command in the file.
# It sets the browser tab title, the icon, and the layout width.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — PROFESSIONAL DARK THEME
# This block injects raw CSS into the Streamlit app.
# CSS is the styling language of the web — it controls colours, fonts,
# spacing, and layout. Everything here is designed to match Power BI quality.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import professional fonts from Google */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root colour palette ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --surface2:  #181c24;
    --border:    #1e2330;
    --accent:    #3b82f6;
    --accent2:   #6366f1;
    --green:     #10b981;
    --amber:     #f59e0b;
    --red:       #ef4444;
    --text:      #e2e8f0;
    --text2:     #94a3b8;
    --text3:     #475569;
}

/* ── Global body and app background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit elements we don't need ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Main content block ── */
[data-testid="stMain"] {
    background-color: var(--bg) !important;
}

[data-testid="block-container"] {
    padding: 2rem 3rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ── All text elements ── */
p, span, div, label, h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Markdown text ── */
.stMarkdown p {
    color: var(--text2) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stPasswordInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
}

[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: var(--surface) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 28px 0 !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Metric box ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text3) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE UI COMPONENTS
# These are small building blocks we use throughout the app.
# Think of them like LEGO bricks — we define them once, use them many times.
# ─────────────────────────────────────────────────────────────────────────────

def render_topbar(step_label: str):
    """
    Renders the top navigation bar that appears on every screen.
    Shows the platform name on the left and the current step on the right.
    step_label: a short string like "Step 1 of 3 — API Key"
    """
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0 24px 0;
        border-bottom: 1px solid #1e2330;
        margin-bottom: 32px;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="
                width:32px; height:32px;
                background: linear-gradient(135deg,#3b82f6,#6366f1);
                border-radius:8px;
                display:flex; align-items:center; justify-content:center;
                font-size:14px; font-weight:700; color:white;
            ">AI</div>
            <div>
                <div style="font-size:13px; font-weight:600; color:#e2e8f0; letter-spacing:0.02em;">
                    Intelligence Platform
                </div>
                <div style="font-size:11px; color:#475569;">Phase 1 — Recruiter Demo</div>
            </div>
        </div>
        <div style="
            background:#181c24;
            border:1px solid #1e2330;
            border-radius:20px;
            padding:5px 14px;
            font-size:11px;
            color:#94a3b8;
        ">{step_label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_screen_title(icon: str, title: str, subtitle: str):
    """
    Renders a large, consistent title block at the top of each screen.
    icon: an emoji like "🔐"
    title: the main heading
    subtitle: a one-line description below the heading
    """
    st.markdown(f"""
    <div style="margin-bottom:32px;">
        <div style="font-size:32px; margin-bottom:8px;">{icon}</div>
        <h1 style="
            font-size:24px;
            font-weight:700;
            color:#e2e8f0;
            margin:0 0 8px 0;
            letter-spacing:-0.02em;
        ">{title}</h1>
        <p style="
            font-size:14px;
            color:#94a3b8;
            margin:0;
            line-height:1.6;
        ">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_card(content_html: str, border_color: str = "#1e2330"):
    """
    Renders a styled card panel — a box with a border and dark background.
    We use these throughout the app to group related information cleanly.
    content_html: the HTML content to put inside the card
    border_color: the colour of the card's border (defaults to subtle grey)
    """
    st.markdown(f"""
    <div style="
        background:#111318;
        border:1px solid {border_color};
        border-radius:12px;
        padding:24px;
        margin-bottom:16px;
    ">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


def render_success_banner(message: str):
    """
    Shows a green success confirmation banner.
    Used after data loads successfully to confirm to the user.
    """
    st.markdown(f"""
    <div style="
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 16px 0;
    ">
        <span style="font-size:16px;">✅</span>
        <span style="font-size:13px; color:#10b981; font-weight:500;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_warning_banner(message: str):
    """
    Shows an amber warning banner.
    Used for non-critical notices — things the user should be aware of.
    """
    st.markdown(f"""
    <div style="
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 16px 0;
    ">
        <span style="font-size:16px;">⚠️</span>
        <span style="font-size:13px; color:#f59e0b; font-weight:500;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_info_banner(message: str):
    """
    Shows a blue information banner.
    Used for helpful context that is not a warning or success.
    """
    st.markdown(f"""
    <div style="
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 16px 0;
    ">
        <span style="font-size:16px;">ℹ️</span>
        <span style="font-size:13px; color:#3b82f6; font-weight:500;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Streamlit reruns the entire script every time the user interacts with
# anything. Session state is the mechanism that remembers information
# between those reruns — like a short-term memory for the app.
# We initialise every variable here so the app never crashes from
# a missing key.
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    """
    Sets up all the session state variables the app needs.
    Each variable is only set if it doesn't already exist — so existing
    values are never accidentally overwritten on a rerun.
    """
    defaults = {
        # ── Navigation ──────────────────────────────────────────────────────
        # Controls which screen the user sees.
        # "api_key"   → Screen 1: Enter and validate API key
        # "privacy"   → Screen 2: Read and accept data privacy notice
        # "data"      → Screen 3: Choose data source and upload data
        # "quality"   → Screen 4: Data quality report (built in Step 4)
        "screen": "api_key",

        # ── API key ──────────────────────────────────────────────────────────
        # Stores the validated OpenAI API key in memory only.
        # This is NEVER written to any file.
        "api_key": "",
        "api_key_validated": False,

        # ── Privacy ───────────────────────────────────────────────────────────
        # Records whether the user has accepted the privacy notice.
        "privacy_accepted": False,

        # ── Data ──────────────────────────────────────────────────────────────
        # The actual loaded dataframe (table of data from the upload).
        # None means no data has been loaded yet.
        "dataframe": None,

        # The name of the file the user uploaded (e.g. "sales_data.csv").
        "file_name": "",

        # The data source type the user selected.
        # "file" → uploaded a CSV or Excel file
        # "database" → connecting to a database (Phase 4)
        "data_source_type": "",

        # How many rows and columns the data has.
        "data_rows": 0,
        "data_columns": 0,

        # ── Session tracking ──────────────────────────────────────────────────
        # Counts how many analyses have been run this session.
        # Maximum allowed: 20 (our rate limit from the PRD).
        "analysis_count": 0,

        # ── Quality report ────────────────────────────────────────────────────
        # Stores the result of the data quality scan so it does not re-run
        # every time the user interacts with the page.
        # Set to None until the scan has been run.
        "quality_acknowledged": False,
    }

    # Loop through every default and set it if not already in session state
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 1 — API KEY ENTRY
# The very first thing the user sees.
# They must enter a valid OpenAI API key before anything else happens.
# The key is validated with a real (but minimal) API call.
# ─────────────────────────────────────────────────────────────────────────────

def screen_api_key():
    """
    Renders the secure API key entry screen.
    On successful validation, navigates to the privacy notice screen.
    """
    render_topbar("Step 1 of 3 — Secure Configuration")
    render_screen_title(
        "🔐",
        "Secure Configuration",
        "Enter your OpenAI API key to begin. Your key is used only for this session and is never stored anywhere."
    )

    # ── Security notice card ──────────────────────────────────────────────────
    render_card("""
        <div style="font-size:12px; color:#94a3b8; line-height:1.8;">
            <div style="font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">
                🛡️ How your API key is protected
            </div>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <div>✓ &nbsp;Stored only in memory — never written to any file</div>
                <div>✓ &nbsp;Masked in the input field at all times</div>
                <div>✓ &nbsp;Validated with a minimal test call before proceeding</div>
                <div>✓ &nbsp;Automatically cleared when your browser tab closes</div>
                <div>✓ &nbsp;Never pushed to GitHub or logged anywhere</div>
            </div>
        </div>
    """)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── API key input ─────────────────────────────────────────────────────────
    # type="password" masks the key as dots while typing
    api_key_input = st.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key. Starts with 'sk-'. Find it at platform.openai.com"
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Model selector — user picks testing (cheap) or demo (full power) model
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox(
            "Model",
            options=["gpt-4o-mini (Testing)", "gpt-4o (Demo)"],
            help="Use gpt-4o-mini while building to keep costs low. Switch to gpt-4o for recruiter demos."
        )

    # Store the actual model string based on the user's choice
    # We strip the label text and keep only the model identifier
    model_id = "gpt-4o-mini" if "mini" in model_choice else "gpt-4o"

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Validate button ───────────────────────────────────────────────────────
    if st.button("Validate and Continue →", type="primary"):
        if not api_key_input:
            # User clicked the button without entering anything
            render_warning_banner("Please enter your OpenAI API key before continuing.")
        elif not api_key_input.startswith("sk-"):
            # Basic format check — all OpenAI keys start with sk-
            render_warning_banner("This does not look like a valid OpenAI key. Keys begin with 'sk-'.")
        else:
            # ── Validation: test the key with a real API call ──────────────
            # We use a spinner to show the user something is happening
            with st.spinner("Validating your API key with OpenAI..."):
                is_valid, error_message = validate_openai_key(api_key_input)

            if is_valid:
                # Store validated key and model in session state
                st.session_state["api_key"] = api_key_input
                st.session_state["api_key_validated"] = True
                st.session_state["model_id"] = model_id
                # Navigate to the next screen
                st.session_state["screen"] = "privacy"
                st.rerun()  # Rerun immediately to show the next screen
            else:
                render_warning_banner(f"API key validation failed: {error_message}")


def validate_openai_key(key: str) -> tuple[bool, str]:
    """
    Tests whether an OpenAI API key is valid by making a minimal API call.
    We list available models — this uses essentially zero tokens and costs nothing.

    Returns: (True, "") if valid
             (False, "error message") if invalid
    """
    try:
        # We import the OpenAI library here (installed via requirements.txt)
        from openai import OpenAI, AuthenticationError, APIConnectionError

        # Create an OpenAI client object using the provided key
        client = OpenAI(api_key=key)

        # Ask for the list of models — cheapest possible API call
        # If the key is invalid, OpenAI will return an authentication error
        client.models.list()

        # If we reach this line without an exception, the key is valid
        return True, ""

    except AuthenticationError:
        # This specific error means the key was rejected by OpenAI
        return False, "This API key was not recognised by OpenAI. Please check and try again."

    except APIConnectionError:
        # This means we could not reach OpenAI at all (network issue)
        return False, "Could not connect to OpenAI. Please check your internet connection."

    except Exception as e:
        # Catch anything else unexpected
        return False, f"Unexpected error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2 — DATA PRIVACY NOTICE
# Required before any data processing begins.
# GDPR compliance — user must explicitly acknowledge the notice.
# ─────────────────────────────────────────────────────────────────────────────

def screen_privacy():
    """
    Renders the data privacy notice screen.
    User must click "I Understand" to proceed. They cannot skip this.
    """
    render_topbar("Step 2 of 3 — Data Privacy Notice")
    render_screen_title(
        "📋",
        "Data Privacy Notice",
        "Please read the following before uploading any data to this platform."
    )

    # ── Privacy notice content ────────────────────────────────────────────────
    render_card("""
        <div style="font-size:13px; color:#94a3b8; line-height:1.9;">

            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:16px;">
                How your data is handled
            </div>

            <div style="display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; gap:12px; align-items:flex-start;">
                    <span style="color:#10b981; flex-shrink:0; margin-top:2px;">✓</span>
                    <span><strong style="color:#e2e8f0;">Session only.</strong>
                    Your data is processed in this session only. When you close this tab,
                    all data is automatically and permanently cleared.</span>
                </div>
                <div style="display:flex; gap:12px; align-items:flex-start;">
                    <span style="color:#10b981; flex-shrink:0; margin-top:2px;">✓</span>
                    <span><strong style="color:#e2e8f0;">Never stored.</strong>
                    Nothing is written to any server, database, or file.
                    Your data exists only in memory while you use the app.</span>
                </div>
                <div style="display:flex; gap:12px; align-items:flex-start;">
                    <span style="color:#10b981; flex-shrink:0; margin-top:2px;">✓</span>
                    <span><strong style="color:#e2e8f0;">OpenAI processing.</strong>
                    Analysis is powered by OpenAI. Your data is sent to OpenAI's API
                    over encrypted HTTPS only. OpenAI processes it under their API terms of service.</span>
                </div>
                <div style="display:flex; gap:12px; align-items:flex-start;">
                    <span style="color:#10b981; flex-shrink:0; margin-top:2px;">✓</span>
                    <span><strong style="color:#e2e8f0;">Minimal data principle.</strong>
                    Only the data columns necessary for analysis are sent to the AI.
                    Unnecessary metadata is stripped before transmission.</span>
                </div>
                <div style="display:flex; gap:12px; align-items:flex-start;">
                    <span style="color:#10b981; flex-shrink:0; margin-top:2px;">✓</span>
                    <span><strong style="color:#e2e8f0;">Your responsibility.</strong>
                    Please ensure you have the legal right to process any data you upload,
                    and that it complies with your organisation's data protection policy.</span>
                </div>
            </div>

            <div style="
                margin-top:20px;
                padding-top:16px;
                border-top:1px solid #1e2330;
                font-size:12px;
                color:#475569;
            ">
                This platform acts as a data processor under GDPR.
                You, the user, act as the data controller.
                OpenAI is named as a data subprocessor.
            </div>
        </div>
    """)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Acceptance button ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("I Understand — Continue →", type="primary"):
            st.session_state["privacy_accepted"] = True
            st.session_state["screen"] = "data"
            st.rerun()

    with col2:
        # Allow user to go back and change their API key if needed
        if st.button("← Change API Key", type="secondary"):
            st.session_state["screen"] = "api_key"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 3 — DATA SOURCE SELECTION  ← NEW THIS STEP
# The user chooses how they want to provide their data.
# Option A: Upload a CSV or Excel file
# Option B: Connect to a database (placeholder — full build in Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def screen_data_source():
    """
    Renders the data source selection screen.
    Handles file upload, validation, parsing, and preview.
    On success, stores the dataframe in session state and shows a preview.
    """
    render_topbar("Step 3 of 3 — Data Source")
    render_screen_title(
        "📂",
        "How would you like to provide your data?",
        "Upload a CSV or Excel file, or connect directly to a database."
    )

    # ── Two source option cards ───────────────────────────────────────────────
    col_file, col_db = st.columns(2, gap="medium")

    with col_file:
        # File upload option card
        is_file_selected = st.session_state.get("data_source_type") == "file"
        border = "#3b82f6" if is_file_selected else "#1e2330"
        st.markdown(f"""
        <div style="
            background:#111318;
            border:1px solid {border};
            border-radius:12px;
            padding:22px;
            margin-bottom:8px;
            cursor:pointer;
        ">
            <div style="font-size:24px; margin-bottom:10px;">📄</div>
            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:6px;">
                Upload a File
            </div>
            <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                CSV files (.csv)<br>
                Excel files (.xlsx, .xls)<br>
                Up to 50MB per file
            </div>
            {"<div style='margin-top:12px; font-size:11px; color:#3b82f6; font-weight:600;'>● SELECTED</div>" if is_file_selected else ""}
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select File Upload", type="primary" if not is_file_selected else "secondary"):
            st.session_state["data_source_type"] = "file"
            st.rerun()

    with col_db:
        # Database option card — marked as Coming Soon
        st.markdown("""
        <div style="
            background:#111318;
            border:1px solid #1e2330;
            border-radius:12px;
            padding:22px;
            margin-bottom:8px;
            opacity:0.6;
        ">
            <div style="font-size:24px; margin-bottom:10px;">🗄️</div>
            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:6px;">
                Connect to Database
            </div>
            <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                MySQL · PostgreSQL<br>
                Microsoft SQL Server<br>
                Supabase · Snowflake
            </div>
            <div style="
                margin-top:12px;
                display:inline-block;
                background:rgba(99,102,241,0.15);
                border:1px solid rgba(99,102,241,0.3);
                border-radius:10px;
                padding:3px 10px;
                font-size:10px;
                color:#818cf8;
                font-weight:600;
                letter-spacing:0.05em;
            ">PHASE 4 — COMING SOON</div>
        </div>
        """, unsafe_allow_html=True)

        # Disabled button for database — not clickable yet
        st.button("Database Connection (Phase 4)", type="secondary", disabled=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── File upload section ───────────────────────────────────────────────────
    # Only shown if the user has selected "file" as their source type
    if st.session_state.get("data_source_type") == "file":
        render_file_upload_section()

    # ── Already loaded — show existing data ──────────────────────────────────
    # If data was already successfully loaded, show it again without requiring
    # a re-upload (in case the user navigated back to this screen)
    if st.session_state.get("dataframe") is not None and \
       st.session_state.get("data_source_type") == "file":
        render_data_preview()
        render_continue_button()


def render_file_upload_section():
    """
    Renders the actual file upload widget and handles the upload process.
    Validates the file, parses it into a dataframe, and stores it in session state.
    """
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">
        Upload your data file
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader widget ──────────────────────────────────────────────────
    # accept_multiple_files=False means only one file at a time
    # type= restricts which file types can be selected
    uploaded_file = st.file_uploader(
        label="Drag and drop your file here, or click to browse",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="Supported formats: CSV (.csv), Excel (.xlsx, .xls). Maximum file size: 50MB."
    )

    # ── Process the uploaded file ─────────────────────────────────────────────
    if uploaded_file is not None:

        # ── File size check ───────────────────────────────────────────────────
        # 50MB = 50 * 1024 * 1024 bytes
        max_size_bytes = 50 * 1024 * 1024
        file_size = uploaded_file.size

        if file_size > max_size_bytes:
            render_warning_banner(
                f"This file is {file_size / 1024 / 1024:.1f}MB. "
                f"The maximum allowed size is 50MB. Please try a smaller file."
            )
            return  # Stop processing — file is too large

        # ── Parse the file ────────────────────────────────────────────────────
        with st.spinner("Reading your file..."):
            df, parse_error = parse_uploaded_file(uploaded_file)

        if parse_error:
            # Something went wrong reading the file
            render_warning_banner(f"Could not read this file: {parse_error}")
            return

        if df is None or df.empty:
            render_warning_banner(
                "This file appears to be empty. Please upload a file containing data."
            )
            return

        # ── Store in session state ────────────────────────────────────────────
        # This is the key moment — the dataframe is now in memory
        # and available to every other part of the app
        st.session_state["dataframe"] = df
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["data_rows"] = len(df)
        st.session_state["data_columns"] = len(df.columns)

        # Force a rerun so the preview section renders immediately
        st.rerun()


def parse_uploaded_file(uploaded_file) -> tuple:
    """
    Reads the uploaded file and converts it into a pandas DataFrame.
    A DataFrame is like a spreadsheet in Python — rows and columns of data.

    Handles both CSV and Excel formats.
    Returns: (dataframe, None) on success
             (None, "error message") on failure
    """
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            # ── CSV file ──────────────────────────────────────────────────────
            # Try standard UTF-8 encoding first
            # If that fails, try latin-1 (common in European datasets)
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                # Reset file position and try again with different encoding
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")

        elif file_name.endswith((".xlsx", ".xls")):
            # ── Excel file ────────────────────────────────────────────────────
            # openpyxl engine handles modern .xlsx files
            # xlrd engine handles legacy .xls files
            try:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)

        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."

        # ── Basic cleaning ────────────────────────────────────────────────────
        # Strip extra whitespace from column names
        # (common issue with exported spreadsheets)
        df.columns = [str(col).strip() for col in df.columns]

        # Remove completely empty rows
        df = df.dropna(how="all")

        # Reset the row index after dropping rows
        df = df.reset_index(drop=True)

        return df, None

    except Exception as e:
        return None, str(e)


def render_data_preview():
    """
    Renders a professional preview of the loaded data.
    Shows key stats (rows, columns, file name) and the first 5 rows of data.
    This gives the user confidence that their data loaded correctly.
    """
    df = st.session_state["dataframe"]
    file_name = st.session_state["file_name"]
    rows = st.session_state["data_rows"]
    cols = st.session_state["data_columns"]

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Success confirmation ───────────────────────────────────────────────────
    render_success_banner(
        f"Data loaded successfully — {rows:,} rows · {cols} columns · {file_name}"
    )

    # ── Key metrics row ───────────────────────────────────────────────────────
    # Count how many columns are numeric (numbers) vs text (categories)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    # Try to detect date columns that were read as text
    for col in text_cols:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            date_cols.append(col)
        except Exception:
            pass

    # Count missing values across the entire dataframe
    total_missing = df.isnull().sum().sum()

    # Display as four metric tiles across the top
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Rows", f"{rows:,}")
    with m2:
        st.metric("Total Columns", cols)
    with m3:
        st.metric("Numeric Columns", len(numeric_cols))
    with m4:
        missing_label = f"{total_missing:,}" if total_missing > 0 else "None"
        st.metric("Missing Values", missing_label)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Column type summary ───────────────────────────────────────────────────
    render_card(f"""
        <div style="font-size:12px; color:#94a3b8; margin-bottom:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">
            Column Summary
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            {"".join([
                f'<span style="background:#181c24;border:1px solid #1e2330;border-radius:6px;padding:4px 10px;font-size:12px;color:#94a3b8;">'
                f'<span style="color:#3b82f6;">📊</span> {col}</span>'
                for col in numeric_cols
            ])}
            {"".join([
                f'<span style="background:#181c24;border:1px solid #1e2330;border-radius:6px;padding:4px 10px;font-size:12px;color:#94a3b8;">'
                f'<span style="color:#10b981;">🔤</span> {col}</span>'
                for col in text_cols if col not in date_cols
            ])}
            {"".join([
                f'<span style="background:#181c24;border:1px solid #1e2330;border-radius:6px;padding:4px 10px;font-size:12px;color:#94a3b8;">'
                f'<span style="color:#f59e0b;">📅</span> {col}</span>'
                for col in date_cols
            ])}
        </div>
        <div style="margin-top:14px; font-size:11px; color:#475569;">
            📊 Numeric &nbsp;&nbsp; 🔤 Text / Category &nbsp;&nbsp; 📅 Date
        </div>
    """)

    # ── Data preview table ────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:12px; margin-top:4px;">
        Data Preview — First 5 Rows
    </div>
    """, unsafe_allow_html=True)

    # Show only the first 5 rows — enough to confirm data loaded correctly
    # use_container_width=True makes the table fill the full width of the screen
    st.dataframe(
        df.head(5),
        use_container_width=True,
        hide_index=True
    )

    # ── Replace file option ───────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("← Upload a different file", type="secondary"):
        # Clear the stored data so the user can start fresh
        st.session_state["dataframe"] = None
        st.session_state["file_name"] = ""
        st.session_state["data_rows"] = 0
        st.session_state["data_columns"] = 0
        st.rerun()


def render_continue_button():
    """
    Shows the final Continue button once data is loaded and ready.
    This will navigate to Screen 4 — the Data Quality Report (Step 4).
    """
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    render_info_banner(
        "Your data is ready. Click Continue to run the Data Quality Report before analysis begins."
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Continue to Data Quality Report →", type="primary"):
            # Navigate to Screen 4 — built in Step 4
            st.session_state["screen"] = "quality"
            st.rerun()

    with col2:
        # Allow going back to change the API key or privacy acceptance
        if st.button("← Back to Privacy Notice", type="secondary"):
            st.session_state["screen"] = "privacy"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY ENGINE
# This is the brain behind the Data Quality Report.
# It scans the dataframe and returns a structured dictionary of findings.
# It never modifies the data — it only reads and reports.
# ─────────────────────────────────────────────────────────────────────────────

def run_data_quality_scan(df: pd.DataFrame) -> dict:
    """
    Scans a dataframe and returns a complete quality report as a dictionary.
    Every finding is computed here — the screen function just displays results.

    Returns a dictionary with these keys:
      row_count, col_count, missing, duplicates, type_issues,
      date_issues, pii_findings, analysis_ready_cols, quality_score
    """

    report = {}

    # ── 1. Basic counts ───────────────────────────────────────────────────────
    report["row_count"] = len(df)
    report["col_count"] = len(df.columns)

    # ── 2. Missing values ─────────────────────────────────────────────────────
    # For every column, count how many cells are empty (NaN = Not a Number,
    # which is Python's way of representing a blank/missing cell)
    missing_counts = df.isnull().sum()

    # Build a list of only the columns that actually have missing values
    # Each entry is a dict with the column name, count, and percentage
    missing_list = []
    for col in df.columns:
        count = int(missing_counts[col])
        if count > 0:
            pct = round((count / len(df)) * 100, 1)
            missing_list.append({
                "column": col,
                "count": count,
                "pct": pct,
                # Severity: high if more than 20% missing, medium if 5-20%, low otherwise
                "severity": "HIGH" if pct > 20 else ("MEDIUM" if pct > 5 else "LOW")
            })

    report["missing"] = missing_list
    report["total_missing_cells"] = int(missing_counts.sum())

    # ── 3. Duplicate rows ─────────────────────────────────────────────────────
    # A duplicate row is one where every single column value is identical
    # to another row already in the dataset
    duplicate_count = int(df.duplicated().sum())
    report["duplicates"] = {
        "count": duplicate_count,
        "pct": round((duplicate_count / len(df)) * 100, 1) if len(df) > 0 else 0
    }

    # ── 4. Data type issues ───────────────────────────────────────────────────
    # A type issue is when a column contains mixed data — e.g. some rows have
    # numbers and other rows have text in the same column.
    # This confuses the AI and must be flagged.
    type_issues = []
    for col in df.columns:
        # Only check columns that pandas read as "object" (mixed/text type)
        if df[col].dtype == object:
            # Try to convert the column to numbers
            # errors="coerce" means failed conversions become NaN instead of crashing
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            numeric_count = numeric_attempt.notna().sum()
            non_numeric_count = numeric_attempt.isna().sum() - df[col].isna().sum()

            # If some rows converted to numbers and some did not — it is mixed
            if numeric_count > 0 and non_numeric_count > 0:
                type_issues.append({
                    "column": col,
                    "detail": f"{numeric_count} numeric values mixed with {non_numeric_count} text values"
                })

    report["type_issues"] = type_issues

    # ── 5. Date format inconsistencies ───────────────────────────────────────
    # Checks columns that look like they contain dates but have
    # inconsistent formatting (e.g. some rows say "2024-01-15" and
    # others say "15/01/2024" in the same column)
    date_issues = []
    for col in df.columns:
        if df[col].dtype == object:
            # Sample up to 50 non-null values for speed
            sample = df[col].dropna().head(50)

            # Common date patterns to detect
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",       # 2024-01-15
                r"\d{2}/\d{2}/\d{4}",        # 15/01/2024 or 01/15/2024
                r"\d{2}-\d{2}-\d{4}",        # 15-01-2024
                r"\d{1,2}\s\w+\s\d{4}",      # 15 January 2024
            ]

            # Count how many distinct patterns appear in this column
            patterns_found = set()
            for val in sample.astype(str):
                for pattern in date_patterns:
                    if re.search(pattern, val):
                        patterns_found.add(pattern)

            # If more than one date pattern exists in the same column — flag it
            if len(patterns_found) > 1:
                date_issues.append({
                    "column": col,
                    "detail": f"{len(patterns_found)} different date formats detected"
                })

    report["date_issues"] = date_issues

    # ── 6. PII Detection ──────────────────────────────────────────────────────
    # PII = Personally Identifiable Information
    # We scan every text column for patterns that look like personal data.
    # This is a GDPR requirement — we must warn the user before processing.
    pii_findings = []

    # Define regex patterns for each PII category
    # A regex (regular expression) is a pattern-matching formula
    pii_patterns = {
        "Email addresses":       r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "Phone numbers":         r"(\+?\d[\d\s\-().]{7,}\d)",
        "Credit card numbers":   r"\b(?:\d[ -]?){13,16}\b",
        "National ID / SSN":     r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "UK National Insurance": r"\b[A-Z]{2}\d{6}[A-Z]\b",
    }

    # Check every text column against every PII pattern
    for col in df.columns:
        if df[col].dtype == object:
            # Convert column values to strings and join into one block of text
            # We sample up to 100 rows for speed — enough to detect PII reliably
            sample_text = " ".join(df[col].dropna().head(100).astype(str))

            for pii_type, pattern in pii_patterns.items():
                matches = re.findall(pattern, sample_text)
                if matches:
                    pii_findings.append({
                        "column": col,
                        "type": pii_type,
                        "matches_found": min(len(matches), 3)  # Show max 3 examples
                    })
                    break  # One PII type per column is enough to flag it

    report["pii_findings"] = pii_findings

    # ── 7. Analysis-ready column count ───────────────────────────────────────
    # A column is "analysis-ready" if it has:
    # - Less than 20% missing values
    # - No type mixing issues
    # - Not flagged as PII
    problem_cols = set(
        [m["column"] for m in missing_list if m["severity"] == "HIGH"] +
        [t["column"] for t in type_issues] +
        [p["column"] for p in pii_findings]
    )
    analysis_ready = len(df.columns) - len(problem_cols)
    report["analysis_ready_cols"] = analysis_ready

    # ── 8. Overall quality score ──────────────────────────────────────────────
    # A simple 0-100 score summarising dataset health.
    # We deduct points for each category of issue found.
    score = 100

    # Deduct for missing values (up to 30 points)
    missing_pct = (report["total_missing_cells"] / max(df.size, 1)) * 100
    score -= min(30, int(missing_pct * 3))

    # Deduct for duplicates (up to 20 points)
    score -= min(20, int(report["duplicates"]["pct"] * 2))

    # Deduct for type issues (5 points each, up to 20)
    score -= min(20, len(type_issues) * 5)

    # Deduct for PII findings (10 points each, up to 20)
    score -= min(20, len(pii_findings) * 10)

    # Deduct for date issues (5 points each, up to 10)
    score -= min(10, len(date_issues) * 5)

    report["quality_score"] = max(0, score)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 4 — DATA QUALITY REPORT
# Displays the results of the quality scan in a professional, readable format.
# The user must click "Proceed" to continue — they cannot skip this screen.
# ─────────────────────────────────────────────────────────────────────────────

def screen_data_quality():
    """
    Renders the full Data Quality Report screen.
    Runs the quality scan once and stores results in session state
    so the scan does not re-run every time the user interacts with the page.
    """
    render_topbar("Step 4 — Data Quality Report")
    render_screen_title(
        "🔍",
        "Data Quality Report",
        "Your dataset has been automatically scanned. Review the findings below before analysis begins."
    )

    df = st.session_state["dataframe"]
    file_name = st.session_state["file_name"]

    # ── Run scan once and cache in session state ───────────────────────────────
    # We only run the scan if we have not run it already for this dataset.
    # "quality_report" not in session state means it has never been run.
    if "quality_report" not in st.session_state:
        with st.spinner("Scanning your dataset for quality issues..."):
            st.session_state["quality_report"] = run_data_quality_scan(df)

    report = st.session_state["quality_report"]

    # ── Quality score banner ──────────────────────────────────────────────────
    score = report["quality_score"]

    # Decide colour and label based on score
    if score >= 80:
        score_color = "#10b981"   # Green
        score_label = "Good"
        score_desc  = "This dataset is in good shape for analysis."
    elif score >= 55:
        score_color = "#f59e0b"   # Amber
        score_label = "Fair"
        score_desc  = "Some issues were found. Review them below before proceeding."
    else:
        score_color = "#ef4444"   # Red
        score_label = "Poor"
        score_desc  = "Several issues detected. Consider cleaning the data first."

    st.markdown(f"""
    <div style="
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 12px;
        padding: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    ">
        <div>
            <div style="font-size:11px; font-weight:600; color:#475569;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Dataset Quality Score
            </div>
            <div style="font-size:13px; color:#94a3b8; margin-bottom:4px;">
                {file_name} &nbsp;·&nbsp; {report['row_count']:,} rows
                &nbsp;·&nbsp; {report['col_count']} columns
            </div>
            <div style="font-size:13px; color:#94a3b8;">{score_desc}</div>
        </div>
        <div style="text-align:center; flex-shrink:0; margin-left:32px;">
            <div style="font-size:48px; font-weight:700; color:{score_color};
                        font-family:'DM Mono',monospace; line-height:1;">
                {score}
            </div>
            <div style="font-size:12px; color:{score_color}; font-weight:600;
                        margin-top:4px;">
                {score_label}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Four summary metric tiles ─────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Missing Cells",
            f"{report['total_missing_cells']:,}",
            help="Total number of empty cells across all columns"
        )
    with m2:
        st.metric(
            "Duplicate Rows",
            f"{report['duplicates']['count']:,}",
            help="Rows where every column value is identical to another row"
        )
    with m3:
        st.metric(
            "Type Issues",
            f"{len(report['type_issues'])}",
            help="Columns where numbers and text are mixed together"
        )
    with m4:
        st.metric(
            "PII Detected",
            f"{len(report['pii_findings'])} column(s)",
            help="Columns containing personal data (emails, phone numbers, IDs)"
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── PII Warning — shown first if detected, must be acknowledged ───────────
    if report["pii_findings"]:
        st.markdown(f"""
        <div style="
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
        ">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="font-size:18px;">⚠️</span>
                <span style="font-size:14px; font-weight:600; color:#ef4444;">
                    Data Protection Notice — Personal Data Detected
                </span>
            </div>
            <div style="font-size:13px; color:#94a3b8; line-height:1.7; margin-bottom:16px;">
                This dataset appears to contain personal information in the following columns.
                Please ensure you have the legal right to process this data and that it
                complies with your organisation's data protection policy before proceeding.
            </div>
            <div style="display:flex; flex-direction:column; gap:8px;">
                {"".join([
                    f'<div style="display:flex; align-items:center; gap:10px; '
                    f'background:#181c24; border:1px solid #2a1f1f; border-radius:8px; '
                    f'padding:10px 14px;">'
                    f'<span style="font-size:14px;">🔴</span>'
                    f'<span style="font-size:13px; color:#e2e8f0; font-weight:500;">'
                    f'{p["column"]}</span>'
                    f'<span style="font-size:12px; color:#94a3b8;">— {p["type"]}</span>'
                    f'</div>'
                    for p in report["pii_findings"]
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Missing Values section ────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#e2e8f0;
                margin-bottom:12px; margin-top:8px;">
        Missing Values
    </div>
    """, unsafe_allow_html=True)

    if not report["missing"]:
        render_success_banner("No missing values detected. All columns are complete.")
    else:
        # Build a visual row for each column with missing data
        for m in report["missing"]:
            sev_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#94a3b8"}[m["severity"]]
            bar_width = min(100, int(m["pct"]))

            st.markdown(f"""
            <div style="
                background:#111318; border:1px solid #1e2330;
                border-radius:10px; padding:14px 18px; margin-bottom:8px;
            ">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:8px;">
                    <span style="font-size:13px; color:#e2e8f0; font-weight:500;">
                        {m["column"]}
                    </span>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:12px; color:{sev_color}; font-weight:600;">
                            {m["severity"]}
                        </span>
                        <span style="font-family:'DM Mono',monospace; font-size:12px;
                                    color:#94a3b8;">
                            {m["count"]:,} missing ({m["pct"]}%)
                        </span>
                    </div>
                </div>
                <div style="background:#1e2330; border-radius:4px; height:4px; overflow:hidden;">
                    <div style="width:{bar_width}%; height:100%;
                                background:{sev_color}; border-radius:4px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Duplicate rows section ────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#e2e8f0;
                margin-bottom:12px; margin-top:16px;">
        Duplicate Rows
    </div>
    """, unsafe_allow_html=True)

    dup = report["duplicates"]
    if dup["count"] == 0:
        render_success_banner("No duplicate rows detected.")
    else:
        render_warning_banner(
            f"{dup['count']:,} duplicate rows detected ({dup['pct']}% of dataset). "
            f"These will be included in analysis but may skew averages and totals."
        )

    # ── Data type issues section ──────────────────────────────────────────────
    if report["type_issues"]:
        st.markdown("""
        <div style="font-size:13px; font-weight:600; color:#e2e8f0;
                    margin-bottom:12px; margin-top:16px;">
            Data Type Issues
        </div>
        """, unsafe_allow_html=True)

        for t in report["type_issues"]:
            render_warning_banner(
                f"Column '{t['column']}' has mixed data types — {t['detail']}. "
                f"This column may produce unreliable numeric analysis."
            )

    # ── Date format issues section ────────────────────────────────────────────
    if report["date_issues"]:
        st.markdown("""
        <div style="font-size:13px; font-weight:600; color:#e2e8f0;
                    margin-bottom:12px; margin-top:16px;">
            Date Format Issues
        </div>
        """, unsafe_allow_html=True)

        for d in report["date_issues"]:
            render_warning_banner(
                f"Column '{d['column']}' — {d['detail']}. "
                f"Time-series analysis may be affected."
            )

    # ── Analysis coverage summary ─────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    ready = report["analysis_ready_cols"]
    total = report["col_count"]
    coverage_pct = int((ready / total) * 100) if total > 0 else 0

    render_card(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    margin-bottom:14px;">
            <div style="font-size:13px; font-weight:600; color:#e2e8f0;">
                Analysis Coverage
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:13px; color:#3b82f6;">
                {ready} of {total} columns analysis-ready
            </div>
        </div>
        <div style="background:#1e2330; border-radius:6px; height:6px; overflow:hidden;">
            <div style="width:{coverage_pct}%; height:100%;
                        background:linear-gradient(90deg,#3b82f6,#6366f1);
                        border-radius:6px;">
            </div>
        </div>
        <div style="font-size:12px; color:#475569; margin-top:10px;">
            Columns with high missing values, type issues, or PII detections
            are excluded from the analysis-ready count.
        </div>
    """)

    # ── Recommended actions ───────────────────────────────────────────────────
    # Build a list of recommendations based on what was found
    actions = []
    if report["total_missing_cells"] > 0:
        high_missing = [m for m in report["missing"] if m["severity"] == "HIGH"]
        if high_missing:
            cols_str = ", ".join([m["column"] for m in high_missing[:3]])
            actions.append(f"Consider filling or removing high-missing columns: {cols_str}")
    if dup["count"] > 0:
        actions.append(
            f"Review and remove {dup['count']:,} duplicate rows before analysis "
            f"if they are data entry errors"
        )
    if report["type_issues"]:
        actions.append(
            "Standardise mixed-type columns so numeric columns contain only numbers"
        )
    if report["pii_findings"]:
        actions.append(
            "Confirm you have the legal right to process personal data, "
            "or remove PII columns before uploading"
        )

    if actions:
        st.markdown("""
        <div style="font-size:13px; font-weight:600; color:#e2e8f0;
                    margin-bottom:12px; margin-top:8px;">
            Recommended Actions
        </div>
        """, unsafe_allow_html=True)

        actions_html = "".join([
            f'<div style="display:flex; gap:10px; align-items:flex-start; '
            f'margin-bottom:8px;">'
            f'<span style="color:#f59e0b; flex-shrink:0; margin-top:1px;">→</span>'
            f'<span style="font-size:13px; color:#94a3b8; line-height:1.6;">{a}</span>'
            f'</div>'
            for a in actions
        ])
        render_card(actions_html)

    # ── Navigation buttons ────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Decide the proceed button label based on PII presence
    if report["pii_findings"]:
        proceed_label = "I Acknowledge the PII Warning — Proceed to Analysis →"
    else:
        proceed_label = "Proceed to Analysis →"

    render_info_banner(
        "You can proceed with analysis even if issues were found. "
        "The AI will work with the data as uploaded and flag any limitations it encounters."
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button(proceed_label, type="primary"):
            # Mark quality report as acknowledged and move to the next screen
            st.session_state["quality_acknowledged"] = True
            st.session_state["screen"] = "role_input"
            st.rerun()

    with col2:
        if st.button("← Back to Data Source", type="secondary"):
            # Clear the cached quality report so it re-runs if new data is uploaded
            if "quality_report" in st.session_state:
                del st.session_state["quality_report"]
            st.session_state["screen"] = "data"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 5 PLACEHOLDER — ROLE INPUT & INDUSTRY DETECTION
# Built in Phase 3, Step 5. Navigation wired up now so flow works end-to-end.
# ─────────────────────────────────────────────────────────────────────────────

def screen_role_input_placeholder():
    """
    Temporary placeholder for the Role Input and Industry Detection screen.
    Will be fully built in Phase 3, Step 5.
    """
    render_topbar("Step 5 — Role & Industry")
    render_screen_title(
        "👤",
        "Tell Us About Your Role",
        "The AI tailors every insight specifically to the role you enter."
    )

    render_card("""
        <div style="text-align:center; padding:40px 0;">
            <div style="font-size:48px; margin-bottom:16px;">🚧</div>
            <div style="font-size:16px; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
                Coming in Phase 3, Step 5
            </div>
            <div style="font-size:13px; color:#94a3b8; max-width:420px;
                        margin:0 auto; line-height:1.6;">
                Role input, industry auto-detection, and the AI analysis engine
                are built in the next step.
            </div>
        </div>
    """)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.session_state.get("dataframe") is not None:
        rows = st.session_state["data_rows"]
        cols = st.session_state["data_columns"]
        fname = st.session_state["file_name"]
        score = st.session_state.get("quality_report", {}).get("quality_score", "—")
        render_success_banner(
            f"Dataset ready: {fname} — {rows:,} rows · {cols} columns · "
            f"Quality score: {score}/100"
        )

    if st.button("← Back to Data Quality Report", type="secondary"):
        st.session_state["screen"] = "quality"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP ROUTER
# This is the central control function.
# It checks which screen should be shown and calls the right function.
# Think of it like a traffic controller at a junction.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    The main entry point for the app.
    Initialises session state and routes the user to the correct screen.
    """
    # Always initialise session state first
    init_session_state()

    # Read which screen the user should currently see
    current_screen = st.session_state.get("screen", "api_key")

    # Route to the correct screen function
    if current_screen == "api_key":
        screen_api_key()

    elif current_screen == "privacy":
        # Safety check: cannot reach privacy screen without a valid key
        if not st.session_state.get("api_key_validated"):
            st.session_state["screen"] = "api_key"
            st.rerun()
        else:
            screen_privacy()

    elif current_screen == "data":
        # Safety check: cannot reach data screen without accepting privacy
        if not st.session_state.get("privacy_accepted"):
            st.session_state["screen"] = "privacy"
            st.rerun()
        else:
            screen_data_source()

    elif current_screen == "quality":
        # Safety check: cannot reach quality screen without loaded data
        if st.session_state.get("dataframe") is None:
            st.session_state["screen"] = "data"
            st.rerun()
        else:
            screen_data_quality()

    elif current_screen == "role_input":
        # Safety check: cannot reach role input without acknowledging quality report
        if not st.session_state.get("quality_acknowledged"):
            st.session_state["screen"] = "quality"
            st.rerun()
        else:
            screen_role_input_placeholder()

    else:
        # Fallback — if screen value is unknown, restart from the beginning
        st.session_state["screen"] = "api_key"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# In Python, this line checks if this file is being run directly
# (as opposed to being imported by another file).
# Streamlit always runs files directly, so this block always executes.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
