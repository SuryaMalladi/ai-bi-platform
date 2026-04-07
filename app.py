"""
AI-Powered Data Analysis & Intelligence Platform
Phase 3, Step 3 — Data Source Selection
Author: Surya (built with Claude AI)
"""

import streamlit as st
import pandas as pd
import io

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
# SCREEN 4 PLACEHOLDER — DATA QUALITY REPORT
# This screen is built in Phase 3, Step 4.
# For now we show a clear placeholder so the navigation works end-to-end.
# ─────────────────────────────────────────────────────────────────────────────

def screen_quality_placeholder():
    """
    Temporary placeholder for the Data Quality Report screen.
    Will be fully built in Phase 3, Step 4.
    """
    render_topbar("Step 4 — Data Quality Report")
    render_screen_title(
        "🔍",
        "Data Quality Report",
        "Scanning your dataset for quality issues before analysis begins."
    )

    render_card("""
        <div style="text-align:center; padding:40px 0;">
            <div style="font-size:48px; margin-bottom:16px;">🚧</div>
            <div style="font-size:16px; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
                Coming in Phase 3, Step 4
            </div>
            <div style="font-size:13px; color:#94a3b8; max-width:400px; margin:0 auto; line-height:1.6;">
                The Data Quality Report will scan your dataset for missing values,
                duplicates, PII, inconsistent formats, and more — before any
                AI analysis begins.
            </div>
        </div>
    """)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Show a summary of what's loaded
    if st.session_state.get("dataframe") is not None:
        rows = st.session_state["data_rows"]
        cols = st.session_state["data_columns"]
        fname = st.session_state["file_name"]
        render_success_banner(
            f"Dataset ready: {fname} — {rows:,} rows · {cols} columns"
        )

    if st.button("← Back to Data Source", type="secondary"):
        st.session_state["screen"] = "data"
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
            screen_quality_placeholder()

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
