import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from groq import Groq
from supabase import create_client
import json
import re
import time
import hashlib

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SIMBA AI — Business Intelligence Platform",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
GROQ_KEY   = st.secrets.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MAX_ANALYSES_PER_SESSION = 20
MAX_FILE_SIZE_MB         = 50
ANALYSIS_TIMEOUT_SECS    = 45
COOLDOWN_SECS            = 3

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
defaults = {
    "step":                  "landing",
    "df":                    None,
    "df2":                   None,
    "data_source":           None,
    "data_label":            "",
    "data_label2":           "",
    "role":                  "",
    "industry":              "",
    "industry_confirmed":    False,
    "detected_domains":      [],
    "analysis":              None,
    "analysis2":             None,
    "verified_facts":        None,
    "verified_facts2":       None,
    "col_classifications":   {},
    "col_classifications2":  {},
    "chat_history":          [],
    "chat_suggestions":      [],
    "show_chat":             False,
    "analysis_count":        0,
    "last_analysis_time":    0,
    "feedback_given":        False,
    "feedback_score":        None,
    "feedback_count_neg":    0,
    "feedback_text":         "",
    "session_feedback":      [],
    "injection_attempts":    0,
    "session_locked":        False,
    "quality_acknowledged":  False,
    "db_sub_step":           "credentials",
    "db_sb_url":             "",
    "db_sb_key":             "",
    "db_tables":             [],
    "audit_log":             [],
    "comparison_mode":       False,
    "role_profile":          {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# STABLE HASH
# ─────────────────────────────────────────
def stable_hash(s):
    return hashlib.md5(str(s).encode()).hexdigest()[:12]

def format_number(val, is_pct=False):
    """
    Format numbers for professional BI display.
    Percentages: show as-is with % suffix.
    Values: abbreviate to K/M for readability.
    """
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if is_pct:
        return f"{v:.1f}%"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 10_000:
        return f"{v/1_000:.0f}K"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"

# ═══════════════════════════════════════════════════════
# RIF — COMPLETE ROLE INTELLIGENCE FRAMEWORK
# Implements RIF v1.0 Section 2, 4, 5, 6 exactly
# ═══════════════════════════════════════════════════════

# RIF Section 4 — Full abbreviation table
RIF_ABBREVIATIONS = {
    # L1 Executive
    "cfo":       {"level": "L1", "title": "Chief Financial Officer",      "function": "Finance"},
    "ceo":       {"level": "L1", "title": "Chief Executive Officer",       "function": "Executive"},
    "coo":       {"level": "L1", "title": "Chief Operating Officer",       "function": "Operations"},
    "cmo":       {"level": "L1", "title": "Chief Marketing Officer",       "function": "Marketing"},
    "cto":       {"level": "L1", "title": "Chief Technology Officer",      "function": "Executive"},
    "chro":      {"level": "L1", "title": "Chief Human Resources Officer", "function": "HR"},
    "cpo":       {"level": "L1", "title": "Chief People Officer",          "function": "HR"},
    "cco":       {"level": "L1", "title": "Chief Commercial Officer",      "function": "Sales"},
    "md":        {"level": "L1", "title": "Managing Director",             "function": "Executive"},
    # L2 Senior Management
    "fd":        {"level": "L2", "title": "Finance Director",              "function": "Finance"},
    "hrd":       {"level": "L2", "title": "HR Director",                   "function": "HR"},
    "sd":        {"level": "L2", "title": "Sales Director",                "function": "Sales"},
    "bdm":       {"level": "L2", "title": "Business Development Manager",  "function": "Sales"},
    "gm":        {"level": "L2", "title": "General Manager",               "function": "Executive"},
    "vp":        {"level": "L2", "title": "Vice President",                "function": "Executive"},
    "svp":       {"level": "L2", "title": "Senior Vice President",         "function": "Executive"},
    "evp":       {"level": "L2", "title": "Executive Vice President",      "function": "Executive"},
    # L3 Mid Management
    "rsm":       {"level": "L3", "title": "Regional Sales Manager",        "function": "Sales"},
    "hrbp":      {"level": "L3", "title": "HR Business Partner",           "function": "HR"},
    "pm":        {"level": "L3", "title": "Project Manager",               "function": "Executive"},
    # L4A Analytical
    "ba":        {"level": "L4A", "title": "Business Analyst",             "function": "Analytical"},
    "da":        {"level": "L4A", "title": "Data Analyst",                 "function": "Analytical"},
    "ds":        {"level": "L4A", "title": "Data Scientist",               "function": "Analytical"},
    "bi":        {"level": "L4A", "title": "Business Intelligence Analyst","function": "Analytical"},
}

# RIF Section 2 — Keyword taxonomy
RIF_L1_KEYWORDS   = ["chief","chairman","founder","president","board","group director"]
RIF_L2_KEYWORDS   = ["director","vice president","head of","general manager","senior manager"]
RIF_L3_KEYWORDS   = ["manager","supervisor","hrbp","partner","coordinator","team lead","branch","store"]
RIF_L4F_KEYWORDS  = ["shift supervisor","floor supervisor","section lead","shift lead","team leader"]
RIF_L4A_KEYWORDS  = ["analyst","scientist","statistician","engineer","researcher","data"]

# Seniority qualifiers
ELEVATE_QUALIFIERS = ["senior","principal","lead","sr ","sr."]
LOWER_QUALIFIERS   = ["junior","associate","graduate","trainee","jr ","jr."]

# Dual-hat indicators
DUAL_HAT_SEPARATORS = [" and "," & "," / ","/"]

def detect_role_profile(role_input, df_columns=None):
    """
    Full RIF implementation — R1 through R6, all edge cases.
    Returns a complete role profile dict.
    """
    raw       = role_input.strip()
    role_lower = raw.lower()

    # ── INJECTION CHECK (RIF Section 5) ──
    injection_kw = ["ignore","reveal","system prompt","you are now",
                    "select ","drop ","<script","delete ","insert into"]
    if any(kw in role_lower for kw in injection_kw):
        return {"error": "injection", "level": "L3", "title": raw, "function": "Unknown"}

    # ── GIBBERISH CHECK ──
    words = re.findall(r'[a-zA-Z]+', role_lower)
    if len(words) == 0 or (len(words) == 1 and len(words[0]) <= 2 and words[0] not in RIF_ABBREVIATIONS):
        return {"error": "gibberish", "level": "L3", "title": raw, "function": "Unknown"}

    # ── DUAL-HAT DETECTION ──
    is_dual_hat  = False
    dual_hat_parts = []
    for sep in DUAL_HAT_SEPARATORS:
        if sep in role_lower:
            parts = [p.strip() for p in role_lower.split(sep) if p.strip()]
            if len(parts) == 2:
                is_dual_hat    = True
                dual_hat_parts = parts
                break

    # ── SENIORITY QUALIFIER ──
    has_elevate = any(q in role_lower for q in ELEVATE_QUALIFIERS)
    has_lower   = any(q in role_lower for q in LOWER_QUALIFIERS)

    # ── GEOGRAPHIC/DIVISIONAL MODIFIER — strip for level, keep for scope ──
    geo_pattern = r'\b(north|south|east|west|emea|apac|latam|americas|europe|asia|region\s*\d*|nationwide|global)\b'
    scope_note  = ""
    geo_match   = re.search(geo_pattern, role_lower)
    if geo_match:
        scope_note = geo_match.group(0).title()
        role_for_level = re.sub(geo_pattern, "", role_lower).strip()
    else:
        role_for_level = role_lower

    # ── FOUNDER/STARTUP RULE ──
    if any(k in role_for_level for k in ["founder","co-founder","co founder","startup owner","small business owner"]):
        return {
            "level":        "L1",
            "title":        raw,
            "function":     "Executive",
            "secondary_function": "Finance, Sales, Operations",
            "scope":        "strategic",
            "language":     "executive",
            "stats_depth":  "headline",
            "dual_hat":     True,
            "scope_note":   scope_note,
            "interpretation_note": f"Analysis tailored for Founder — merged Executive, Finance, Sales, and Operations profile.",
            "decision_question": "What does this mean for the organisation and what must I decide now?"
        }

    # ── R1: ABBREVIATION EXACT MATCH ──
    # Try each word in the role as a potential abbreviation
    role_words = re.findall(r'[a-zA-Z]+', role_for_level)
    # Only match abbreviations that are standalone words (not parts of longer words)
    # Split on spaces and punctuation to get true word tokens
    true_words = re.split(r'[\s\-_/&,\.]+', role_for_level.strip())
    true_words = [w.strip().lower() for w in true_words if w.strip()]
    for word in true_words:
        if word in RIF_ABBREVIATIONS:
            entry   = RIF_ABBREVIATIONS[word]
            level   = entry["level"]
            title   = entry["title"]
            fn      = entry["function"]
            # Apply seniority qualifiers within band
            sub_note = ""
            if has_elevate:
                sub_note = f"Senior — elevated depth within {level} band."
            elif has_lower:
                sub_note = f"Junior — increased explanation depth within {level} band."
            return _build_profile(level, title, fn, raw, scope_note, sub_note, is_dual_hat, dual_hat_parts, df_columns)

    # ── DUAL-HAT: process each part separately ──
    if is_dual_hat and dual_hat_parts:
        profiles = []
        for part in dual_hat_parts:
            p = _level_from_keywords(part)
            if p:
                profiles.append(p)
        if profiles:
            # Use highest level found
            level_order = {"L1": 0, "L2": 1, "L3": 2, "L4F": 3, "L4A": 4}
            profiles.sort(key=lambda x: level_order.get(x["level"], 5))
            primary    = profiles[0]
            functions  = " / ".join([p["function"] for p in profiles])
            return _build_profile(
                primary["level"], raw, functions, raw, scope_note,
                f"Dual-hat role — merged {functions} profile.",
                True, dual_hat_parts, df_columns
            )

    # ── R2–R6: KEYWORD MATCHING ──
    kw_result = _level_from_keywords(role_for_level)
    if kw_result:
        level = kw_result["level"]
        fn    = kw_result["function"]
        sub_note = ""
        if has_elevate:
            sub_note = f"Senior qualifier detected — elevated depth within {level} band."
        elif has_lower:
            sub_note = f"Junior qualifier detected — increased explanation depth within {level} band."
        return _build_profile(level, raw, fn, raw, scope_note, sub_note, False, [], df_columns)

    # ── R6: AMBIGUOUS — infer from dataset ──
    fn = _infer_function_from_columns(df_columns) if df_columns else "Executive"
    return _build_profile(
        "L3", raw, fn, raw, scope_note,
        f"Role interpreted as Mid Management — function inferred from dataset as {fn}. Use Switch Role to refine.",
        False, [], df_columns
    )

def _level_from_keywords(role_lower):
    """Map role string to level using keyword rules R2 onwards."""
    # L4F check first — more specific than L3
    if any(k in role_lower for k in RIF_L4F_KEYWORDS):
        return {"level": "L4F", "function": "Operations"}
    # L4A
    if any(k in role_lower for k in RIF_L4A_KEYWORDS):
        return {"level": "L4A", "function": "Analytical"}
    # L1
    if any(k in role_lower for k in RIF_L1_KEYWORDS):
        fn = _infer_function_keyword(role_lower)
        return {"level": "L1", "function": fn}
    # L2
    if any(k in role_lower for k in RIF_L2_KEYWORDS):
        fn = _infer_function_keyword(role_lower)
        return {"level": "L2", "function": fn}
    # L3
    if any(k in role_lower for k in RIF_L3_KEYWORDS):
        fn = _infer_function_keyword(role_lower)
        return {"level": "L3", "function": fn}
    return None

def _infer_function_keyword(role_lower):
    """Infer function domain from keywords in role string."""
    if any(k in role_lower for k in ["finance","financial","budget","treasury","accounting","cfo","fd"]):
        return "Finance"
    if any(k in role_lower for k in ["sales","commercial","revenue","account","business development","cco"]):
        return "Sales"
    if any(k in role_lower for k in ["hr","human resource","people","talent","workforce","chro","hrbp","hrd"]):
        return "HR"
    if any(k in role_lower for k in ["operations","ops","production","supply chain","logistics","coo"]):
        return "Operations"
    if any(k in role_lower for k in ["marketing","brand","campaign","digital","growth","cmo"]):
        return "Marketing"
    if any(k in role_lower for k in ["data","analytics","bi","intelligence","research","insight"]):
        return "Analytical"
    return "Executive"

def _infer_function_from_columns(df_columns):
    """Infer function from dataset column names when role is vague."""
    if not df_columns:
        return "Executive"
    cols_lower = " ".join(df_columns).lower()
    if any(k in cols_lower for k in ["budget","actual","variance","cost","expense","profit"]):
        return "Finance"
    if any(k in cols_lower for k in ["sales","revenue","target","quota","units sold"]):
        return "Sales"
    if any(k in cols_lower for k in ["employee","attrition","headcount","tenure","satisfaction"]):
        return "HR"
    if any(k in cols_lower for k in ["defects","downtime","efficiency","produced","cycle"]):
        return "Operations"
    return "Executive"

def _build_profile(level, title, function, raw_input, scope_note, sub_note, dual_hat, dual_parts, df_columns):
    """Build complete role profile dict from detected level and function."""
    level_map = {
        "L1":  {"scope": "strategic",   "language": "executive",     "stats_depth": "headline",      "decision": "What does this mean for the organisation and what must I decide now?"},
        "L2":  {"scope": "tactical",    "language": "professional",  "stats_depth": "moderate",      "decision": "What should I focus on this month and what do I escalate or delegate?"},
        "L3":  {"scope": "operational", "language": "plain",         "stats_depth": "essential",     "decision": "What do I do next and what does my team need to know right now?"},
        "L4F": {"scope": "operational", "language": "plain",         "stats_depth": "outputs-only",  "decision": "What is the number and what action do I take today?"},
        "L4A": {"scope": "analytical",  "language": "technical",     "stats_depth": "full",          "decision": "What does this data tell us statistically and what are the caveats?"},
    }
    lm = level_map.get(level, level_map["L3"])

    # Chart rules per level — enforced in Python after AI returns
    allowed_charts = {
        "L1":  ["kpi", "bar", "line", "pie"],
        "L2":  ["bar", "line", "kpi"],
        "L3":  ["bar", "line", "kpi"],
        "L4F": ["kpi", "bar"],
        "L4A": ["scatter", "bar", "line"],
    }
    max_charts = {
        "L1": 4, "L2": 4, "L3": 4, "L4F": 2, "L4A": 5
    }

    interpretation_note = ""
    if sub_note:
        interpretation_note = sub_note
    if scope_note:
        interpretation_note += f" Scope: {scope_note} area."
    if dual_hat:
        interpretation_note += f" Dual-hat role detected."

    return {
        "level":              level,
        "title":              title,
        "function":           function,
        "raw_input":          raw_input,
        "scope":              lm["scope"],
        "language":           lm["language"],
        "stats_depth":        lm["stats_depth"],
        "decision_question":  lm["decision"],
        "allowed_charts":     allowed_charts.get(level, ["bar","line","kpi"]),
        "max_charts":         max_charts.get(level, 4),
        "scope_note":         scope_note,
        "dual_hat":           dual_hat,
        "interpretation_note": interpretation_note.strip(),
    }

def enforce_chart_rules(charts, role_profile):
    """
    Python-side enforcement of chart type rules per level.
    Rejects forbidden chart types. Caps chart count.
    Returns only permitted charts.
    """
    allowed  = role_profile.get("allowed_charts", ["bar","line","kpi","pie","scatter"])
    max_n    = role_profile.get("max_charts", 4)
    enforced = []
    for ch in charts:
        ctype = ch.get("type","bar").lower()
        if ctype in allowed:
            enforced.append(ch)
        else:
            # Replace forbidden type with best allowed alternative
            if "bar" in allowed:
                ch["type"] = "bar"
                enforced.append(ch)
            elif "kpi" in allowed:
                ch["type"] = "kpi"
                enforced.append(ch)
    return enforced[:max_n]

# ═══════════════════════════════════════════════════════
# MASTER SYSTEM PROMPT
# ═══════════════════════════════════════════════════════
MASTER_PROMPT = """You are a specialist BI analyst AI embedded in SIMBA AI.

You receive:
1. VERIFIED FACTS — numbers computed by Python from the actual data. Absolute ground truth.
2. ROLE PROFILE — the complete interpreted profile for this user including level, function, decision question, language register, and allowed chart types.

═══ CORE LAW ═══
Use ONLY verified facts for every number you state. Never compute your own numbers.
Same data + different role = genuinely different analysis. If two roles get the same charts or recommendations — the system has failed.

═══ YOUR ANALYSIS PROCESS — FOLLOW IN ORDER ═══
Step 1: Read ROLE PROFILE. Anchor every output to the DECISION QUESTION for this role.
Step 2: Read verified_stats. Identify the 3-5 most critical facts for THIS role's function.
Step 3: For every negative finding — look in group_aggregates. Find the SAME SEGMENT across multiple columns. If West region Sales is low, look up "West" in group_aggregates for ALL other columns (defects, returns, efficiency, etc.). If a correlated column shows the same segment performing badly — state it as evidence with the verified number. If no correlated data — write "Possible cause (not in data):" and give a real-world hypothesis.
Step 4: Select charts based on ROLE PROFILE allowed_charts only. Never use a chart type not in the allowed list.
Step 5: Write in the language register for this level — executive (max 20 words/sentence), professional (max 25), plain (max 30), technical (full stats).

═══ LEVEL-SPECIFIC OUTPUT REQUIREMENTS ═══

L1 EXECUTIVE (CEO, CFO, COO, CMO, CTO, CHRO, MD, Board, Founder, Chairman, President):
Decision question: "What does this mean for the organisation and what must I decide now?"
Language: Max 20 words per sentence. Decisive. No hedging. No operational detail. No raw stats.
Charts: Org-level KPIs, period trends, high-level segment comparisons only.
Confidence language HIGH: "Confirmed significant trend."
Confidence language MEDIUM: "Warrants monitoring."
Confidence language INDICATIVE: "Early signal only."

L2 SENIOR MANAGEMENT (Director, VP, SVP, EVP, Head of Function, GM, BDM, FD, HRD, SD):
Decision question: "What should I focus on this month and what do I escalate or delegate?"
Language: Max 25 words. Professional. Always give context with numbers.
Finance: Always state favourable/adverse. Always period-qualify YTD/QTD/MTD.
Sales: Rankings always appropriate. Time-bound recommendations preferred.
HR: Aggregate only. Never reference individuals. Never punitive conclusions.
Operations: Precision non-negotiable.
Confidence language HIGH: "Data strongly supports this."
Confidence language MEDIUM: "Suggests this — investigate first."
Confidence language INDICATIVE: "Too early — flag for review."

L3 MID MANAGEMENT (Store Manager, Team Manager, Branch Manager, HRBP, RSM, Ops Supervisor):
Decision question: "What do I do next and what does my team need to know right now?"
Language: Max 30 words. Plain English. Action-oriented. Specific numbers.
Charts: Their area's performance, team/product breakdowns, their specific targets.
Confidence language HIGH: "Consistent — not a one-off."
Confidence language MEDIUM: "Worth watching."
Confidence language INDICATIVE: "Keep on radar."

L4F FRONTLINE (Shift Supervisor, Floor Supervisor, Section Lead, Team Leader ops-focus):
Decision question: "What is the number and what action do I take today?"
Language: Max 15 words. Number then action only.
Charts: Maximum 2. Simple bar or KPI only.
Confidence language HIGH: "Numbers are clear."
Confidence language MEDIUM: "Not certain — monitor."
Confidence language INDICATIVE: "Too early to say."

L4A ANALYTICAL (Data Scientist, Data Analyst, Business Analyst, BI Analyst, Statistician):
Decision question: "What does this data tell us statistically and what are the caveats?"
Language: Technical. p-values, confidence intervals, N counts welcomed.
Charts: Scatter for correlations, distributions, detailed breakdowns. No KPI cards.
Full methodology visible. Assumptions always stated explicitly.
Confidence language HIGH: "Significant p<0.05, N=[count]."
Confidence language MEDIUM: "Directional only — expand dataset."
Confidence language INDICATIVE: "Significance unconfirmed."

═══ TRAFFIC LIGHT RULES ═══
Value field: copy from verified_stats correct_aggregate exactly.
has_targets=true: GREEN=within 5% of target, AMBER=5-15% below, RED=>15% below.
has_targets=false: base on trend — GREEN=positive/improving, AMBER=flat/mixed, RED=declining. Add target_note: "No target column — status based on trend direction."

═══ CHART FIELD RULES ═══
x_field and y_field: character-for-character exact match from exact_column_names list.
Never reference a column not in that list.
aggregation: follow aggregation_hint from verified_stats exactly.
Only use chart types listed in role_profile.allowed_charts.

═══ CAPTION RULE ═══
Every chart caption must reference at least one specific verified number from verified_stats.
No generic captions. Every caption anchors to actual data.

═══ EXECUTIVE SUMMARY RULES ═══
sentence_1: Most critical quantified finding — must include a verified number.
sentence_2: Cause found in data with evidence, or clearly labelled hypothesis.
sentence_3: Specific action THIS ROLE should take today — not generic.
Sentence length must respect word limits for the level.

═══ NARRATIVE RULES ═══
Written in the voice and register of the role level.
L1: Strategic, decisive, board-room tone. Short sentences. No jargon.
L2: Professional, analytical, departmental focus.
L3: Plain English, action-oriented, team-focused.
L4A: Technical, methodological, statistical.

═══ WHAT YOU MUST NEVER DO ═══
Never write a number not in verified_facts.
Never assume a target not in verified_facts.
Never present a hypothesis as a data fact.
Never make judgements about individual employees.
Never suggest discriminatory conclusions from demographic data.
Never present correlation as causation without qualification.
Never suppress a critical finding because it falls outside the role's primary domain.

═══ OUTPUT — VALID JSON ONLY. NO MARKDOWN. NO PREAMBLE. ═══
{
  "role_interpreted": "string — full expanded title",
  "level": "L1|L2|L3|L4F|L4A",
  "function": "string",
  "interpretation_note": "string or empty",
  "executive_summary": {
    "sentence_1": "string — quantified finding with verified number",
    "sentence_2": "string — cause with evidence or labelled hypothesis",
    "sentence_3": "string — specific action for this role today"
  },
  "traffic_lights": [
    {
      "metric": "string",
      "status": "GREEN|AMBER|RED",
      "value": "string — exact value from verified_stats",
      "reason": "string",
      "target_note": "string or empty"
    }
  ],
  "anomalies": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "description": "string — finding plus cause if visible in data",
      "metric": "string"
    }
  ],
  "charts": [
    {
      "type": "bar|line|pie|scatter|kpi",
      "title": "string",
      "x_field": "string — EXACT column name",
      "y_field": "string — EXACT column name",
      "aggregation": "sum|mean",
      "caption": "string — must reference a specific verified number",
      "sentiment": "POSITIVE|NEGATIVE|NEUTRAL|URGENT",
      "confidence": "HIGH|MEDIUM|INDICATIVE"
    }
  ],
  "recommendations": [
    {
      "priority": 1,
      "action": "string",
      "evidence": "string — verified fact that justifies this action",
      "hypothesis": "string or empty",
      "owner": "string",
      "timeframe": "IMMEDIATE|SHORT_TERM|STRATEGIC"
    }
  ],
  "narrative": {
    "opening": "string",
    "body": ["string", "string"],
    "close": "string"
  },
  "chat_suggestions": [
    "string — MUST reference a specific column name or metric value from exact_column_names AND be framed for this role's decision question. Example for CFO on sales data: 'What is the total Sales figure for the North region and how does it compare to the South?' Example for Store Manager: 'Which product had the highest Units Sold last month?' NEVER write generic questions like 'What are the risks?' or 'What should I prioritise?' Every question must name a real column or real segment value from the data.",
    "string — different column or metric from suggestion 1, still role-specific",
    "string — different column or metric from suggestions 1 and 2, still role-specific",
    "string — different column or metric, forward-looking action question for this role"
  ],
  "evaluation": {
    "relevance_score": 8,
    "accuracy_validated": "YES|PARTIAL|NO",
    "coverage": "string",
    "confidence_overall": "HIGH|MEDIUM|INDICATIVE",
    "bias_check": "BALANCED|IMBALANCED|NOT_APPLICABLE",
    "bias_detail": "string or NONE",
    "evaluation_status": "COMPLETE|PARTIAL|FAILED"
  }
}"""

# ═══════════════════════════════════════════════════════
# COMPARISON MODE PROMPT
# ═══════════════════════════════════════════════════════
COMPARISON_PROMPT = """You are a specialist BI analyst comparing two datasets for a specific role.

You receive VERIFIED FACTS for Dataset A and Dataset B, and a ROLE PROFILE.

Your job:
1. Compare the same metrics across both datasets.
2. Use ONLY numbers from verified_facts_A and verified_facts_B.
3. Identify what improved, what declined, what stayed stable, what changed significantly.
4. For major changes — look in group_aggregates of both datasets to find which segments drove the change.
5. Provide a root cause hypothesis where the data shows a pattern.
6. Write in the language register of the role level.

OUTPUT — VALID JSON ONLY. NO MARKDOWN. NO PREAMBLE.
{
  "comparison_summary": {
    "sentence_1": "string — most significant change between A and B with verified numbers",
    "sentence_2": "string — second key finding with evidence",
    "sentence_3": "string — action this role should take based on the comparison"
  },
  "improved": [
    {"metric": "string", "value_a": "string", "value_b": "string", "change_pct": "string", "note": "string"}
  ],
  "declined": [
    {"metric": "string", "value_a": "string", "value_b": "string", "change_pct": "string", "note": "string"}
  ],
  "stable": [
    {"metric": "string", "value_a": "string", "value_b": "string", "note": "string"}
  ],
  "significant_changes": [
    {"metric": "string", "finding": "string", "root_cause_hypothesis": "string"}
  ],
  "recommendations": [
    {"priority": 1, "action": "string", "evidence": "string", "timeframe": "IMMEDIATE|SHORT_TERM|STRATEGIC"}
  ],
  "narrative": {
    "opening": "string",
    "body": ["string"],
    "close": "string"
  }
}"""

# ═══════════════════════════════════════════════════════
# PYTHON FACT ENGINE
# ═══════════════════════════════════════════════════════

def classify_columns(df):
    """
    Classify every numeric column as percentage or value.
    Fixed: added missing keywords (headcount, units, tenure, cycle, count).
    Percentage → MEAN aggregation. Value → SUM aggregation.
    """
    pct_keywords = [
        "%", "pct", "percent", "rate", "ratio", "score",
        "efficiency", "satisfaction", "margin", "accuracy",
        "utilisation", "utilization", "attendance", "conversion",
        "churn", "yield", "quality", "performance", "rating",
        "completion", "occupancy", "fill"
    ]
    value_keywords = [
        "sales", "revenue", "cost", "budget", "actual", "spend",
        "amount", "units", "count", "total", "quantity", "volume",
        "output", "produced", "defects", "returns", "headcount",
        "salary", "forecast", "profit", "loss", "income", "expense",
        "price", "target", "quota", "orders", "transactions",
        "tenure", "days", "hours", "minutes", "seconds",
        "cycle", "downtime", "time", "duration", "age",
        "number", "num", "no", "qty", "hrs", "ft", "sq"
    ]

    classifications = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        col_clean = col.lower().replace("_", " ").replace("-", " ").replace("%", " pct ")
        words     = set(col_clean.split())

        is_pct = any(k in words or col_clean.endswith(k) for k in pct_keywords)
        is_val = any(k in words for k in value_keywords)

        if is_pct and not is_val:
            classifications[col] = "percentage"
        elif is_val and not is_pct:
            classifications[col] = "value"
        elif is_pct and is_val:
            # Both match — percentage wins if column ends with a pct keyword
            if any(col_clean.strip().endswith(k) for k in pct_keywords):
                classifications[col] = "percentage"
            else:
                classifications[col] = "value"
        else:
            # Heuristic — only apply when range is genuinely 0-100 AND not a count
            s = df[col].dropna()
            if (len(s) > 0
                    and s.min() >= 0
                    and s.max() <= 100
                    and s.max() > 1
                    and s.mean() <= 100
                    and s.nunique() > 5):     # Avoid binary and near-binary columns
                classifications[col] = "percentage"
            else:
                classifications[col] = "value"

    return classifications

def detect_target_columns(df):
    """
    Fixed version: correct best_score update logic.
    Pairs each target column with its best matching actual column.
    """
    target_kw = ["target", "budget", "quota", "plan", "goal"]
    actual_kw = ["actual", "sales", "revenue", "spend", "output",
                 "produced", "achieved", "amount"]

    num_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in num_cols if any(k in c.lower() for k in target_kw)]
    actual_cols = [c for c in num_cols if any(k in c.lower() for k in actual_kw)]

    pairs        = {}
    used_actuals = set()
    stop_words   = {"target","actual","budget","plan","goal","vs","and","by"}

    for tc in target_cols:
        tc_words   = set(tc.lower().replace("_"," ").split()) - stop_words
        best_ac    = None
        best_score = -1                          # Fixed: simple int comparison

        for ac in actual_cols:
            if ac in used_actuals or ac == tc:
                continue
            ac_words = set(ac.lower().replace("_"," ").split()) - stop_words
            shared   = len(tc_words & ac_words)
            if shared > best_score:              # Fixed: correct comparison
                best_score = shared
                best_ac    = ac

        if best_ac:
            pairs[tc] = best_ac
            used_actuals.add(best_ac)
        elif actual_cols:
            for ac in actual_cols:
                if ac not in used_actuals and ac != tc:
                    pairs[tc] = ac
                    used_actuals.add(ac)
                    break

    return pairs, target_cols, actual_cols

def is_binary_column(s):
    return set(s.dropna().unique()).issubset({0, 1, 0.0, 1.0})

def is_identifier_column(col_name, df):
    col_lower  = col_name.lower()
    id_kw      = ["id", "_id", "code", "ref", "number", "no", "num", "key", "index"]
    if any(col_lower == k or col_lower.endswith(f"_{k}") or col_lower.startswith(f"{k}_")
           for k in id_kw):
        return True
    col_data = df[col_name].dropna()
    if len(col_data) > 0 and col_data.nunique() / len(col_data) > 0.9:
        return True
    return False

def compute_group_aggregates(df, col_classifications):
    """Group aggregates with correct aggregation per column type."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    groups   = {}
    for cat in cat_cols[:5]:
        groups[cat] = {}
        for col, ctype in list(col_classifications.items())[:8]:
            try:
                if ctype == "percentage":
                    agg = df.groupby(cat)[col].mean().round(2).to_dict()
                else:
                    agg = df.groupby(cat)[col].sum().round(2).to_dict()
                groups[cat][col] = {
                    "values":      agg,
                    "aggregation": "mean" if ctype == "percentage" else "sum"
                }
            except Exception:
                pass
    return groups

def detect_anomalies_python(df, col_classifications):
    """Statistical anomaly detection. Clean, deduplicated, business-readable."""
    anomalies = []
    seen_msgs = set()
    cat_cols  = df.select_dtypes(include=["object"]).columns.tolist()

    skip_cols = set()
    for col in col_classifications:
        s = df[col].dropna()
        if is_binary_column(s) or is_identifier_column(col, df):
            skip_cols.add(col)

    skip_cats = {cat for cat in cat_cols if is_identifier_column(cat, df)}

    for col, ctype in col_classifications.items():
        if col in skip_cols:
            continue
        s = df[col].dropna()
        if len(s) < 4:
            continue
        mean_val = s.mean()
        std_val  = s.std()
        if std_val == 0:
            continue
        outliers = s[np.abs(s - mean_val) > 2 * std_val]
        if len(outliers) > 0:
            col_display = col.replace("_", " ").title()
            if ctype == "percentage":
                msg = (f"{col_display}: {len(outliers)} unusual value(s) "
                       f"(normal avg: {mean_val:.1f}%, "
                       f"range: {outliers.min():.1f}% – {outliers.max():.1f}%)")
            else:
                msg = (f"{col_display}: {len(outliers)} outlier value(s) "
                       f"(avg: {mean_val:,.0f}, "
                       f"outlier range: {outliers.min():,.0f} – {outliers.max():,.0f})")
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                anomalies.append({
                    "column":   col,
                    "type":     "statistical_outlier",
                    "finding":  msg,
                    "severity": "HIGH" if len(outliers) > 3 else "MEDIUM"
                })

        if ctype == "value" and col not in skip_cols:
            zeros    = (s == 0).sum()
            zero_pct = zeros / len(s)
            if zeros > 0 and zero_pct > 0.20:
                col_display = col.replace("_", " ").title()
                msg = (f"{col_display}: {zeros} zero value(s) "
                       f"({zero_pct*100:.0f}% of records) — check for missing data")
                if msg not in seen_msgs:
                    seen_msgs.add(msg)
                    anomalies.append({
                        "column": col, "type": "zero_values",
                        "finding": msg, "severity": "MEDIUM"
                    })

    for cat in cat_cols[:3]:
        if cat in skip_cats:
            continue
        for col, ctype in list(col_classifications.items())[:4]:
            if col in skip_cols or ctype != "value":
                continue
            if is_binary_column(df[col].dropna()):
                continue
            try:
                grouped    = df.groupby(cat)[col].sum()
                seg_counts = df.groupby(cat)[col].count()
                zero_segs  = [seg for seg in grouped.index
                               if grouped[seg] == 0 and seg_counts.get(seg, 0) > 0]
                if zero_segs:
                    col_display = col.replace("_", " ").title()
                    cat_display = cat.replace("_", " ").title()
                    shown = zero_segs[:3]
                    extra = f" and {len(zero_segs)-3} more" if len(zero_segs) > 3 else ""
                    msg   = (f"No {col_display} recorded for {cat_display}: "
                             f"{', '.join(str(s) for s in shown)}{extra}")
                    if msg not in seen_msgs:
                        seen_msgs.add(msg)
                        anomalies.append({
                            "column": col, "type": "zero_segment",
                            "finding": msg, "severity": "MEDIUM"
                        })
            except Exception:
                pass

    return anomalies[:6]

def compute_target_gaps(df, target_pairs, col_classifications):
    """Compute actual vs target gaps with correct aggregation."""
    if not target_pairs:
        return {}
    gaps     = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for target_col, actual_col in target_pairs.items():
        act_ctype    = col_classifications.get(actual_col, "value")
        total_actual = round(float(df[actual_col].mean()), 2) \
                       if act_ctype == "percentage" else round(float(df[actual_col].sum()), 2)
        tgt_ctype    = col_classifications.get(target_col, "value")
        total_target = round(float(df[target_col].mean()), 2) \
                       if tgt_ctype == "percentage" else round(float(df[target_col].sum()), 2)
        gap_pct  = round((total_actual - total_target) / total_target * 100, 1) \
                   if total_target != 0 else 0
        status   = "GREEN" if gap_pct >= -5 else ("AMBER" if gap_pct >= -15 else "RED")
        agg_label = "average" if act_ctype == "percentage" else "total"
        gaps[f"{actual_col}_vs_{target_col}"] = {
            "actual": total_actual, "target": total_target,
            "gap_pct": gap_pct, "status": status, "agg_label": agg_label
        }
        for cat in cat_cols[:3]:
            try:
                if act_ctype == "percentage":
                    seg_actual = df.groupby(cat)[actual_col].mean().round(2)
                    seg_target = df.groupby(cat)[target_col].mean().round(2)
                else:
                    seg_actual = df.groupby(cat)[actual_col].sum().round(2)
                    seg_target = df.groupby(cat)[target_col].sum().round(2)
                seg_gap = ((seg_actual - seg_target) / seg_target * 100).round(1)
                gaps[f"{actual_col}_vs_{target_col}_by_{cat}"] = {
                    "actual_by_segment":  seg_actual.to_dict(),
                    "target_by_segment":  seg_target.to_dict(),
                    "gap_pct_by_segment": seg_gap.to_dict()
                }
            except Exception:
                pass
    return gaps

def compute_verified_stats(df, col_classifications):
    """Python-computed stats. The only numbers the AI is allowed to reference."""
    stats = {}
    for col, ctype in col_classifications.items():
        s = df[col].dropna()
        if len(s) == 0:
            continue
        correct_agg = round(float(s.mean()), 2) if ctype == "percentage" \
                      else round(float(s.sum()), 2)
        stats[col] = {
            "type":                    ctype,
            "aggregation_hint":        "mean — NEVER sum" if ctype == "percentage" else "sum",
            "correct_aggregate":       correct_agg,
            "correct_aggregate_label": "average" if ctype == "percentage" else "total",
            "mean":                    round(float(s.mean()), 2),
            "median":                  round(float(s.median()), 2),
            "std_dev":                 round(float(s.std()), 2),
            "min":                     round(float(s.min()), 2),
            "max":                     round(float(s.max()), 2),
            "total":                   round(float(s.sum()), 2),
            "count":                   int(s.count()),
        }
    return stats

def build_verified_facts(df):
    """Master fact builder. All numbers computed here. AI receives this as ground truth."""
    col_classifications                    = classify_columns(df)
    target_pairs, target_cols, actual_cols = detect_target_columns(df)
    cat_cols  = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = [c for c in df.columns
                 if any(k in c.lower() for k in
                        ["date","time","period","month","year","week","day","quarter"])]
    verified_stats   = compute_verified_stats(df, col_classifications)
    group_aggregates = compute_group_aggregates(df, col_classifications)
    target_gaps      = compute_target_gaps(df, target_pairs, col_classifications)
    anomalies        = detect_anomalies_python(df, col_classifications)
    cat_distributions = {}
    for c in cat_cols[:8]:
        cat_distributions[c] = df[c].value_counts().head(10).to_dict()

    facts = {
        "total_rows":                len(df),
        "total_columns":             len(df.columns),
        "exact_column_names":        list(df.columns),
        "numeric_columns":           list(col_classifications.keys()),
        "categorical_columns":       cat_cols,
        "date_columns":              date_cols,
        "column_classifications":    {
            col: {"type": ctype,
                  "aggregation_hint": "mean — NEVER sum" if ctype == "percentage" else "sum"}
            for col, ctype in col_classifications.items()
        },
        "verified_stats":            verified_stats,
        "group_aggregates":          group_aggregates,
        "target_columns_found":      target_cols,
        "actual_columns_found":      actual_cols,
        "has_targets":               len(target_pairs) > 0,
        "target_gaps":               target_gaps,
        "python_detected_anomalies": anomalies,
        "categorical_distributions": cat_distributions,
    }
    return facts, col_classifications

def trim_facts_for_token_limit(facts, max_chars=12000):
    """
    Trim facts payload to stay within token limits.
    Priority: always remove sample_rows first, then
    trim categorical_distributions, preserve group_aggregates last.
    """
    facts_copy = dict(facts)
    # Step 1: always remove sample_rows
    facts_copy.pop("sample_rows", None)

    # Step 2: check size
    payload = json.dumps(facts_copy, default=str)
    if len(payload) <= max_chars:
        return facts_copy

    # Step 3: trim categorical_distributions to 3 items per key
    facts_copy["categorical_distributions"] = {
        k: dict(list(v.items())[:3])
        for k, v in facts_copy.get("categorical_distributions", {}).items()
    }
    payload = json.dumps(facts_copy, default=str)
    if len(payload) <= max_chars:
        return facts_copy

    # Step 4: remove categorical_distributions entirely
    facts_copy.pop("categorical_distributions", None)
    payload = json.dumps(facts_copy, default=str)
    if len(payload) <= max_chars:
        return facts_copy

    # Step 5: trim group_aggregates to first 3 category columns
    ga = facts_copy.get("group_aggregates", {})
    facts_copy["group_aggregates"] = dict(list(ga.items())[:3])
    return facts_copy

def build_stat_summary_table(df, col_classifications):
    """Statistical summary — Python computed, never AI written."""
    rows = []
    for col, ctype in col_classifications.items():
        s = df[col].dropna()
        if len(s) == 0:
            continue
        correct_agg = round(float(s.mean()), 2) if ctype == "percentage" \
                      else round(float(s.sum()), 2)
        rows.append({
            "Metric":      col,
            "Type":        "Percentage/Rate" if ctype == "percentage" else "Count/Value",
            "Correct Agg": correct_agg,
            "Agg Method":  "Average" if ctype == "percentage" else "Total",
            "Mean":        round(float(s.mean()), 2),
            "Median":      round(float(s.median()), 2),
            "Std Dev":     round(float(s.std()), 2),
            "Min":         round(float(s.min()), 2),
            "Max":         round(float(s.max()), 2),
            "Data Points": int(s.count()),
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════
# PYTHON EVALUATION ENGINE — EF Methods 2, 3, 5, 6
# ═══════════════════════════════════════════════════════

def run_accuracy_validation(analysis, verified_stats, tolerance_pct=0.5):
    """
    EF Method 2: Cross-validate AI traffic light values against Python stats.
    Returns pass/fail per metric and overall status.
    """
    results    = {}
    all_passed = []
    for tl in analysis.get("traffic_lights", []):
        metric = tl.get("metric", "").lower()
        value  = tl.get("value", "")
        for col, stats in verified_stats.items():
            col_lower = col.lower().replace("_"," ")
            # Match if metric contains column name words or vice versa
            col_words    = set(col_lower.split())
            metric_words = set(metric.split())
            if len(col_words & metric_words) > 0:
                try:
                    nums = re.findall(r"[\d]+\.?\d*", value.replace(",",""))
                    if nums:
                        ai_val   = float(nums[0])
                        py_val   = stats["correct_aggregate"]
                        pct_diff = abs(ai_val - py_val) / max(abs(py_val), 1) * 100
                        passed   = pct_diff <= tolerance_pct
                        results[f"tl_{col}"] = {
                            "ai_value": ai_val, "py_value": py_val,
                            "pct_diff": round(pct_diff, 2), "passed": passed
                        }
                        all_passed.append(passed)
                except Exception:
                    pass
                break

    if not all_passed:
        return results, "PARTIAL"
    if all(all_passed):
        return results, "YES"
    if any(all_passed):
        return results, "PARTIAL"
    return results, "NO"

def run_completeness_check(analysis, facts):
    """EF Method 3: Coverage of key dataset areas."""
    numeric_cols = list(facts.get("column_classifications", {}).keys())
    total_areas  = len(numeric_cols)
    if total_areas == 0:
        return 100, "0 of 0 areas", "Analysis Coverage: No numeric columns to check."
    all_text = json.dumps(analysis, default=str).lower()
    addressed = sum(1 for col in numeric_cols if col.lower() in all_text)
    uncovered = [col for col in numeric_cols if col.lower() not in all_text]
    pct       = int(addressed / total_areas * 100)
    cov_str   = f"{addressed} of {total_areas} key areas addressed"
    if pct >= 90:
        msg = f"✅ Analysis Coverage: {cov_str}."
    elif pct >= 60:
        msg = f"ℹ️ Analysis Coverage: {cov_str}. Not addressed: {', '.join(c.replace('_',' ').title() for c in uncovered[:5])}"
    else:
        msg = f"⚠️ Coverage Notice: {cov_str}. Consider switching role for broader coverage."
    return pct, cov_str, msg

def run_bias_detection(analysis, df):
    """EF Method 6: Segment coverage check."""
    date_indicators = ['date','time','period','month','year','week','day','quarter','timestamp']
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if df[c].nunique() >= 2
                and not any(k in c.lower() for k in date_indicators)]
    if not cat_cols:
        return "NOT_APPLICABLE", "No categorical columns with 2+ values.", []
    all_text       = json.dumps(analysis, default=str).lower()
    uncovered_segs = []
    for cat in cat_cols[:4]:
        values     = df[cat].value_counts()
        total_rows = len(df)
        for val, count in values.items():
            if count / total_rows >= 0.15:
                if str(val).lower() not in all_text:
                    uncovered_segs.append(f"{cat}: {val}")
    if not uncovered_segs:
        return "BALANCED", "Coverage Balance: All key segments addressed.", []
    return "IMBALANCED", f"⚠️ Coverage Notice: Not addressed: {', '.join(uncovered_segs[:5])}", uncovered_segs

def compute_confidence_overall(analysis, df):
    """EF Method 5: CONFIDENCE_OVERALL = lowest individual confidence. Never average."""
    tier_order = {"INDICATIVE": 0, "MEDIUM": 1, "HIGH": 2}
    lowest     = "HIGH"
    for ch in analysis.get("charts", []):
        conf = ch.get("confidence", "HIGH")
        if tier_order.get(conf, 2) < tier_order.get(lowest, 2):
            lowest = conf
    # Hard rule: below 30 rows = INDICATIVE
    if len(df) < 30:
        lowest = "INDICATIVE"
    return lowest

# ═══════════════════════════════════════════════════════
# EF METHOD 1 — RELEVANCE SCORING (second AI call)
# ═══════════════════════════════════════════════════════

def run_relevance_scoring(analysis, role_profile):
    """
    EF Method 1: Second AI call scores each chart and insight for role relevance.
    Returns scored charts with labels. Filters low-relevance items.
    """
    level    = role_profile.get("level", "L3")
    function = role_profile.get("function", "Executive")
    decision = role_profile.get("decision_question", "")

    charts       = analysis.get("charts", [])
    if not charts:
        return analysis, 8

    # Build scoring request
    items_to_score = []
    for i, ch in enumerate(charts):
        items_to_score.append({
            "index":   i,
            "title":   ch.get("title", ""),
            "caption": ch.get("caption", ""),
            "type":    ch.get("type", "")
        })

    scoring_prompt = (
        f"You are scoring chart relevance for a {level} {function} role.\n"
        f"Decision question: {decision}\n"
        f"Score each chart 1-10 for relevance to this role.\n"
        f"8-10: Highly relevant — directly answers decision question.\n"
        f"5-7: Moderately relevant — useful context.\n"
        f"1-4: Low relevance — wrong level or function.\n"
        f"Return ONLY JSON: {{\"scores\": [{{\"index\": 0, \"score\": 8}}, ...]}}\n\n"
        f"Charts to score:\n{json.dumps(items_to_score)}"
    )
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=0.1,
            max_tokens=300
        )
        raw     = r.choices[0].message.content.strip()
        raw     = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw)
        scored  = json.loads(raw)
        scores  = {item["index"]: item["score"] for item in scored.get("scores", [])}

        # Apply EF rules: tag charts with relevance labels
        avg_score    = 0
        scored_charts = []
        for i, ch in enumerate(charts):
            score = scores.get(i, 7)
            avg_score += score
            ch["_relevance_score"] = score
            if score >= 8:
                ch["_relevance_label"] = ""
            elif score >= 5:
                ch["_relevance_label"] = "ℹ️ This insight may be more relevant in a different role context."
            else:
                ch["_relevance_label"] = "⚠️ Low relevance for your role. You may wish to dismiss this."
            scored_charts.append(ch)

        avg_score = round(avg_score / max(len(charts), 1), 1)
        analysis["charts"] = scored_charts
        return analysis, avg_score

    except Exception:
        # If second AI call fails — mark all as standard, return original
        for ch in charts:
            ch["_relevance_score"] = 7
            ch["_relevance_label"] = ""
        analysis["charts"] = charts
        return analysis, 7

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def detect_pii(df):
    pii_flags = []
    ep = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    pp = r'(\+?\d[\d\s\-]{8,}\d)'
    # Columns that are clearly date/time — never flag as PII
    date_indicators = ['date','time','period','month','year','week','day','quarter','timestamp']
    for col in df.columns:
        cl = col.lower()
        # Skip date columns entirely — dates are not PII
        if any(k in cl for k in date_indicators):
            continue
        if any(k in cl for k in ['email','phone','mobile','ssn','passport',
                                  'credit','card','national_id','nid']):
            pii_flags.append(col)
            continue
        for val in df[col].dropna().astype(str).head(20):
            if re.search(ep, val) or re.search(pp, val):
                pii_flags.append(col)
                break
    return list(set(pii_flags))

def check_date_format_issues(df):
    issues = []
    for col in df.columns:
        if any(k in col.lower() for k in ["date","time","period"]):
            if df[col].dtype == object:
                sample   = df[col].dropna().astype(str).head(20).tolist()
                patterns = set()
                for v in sample:
                    if re.match(r'\d{4}-\d{2}-\d{2}', v):
                        patterns.add("YYYY-MM-DD")
                    elif re.match(r'\d{2}/\d{2}/\d{4}', v):
                        patterns.add("DD/MM/YYYY")
                    elif re.match(r'\d{2}-\d{2}-\d{4}', v):
                        patterns.add("DD-MM-YYYY")
                    elif re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', v):
                        patterns.add("mixed_slash")
                if len(patterns) > 1:
                    issues.append(f"{col}: mixed date formats ({', '.join(patterns)})")
    return issues

def check_type_inconsistencies(df):
    issues = []
    date_indicators = ['date','time','period','month','year','week','day','quarter','timestamp']
    for col in df.select_dtypes(include=["object"]).columns:
        # Skip date columns — they legitimately contain date strings
        if any(k in col.lower() for k in date_indicators):
            continue
        sample        = df[col].dropna().head(50)
        numeric_count = sum(1 for v in sample
                            if str(v).replace('.','').replace('-','').replace(',','').isdigit())
        if len(sample) > 0 and numeric_count / len(sample) > 0.5:
            issues.append(f"{col.replace('_',' ').title()}: appears to contain mixed numeric and text values")
    return issues

def clean_ai_text(text):
    """
    Strip code-style formatting, backticks, and markdown artifacts
    from AI-generated text before displaying to user.
    """
    if not text:
        return text
    # Remove backtick code spans
    text = re.sub(r'`[^`]*`', lambda m: m.group(0).strip('`'), text)
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'', text)
    text = re.sub(r'\*([^*]+)\*', r'', text)
    # Remove any remaining backticks
    text = text.replace('`', '')
    return text.strip()

def detect_industry_from_data(df):
    all_text = " ".join(df.columns).lower()
    try:
        all_text += " " + " ".join(
            df.select_dtypes(include=["object"]).iloc[:5].to_string().lower()
        )
    except Exception:
        pass
    domains        = []
    domain_signals = {
        "Retail / Sales":    ["sales","product","store","category","revenue","customer","units sold","returns"],
        "HR / People":       ["employee","attrition","headcount","salary","tenure","satisfaction","department","gender"],
        "Finance / Budget":  ["budget","actual","variance","forecast","cost centre","approved","expense","profit"],
        "Operations":        ["defects","cycle time","downtime","efficiency","produced","shift","line","output"],
        "Marketing":         ["campaign","impressions","clicks","ctr","roas","cpl","cac","conversion"],
        "Healthcare":        ["patient","diagnosis","treatment","hospital","clinical","medication"],
        "Logistics":         ["shipment","delivery","freight","warehouse","route","carrier","dispatch"],
        "Education":         ["student","grade","course","enrollment","attendance","teacher","score"],
    }
    for domain, signals in domain_signals.items():
        if sum(1 for s in signals if s in all_text) >= 2:
            domains.append(domain)
    return domains if domains else ["General Business"]

def add_audit_entry(role, data_label, analysis_count):
    """In-session audit log per PRD Section 6.3."""
    st.session_state.audit_log.append({
        "timestamp":      time.strftime("%H:%M:%S"),
        "role":           role,
        "data_source":    data_label,
        "analysis_no":    analysis_count,
    })

def call_openai(messages, temperature=0.3, max_tokens=4000):
    client = OpenAI(api_key=OPENAI_KEY)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return r.choices[0].message.content.strip()

def call_groq(system_msg, user_msg, max_tokens=600):
    client = Groq(api_key=GROQ_KEY)
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=0.2,
        max_tokens=max_tokens
    )
    return r.choices[0].message.content.strip()

def parse_json_safe(raw):
    try:
        raw = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw.strip())
        raw = re.sub(r',(\s*[}\]])', r'\1', raw)
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, str(e)

# ─────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────

def sb_get_tables(url, key):
    candidates = [
        "retail_sales","hr_people","finance_budget","operations_data",
        "sales","orders","employees","customers","products",
        "transactions","inventory","finance","hr","operations",
        "budget","performance","targets","reports"
    ]
    try:
        client    = create_client(url, key)
        available = []
        for t in candidates:
            try:
                r = client.table(t).select("*").limit(1).execute()
                if r.data is not None:
                    available.append(t)
            except Exception:
                pass
        return available if available else [], None
    except Exception as e:
        return [], str(e)

def sb_load_table(url, key, table_name):
    try:
        client = create_client(url, key)
        r      = client.table(table_name).select("*").execute()
        if r.data:
            return pd.DataFrame(r.data), None
        return None, "Table is empty."
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────
# CHART RENDERER
# ─────────────────────────────────────────

COLOURS = ["#3b82f6","#10b981","#f59e0b","#ef4444",
           "#6366f1","#8b5cf6","#06b6d4","#f97316"]
SENTIMENT_COLOURS = {
    "POSITIVE":  "#10b981",   # green
    "NEGATIVE":  "#ef4444",   # red
    "URGENT":    "#f59e0b",   # amber
    "NEUTRAL":   "#3b82f6",   # blue
    "positive":  "#10b981",
    "negative":  "#ef4444",
    "urgent":    "#f59e0b",
    "neutral":   "#3b82f6",
}
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="sans-serif", size=11),
    title=dict(font=dict(color="#e2e8f0", size=13), x=0),
    xaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", showgrid=True),
    yaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), orientation="h"),
    margin=dict(l=10, r=10, t=45, b=10),
    colorway=COLOURS,
    height=320
)

def exact_col(field, df_columns):
    if not field:
        return None
    if field in df_columns:
        return field
    fl = field.lower().strip()
    for col in df_columns:
        if col.lower().strip() == fl:
            return col
    return None

def clean_axis_label(col_name):
    return col_name.replace("_", " ").replace("%", " %").strip().title()

def render_chart(ch, df, col_classifications):
    """Render chart with correct aggregation and clean labels."""
    chart_type = ch.get("type", "bar")
    title      = str(ch.get("title", "")) or "Chart"
    x_field    = ch.get("x_field", "")
    y_field    = ch.get("y_field", "")
    sentiment  = ch.get("sentiment", "NEUTRAL")
    cols       = df.columns.tolist()
    xc         = exact_col(x_field, cols)
    yc         = exact_col(y_field, cols)
    primary_colour = SENTIMENT_COLOURS.get(str(sentiment).upper(), "#3b82f6")

    try:
        if chart_type == "kpi":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Prefer yc if specified and valid
            # But never use a target/budget/quota column as the primary KPI value
            target_kw = ["target","budget","quota","plan","goal"]
            if yc and yc in num_cols and not any(k in yc.lower() for k in target_kw):
                kpi_col = yc
            else:
                # Find first non-target numeric column
                kpi_col = None
                for c in num_cols:
                    if not any(k in c.lower() for k in target_kw):
                        kpi_col = c
                        break
                if not kpi_col:
                    kpi_col = num_cols[0] if num_cols else None
            if not kpi_col:
                return None, "No numeric column for KPI."
            ctype  = col_classifications.get(kpi_col, "value")
            is_pct = ctype == "percentage"
            val    = round(float(df[kpi_col].mean()), 2) if is_pct                      else round(float(df[kpi_col].sum()), 2)
            suffix = "%" if is_pct else ""
            label      = "average" if is_pct else "total"
            fmt_val    = format_number(val, is_pct=is_pct)
            fig    = go.Figure(go.Indicator(
                mode="number",
                value=val,
                title={"text": f"{title}<br><span style='font-size:11px;color:#64748b'>{label}</span>",
                       "font": {"color": "#e2e8f0", "size": 13}},
                number={"font": {"color": primary_colour, "size": 40},
                        "valueformat": ",.1f" if is_pct else ",.0f",
                        "suffix": suffix}
            ))
            # Override Plotly number with formatted value via annotation
            fig.add_annotation(
                text=fmt_val, x=0.5, y=0.45, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=40, color=primary_colour, family="sans-serif"),
                xanchor="center"
            )
            fig.update_traces(visible=False)
            fig.update_layout(**CHART_THEME)
            return fig, None

        if not xc:
            return None, f"Column '{x_field}' not found."
        if not yc:
            if chart_type == "pie":
                counts = df[xc].value_counts().reset_index()
                counts.columns = [xc, "count"]
                fig = px.pie(counts, names=xc, values="count", title=title,
                             color_discrete_sequence=COLOURS)
                fig.update_layout(**CHART_THEME)
                return fig, None
            return None, f"Column '{y_field}' not found."

        if df[yc].dtype not in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]:
            try:
                df     = df.copy()
                df[yc] = pd.to_numeric(df[yc], errors="coerce")
            except Exception:
                return None, f"Column '{y_field}' is not numeric."

        ctype    = col_classifications.get(yc, "value")
        use_mean = ctype == "percentage"
        x_label  = clean_axis_label(xc)
        y_label  = f"Avg {clean_axis_label(yc)}" if use_mean else f"Total {clean_axis_label(yc)}"

        if chart_type == "line":
            g = df.groupby(xc)[yc].mean().reset_index() if use_mean \
                else df.groupby(xc)[yc].sum().reset_index()
            try:
                g = g.sort_values(xc)
            except Exception:
                pass
            g.columns = [xc, y_label]
            fig = px.line(g, x=xc, y=y_label, title=title, markers=True,
                          color_discrete_sequence=[primary_colour],
                          labels={xc: x_label, y_label: y_label})
            theme = dict(CHART_THEME)
            theme["yaxis"] = dict(CHART_THEME["yaxis"], tickformat=",.0f" if not use_mean else ".1f")
            fig.update_layout(**theme)
            return fig, None

        if chart_type == "pie":
            g = df.groupby(xc)[yc].mean().reset_index() if use_mean \
                else df.groupby(xc)[yc].sum().reset_index()
            g = g[g[yc] > 0]
            if g.empty:
                return None, "No positive values for pie chart."
            fig = px.pie(g, names=xc, values=yc, title=title,
                         color_discrete_sequence=COLOURS)
            fig.update_layout(**CHART_THEME)
            return fig, None

        if chart_type == "scatter":
            sample = df[[xc, yc]].dropna().head(500)
            fig    = px.scatter(sample, x=xc, y=yc, title=title,
                                color_discrete_sequence=[primary_colour],
                                labels={xc: x_label, yc: y_label})
            fig.update_traces(marker=dict(size=7, opacity=0.7))
            fig.update_layout(**CHART_THEME)
            return fig, None

        # Default: bar
        g = df.groupby(xc)[yc].mean().reset_index() if use_mean \
            else df.groupby(xc)[yc].sum().reset_index()
        g.columns = [xc, y_label]
        g         = g.sort_values(y_label, ascending=False)
        fig = px.bar(g, x=xc, y=y_label, title=title,
                     color_discrete_sequence=[primary_colour],
                     labels={xc: x_label, y_label: y_label})
        theme = dict(CHART_THEME)
        theme["yaxis"] = dict(CHART_THEME["yaxis"], tickformat=",.0f" if not use_mean else ".1f")
        fig.update_layout(**theme)
        # Only override colour when sentiment is NEGATIVE or URGENT
        # Otherwise all bars stay uniform primary_colour
        if sentiment in ("NEGATIVE", "URGENT"):
            bar_vals = g[y_label].tolist()
            if len(bar_vals) > 1:
                mean_val = sum(bar_vals) / len(bar_vals)
                bar_colours = []
                for bv in bar_vals:
                    if bv <= mean_val * 0.85:
                        bar_colours.append("#ef4444")  # red — significantly below average
                    else:
                        bar_colours.append(primary_colour)
                fig.update_traces(marker_color=bar_colours)
        return fig, None

    except Exception as e:
        return None, f"Rendering error: {str(e)}"

# ─────────────────────────────────────────
# ANALYSIS GENERATOR
# ─────────────────────────────────────────

def generate_analysis(df, role, role_profile, facts, col_classifications, industry, data_label):
    """
    Generate AI analysis with full RIF enforcement.
    Returns parsed analysis dict or raises exception.
    """
    # Cooldown enforcement
    now     = time.time()
    elapsed = now - st.session_state.last_analysis_time
    if elapsed < COOLDOWN_SECS and st.session_state.analysis_count > 0:
        time.sleep(COOLDOWN_SECS - elapsed)

    # Trim facts to token limit — correct priority order
    trimmed_facts = trim_facts_for_token_limit(facts)

    # Feedback context
    feedback_context = ""
    if st.session_state.session_feedback:
        neg = [f["text"] for f in st.session_state.session_feedback
               if f["score"] == "negative" and f.get("text")]
        if neg:
            feedback_context = (f"\nUser feedback from previous analysis: "
                                f"{'; '.join(neg[-2:])}. Address these gaps.")

    # Build user message with complete role profile
    level       = role_profile.get("level", "L3")
    function    = role_profile.get("function", "Executive")
    decision_q  = role_profile.get("decision_question", "")
    allowed_c   = role_profile.get("allowed_charts", ["bar","line","kpi"])
    language    = role_profile.get("language", "plain")
    stats_depth = role_profile.get("stats_depth", "moderate")
    scope_note  = role_profile.get("scope_note", "")
    interp_note = role_profile.get("interpretation_note", "")

    user_message = (
        f"VERIFIED FACTS — Python-computed. ONLY these numbers allowed.\n\n"
        f"{json.dumps(trimmed_facts, indent=2, default=str)}\n\n"
        f"{'='*60}\n"
        f"ROLE PROFILE:\n"
        f"  Raw input:          {role}\n"
        f"  Interpreted title:  {role_profile.get('title', role)}\n"
        f"  Level:              {level}\n"
        f"  Function:           {function}\n"
        f"  Language register:  {language} (max word limits apply)\n"
        f"  Stats depth:        {stats_depth}\n"
        f"  Decision question:  {decision_q}\n"
        f"  Allowed chart types: {', '.join(allowed_c)} ONLY — no other types\n"
        f"  Scope note:         {scope_note or 'None'}\n"
        f"  Interpretation:     {interp_note or 'None'}\n"
        f"  Industry:           {industry or 'Auto-detected'}\n"
        f"  Data source:        {data_label}\n"
        f"{'='*60}\n"
        f"CHART ENFORCEMENT:\n"
        f"  You MUST only use these chart types: {', '.join(allowed_c)}\n"
        f"  Any other chart type will be rejected by the system.\n"
        f"  For level {level}, forbidden types include: "
        f"{'scatter, histogram' if level in ['L1','L2','L3'] else 'kpi' if level == 'L4A' else 'pie, scatter, line'}\n"
        f"{'='*60}\n"
        f"INVESTIGATION INSTRUCTION:\n"
        f"  For every negative finding (declining metric, underperforming segment):\n"
        f"  1. Take the segment name (e.g. 'West', 'Q3', 'Product B').\n"
        f"  2. Look up that EXACT segment name in group_aggregates across ALL other columns.\n"
        f"  3. If any other column shows the same segment performing worse than average — "
        f"state it as evidence: 'West region also shows [metric] of [verified number] vs avg [verified number].'\n"
        f"  4. Only if no correlated data exists: 'Possible cause (not in data): [hypothesis]'\n"
        f"{'='*60}\n"
        f"has_targets={trimmed_facts.get('has_targets', False)}\n"
        f"python_detected_anomalies={json.dumps(trimmed_facts.get('python_detected_anomalies',[]))}\n"
        f"{feedback_context}\n\n"
        f"Generate analysis for: {role_profile.get('title', role)} ({level} · {function})\n"
        f"Answer their decision question: {decision_q}"
    )

    raw    = call_openai([
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user",   "content": user_message}
    ])
    result, parse_err = parse_json_safe(raw)
    if parse_err or result is None:
        raise ValueError(f"Parse error: {parse_err}")
    if "error" in result:
        raise ValueError("Invalid role or injection detected.")

    # Python-side chart type enforcement
    result["charts"] = enforce_chart_rules(result.get("charts", []), role_profile)

    # EF Method 1 — Second AI call for relevance scoring
    result, avg_relevance = run_relevance_scoring(result, role_profile)

    # EF Method 2 — Accuracy validation
    val_results, val_status = run_accuracy_validation(result, facts["verified_stats"])
    result["_py_accuracy"]   = val_results
    result["_py_acc_status"] = val_status

    # EF Method 3 — Completeness
    cov_pct, cov_str, cov_msg = run_completeness_check(result, facts)
    result["_py_coverage"]    = {"pct": cov_pct, "str": cov_str, "msg": cov_msg}

    # EF Method 5 — Confidence overall
    result["_py_confidence"] = compute_confidence_overall(result, df)

    # EF Method 6 — Bias detection
    bias_status, bias_msg, bias_segs = run_bias_detection(result, df)
    result["_py_bias"] = {"status": bias_status, "msg": bias_msg, "segs": bias_segs}

    result["_relevance_avg"] = avg_relevance

    # Audit log
    st.session_state.last_analysis_time = time.time()
    return result

def build_chat_facts(df, col_classifications):
    """Pre-computed facts for chat. Correct aggregation per column type."""
    results  = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col, ctype in col_classifications.items():
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if ctype == "percentage":
            results[f"avg_{col}"]    = round(float(s.mean()), 2)
            results[f"max_{col}"]    = round(float(s.max()), 2)
            results[f"min_{col}"]    = round(float(s.min()), 2)
            results[f"median_{col}"] = round(float(s.median()), 2)
        else:
            results[f"total_{col}"] = round(float(s.sum()), 2)
            results[f"avg_{col}"]   = round(float(s.mean()), 2)
            results[f"max_{col}"]   = round(float(s.max()), 2)
            results[f"min_{col}"]   = round(float(s.min()), 2)
    for cat in cat_cols[:4]:
        for col, ctype in list(col_classifications.items())[:6]:
            try:
                if ctype == "percentage":
                    agg = df.groupby(cat)[col].mean().round(2).to_dict()
                    results[f"avg_{col}_by_{cat}"] = agg
                else:
                    agg = df.groupby(cat)[col].sum().round(2).to_dict()
                    results[f"total_{col}_by_{cat}"] = agg
            except Exception:
                pass
    result_str = json.dumps(results, default=str)
    if len(result_str) > 8000:
        results    = {k: v for k, v in results.items() if not isinstance(v, dict)}
        result_str = json.dumps(results, default=str)
    return result_str

def generate_followup_questions(role, level, question, answer, df_columns, cat_values):
    system = """You are SIMBA AI. Generate exactly 3 follow-up questions as a JSON array.
Rules:
- Questions must suit the role and flow naturally from what was discussed.
- ONLY reference entities that appear in actual_values_in_data.
- Return ONLY: ["Q1?", "Q2?", "Q3?"] — no other text."""
    user = (f"Role: {role} (Level: {level})\n"
            f"Data columns: {', '.join(df_columns[:15])}\n"
            f"actual_values_in_data: {json.dumps(cat_values, default=str)}\n"
            f"Question: {question}\n"
            f"Answer: {answer[:300]}\n"
            f"Generate 3 follow-up questions.")
    try:
        raw       = call_groq(system, user, max_tokens=200)
        raw       = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw.strip())
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str)][:3]
    except Exception:
        pass
    return []

# ─────────────────────────────────────────
# CHAT SUGGESTION GENERATOR + VALIDATOR
# ─────────────────────────────────────────

def generate_python_chat_suggestions(role_profile, facts, df):
    """
    Generate role-specific chat suggestions entirely in Python.
    Every suggestion references a real column name and real segment value.
    Used as fallback AND to validate/replace generic AI suggestions.
    """
    level      = role_profile.get("level", "L3")
    function   = role_profile.get("function", "Executive")
    col_class  = facts.get("column_classifications", {})
    cat_cols   = facts.get("categorical_columns", [])
    num_cols   = list(col_class.keys())
    cat_dists  = facts.get("categorical_distributions", {})
    has_targets = facts.get("has_targets", False)

    suggestions = []

    function_col_hints = {
        "Finance":    ["budget","actual","variance","spend","cost","revenue","profit"],
        "Sales":      ["sales","revenue","units","orders","target","quota"],
        "HR":         ["attrition","headcount","satisfaction","tenure","salary"],
        "Operations": ["defects","efficiency","downtime","cycle","output","produced"],
        "Marketing":  ["impressions","clicks","ctr","roas","cpl","conversion"],
        "Analytical": [],
        "Executive":  [],
    }
    hints    = function_col_hints.get(function, [])
    best_num = None
    for hint in hints:
        for col in num_cols:
            if hint in col.lower():
                best_num = col
                break
        if best_num:
            break
    if not best_num and num_cols:
        best_num = num_cols[0]

    best_cat  = cat_cols[0] if cat_cols else None
    best_cat2 = cat_cols[1] if len(cat_cols) > 1 else None
    best_seg  = None
    if best_cat and best_cat in cat_dists:
        segs = list(cat_dists[best_cat].keys())
        if segs:
            best_seg = segs[0]

    second_num = None
    for col in num_cols:
        if col != best_num:
            second_num = col
            break

    col_label    = best_num.replace("_"," ").title() if best_num else "performance"
    cat_label    = best_cat.replace("_"," ").title() if best_cat else "segment"
    cat2_label   = best_cat2.replace("_"," ").title() if best_cat2 else None
    seg_label    = str(best_seg) if best_seg else None
    second_label = second_num.replace("_"," ").title() if second_num else None

    if level == "L1":
        if best_cat and best_num:
            suggestions.append(f"Which {cat_label} is driving the most risk to overall {col_label}?")
        if has_targets and best_num:
            suggestions.append(f"What is the overall gap between actual and target {col_label} across all {cat_label}s?")
        if second_num:
            suggestions.append(f"Is there a relationship between {col_label} and {second_label} at the organisation level?")
        if best_cat and best_num:
            suggestions.append(f"Which {cat_label} is performing strongest on {col_label} and what is driving it?")

    elif level == "L2":
        if best_cat and best_num:
            suggestions.append(f"What is the {col_label} variance across each {cat_label} and which needs escalation?")
        if has_targets and best_num and best_cat:
            suggestions.append(f"Which {cat_label} is furthest below target on {col_label} this period?")
        if second_num and best_cat:
            suggestions.append(f"How does {second_label} compare across each {cat_label}?")
        if seg_label and best_num:
            suggestions.append(f"What is the trend in {col_label} for {seg_label} compared to the {cat_label} average?")

    elif level == "L3":
        if seg_label and best_num:
            suggestions.append(f"What is the {col_label} for {seg_label} and is it above or below average?")
        if best_cat2 and best_num:
            suggestions.append(f"Which {cat2_label} had the highest {col_label} and what should I do about it?")
        if has_targets and best_num:
            suggestions.append(f"Are we on track to hit our {col_label} target and by how much are we off?")
        if second_num and seg_label:
            suggestions.append(f"What is the {second_label} figure for {seg_label} right now?")

    elif level == "L4F":
        if best_num:
            suggestions.append(f"What is the current {col_label} number?")
        if second_num:
            suggestions.append(f"What is today's {second_label}?")
        if seg_label and best_num:
            suggestions.append(f"What is {col_label} for {seg_label}?")
        suggestions.append(f"What action should I take on {col_label} right now?")

    elif level == "L4A":
        if best_num and second_num:
            suggestions.append(f"What is the correlation between {col_label} and {second_label} and is it statistically significant?")
        if best_cat and best_num:
            suggestions.append(f"What is the distribution of {col_label} across each {cat_label} and are any outliers significant?")
        if best_num:
            suggestions.append(f"What is the standard deviation of {col_label} and how many points fall outside 2 standard deviations?")
        if second_num and best_cat:
            suggestions.append(f"Which {cat_label} has the highest variance in {second_label}?")

    generic_fallbacks = [
        f"What does the {col_label} data tell us overall?",
        f"Which {cat_label} has the best performance on {col_label}?" if cat_label else f"What is the best performing metric?",
        f"What is the average {col_label} across the dataset?",
        f"What is the range of {col_label} values in this dataset?",
    ]
    for fb in generic_fallbacks:
        if len(suggestions) >= 4:
            break
        if fb not in suggestions:
            suggestions.append(fb)

    return suggestions[:4]


def validate_and_fix_suggestions(ai_suggestions, facts, role_profile, df):
    """
    Validate AI suggestions. Replace any generic ones with Python-generated specific ones.
    Returns 4 validated suggestions always referencing real columns or segments.
    """
    col_names_lower = [c.lower() for c in facts.get("exact_column_names", [])]
    cat_dists       = facts.get("categorical_distributions", {})

    real_segments = set()
    for col, dist in cat_dists.items():
        for val in dist.keys():
            real_segments.add(str(val).lower())

    def is_specific(suggestion):
        s_lower = suggestion.lower()
        for col in col_names_lower:
            col_words = col.replace("_"," ").split()
            if any(word in s_lower for word in col_words if len(word) > 3):
                return True
        for seg in real_segments:
            if seg in s_lower and len(seg) > 2:
                return True
        return False

    generic_phrases = [
        "what are the risks", "what should i prioritise", "biggest risk",
        "what is the trend overall", "overall performance", "key findings",
        "what can you tell me", "summarise", "summary of", "give me an overview",
        "what should i focus", "top priorities"
    ]

    validated   = []
    python_subs = generate_python_chat_suggestions(role_profile, facts, df)
    sub_idx     = 0

    for sug in (ai_suggestions or []):
        sug_lower = sug.lower()
        is_gen    = any(phrase in sug_lower for phrase in generic_phrases)
        if is_specific(sug) and not is_gen:
            validated.append(sug)
        else:
            if sub_idx < len(python_subs):
                validated.append(python_subs[sub_idx])
                sub_idx += 1

    while len(validated) < 4 and sub_idx < len(python_subs):
        validated.append(python_subs[sub_idx])
        sub_idx += 1

    return validated[:4]


# ═══════════════════════════════════════════════════════
# COMPARISON MODE ENGINE
# ═══════════════════════════════════════════════════════

def generate_comparison(facts_a, facts_b, label_a, label_b, role_profile):
    """Generate AI comparison between two datasets."""
    level    = role_profile.get("level", "L3")
    function = role_profile.get("function", "Executive")
    decision = role_profile.get("decision_question", "")

    # Build comparison user message
    user_message = (
        f"DATASET A: {label_a}\n"
        f"VERIFIED FACTS A:\n{json.dumps(trim_facts_for_token_limit(facts_a, 6000), indent=2, default=str)}\n\n"
        f"DATASET B: {label_b}\n"
        f"VERIFIED FACTS B:\n{json.dumps(trim_facts_for_token_limit(facts_b, 6000), indent=2, default=str)}\n\n"
        f"{'='*60}\n"
        f"ROLE PROFILE:\n"
        f"  Level:             {level}\n"
        f"  Function:          {function}\n"
        f"  Decision question: {decision}\n"
        f"{'='*60}\n"
        f"Compare Dataset A vs Dataset B for a {level} {function}.\n"
        f"Use ONLY numbers from verified_stats in each dataset.\n"
        f"For each changed metric: compute change = B value minus A value, "
        f"change_pct = (B-A)/A * 100.\n"
        f"Look in group_aggregates for segment-level drivers of major changes.\n"
        f"Answer: {decision}"
    )

    raw    = call_openai([
        {"role": "system", "content": COMPARISON_PROMPT},
        {"role": "user",   "content": user_message}
    ], max_tokens=3000)
    result, parse_err = parse_json_safe(raw)
    if parse_err or result is None:
        raise ValueError(f"Comparison parse error: {parse_err}")
    return result

def compute_comparison_metrics(facts_a, facts_b):
    """Python-computed comparison — change values computed here, not by AI."""
    stats_a = facts_a.get("verified_stats", {})
    stats_b = facts_b.get("verified_stats", {})
    common  = set(stats_a.keys()) & set(stats_b.keys())

    improved  = []
    declined  = []
    stable    = []

    for col in common:
        val_a = stats_a[col].get("correct_aggregate", 0)
        val_b = stats_b[col].get("correct_aggregate", 0)
        ctype = stats_a[col].get("type", "value")
        label = stats_a[col].get("correct_aggregate_label", "total")

        if val_a == 0:
            continue
        change_pct = round((val_b - val_a) / abs(val_a) * 100, 1)
        entry = {
            "metric":     col,
            "value_a":    val_a,
            "value_b":    val_b,
            "change_pct": change_pct,
            "type":       ctype,
            "label":      label,
        }
        if change_pct > 2:
            improved.append(entry)
        elif change_pct < -2:
            declined.append(entry)
        else:
            stable.append(entry)

    improved.sort(key=lambda x: x["change_pct"], reverse=True)
    declined.sort(key=lambda x: x["change_pct"])
    return improved, declined, stable

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🦁 SIMBA AI")
    st.caption("AI Intelligence Platform")
    st.divider()

    if st.session_state.step == "dashboard" and st.session_state.analysis:
        st.markdown("**Session**")
        rp = st.session_state.role_profile
        st.caption(f"Role: {rp.get('title', st.session_state.role)}")
        st.caption(f"Level: {rp.get('level','—')} · {rp.get('function','—')}")
        st.caption(f"Data: {st.session_state.data_label}")
        analyses_left = MAX_ANALYSES_PER_SESSION - st.session_state.analysis_count
        st.caption(f"Analyses: {st.session_state.analysis_count}/{MAX_ANALYSES_PER_SESSION} ({analyses_left} remaining)")
        st.divider()

        st.markdown("**Switch Role**")
        new_role = st.text_input("New role", placeholder="e.g. CFO, Store Manager...",
                                 label_visibility="collapsed")
        if st.button("🔄 Regenerate for New Role", use_container_width=True):
            if new_role.strip():
                new_profile = detect_role_profile(new_role.strip(),
                                                  list(st.session_state.df.columns))
                if new_profile.get("error") == "injection":
                    st.error("Invalid input.")
                else:
                    st.session_state.role           = new_role.strip()
                    st.session_state.role_profile   = new_profile
                    st.session_state.analysis       = None
                    st.session_state.chat_history   = []
                    st.session_state.show_chat      = False
                    st.session_state.feedback_given = False
                    st.rerun()
        st.divider()

        # EF Method 4 — Feedback
        if not st.session_state.feedback_given:
            st.markdown("**Was this analysis useful?**")
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("👍 Yes", use_container_width=True):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_score = "positive"
                    st.session_state.session_feedback.append({"score":"positive","text":""})
                    st.rerun()
            with fc2:
                if st.button("👎 No", use_container_width=True):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_score = "negative"
                    st.session_state.feedback_count_neg += 1
                    st.rerun()
            if st.session_state.feedback_score == "negative" and st.session_state.feedback_given:
                fb_text = st.text_area("What was missing or incorrect?",
                                       placeholder="Tell us what to improve...", height=80)
                if st.button("Submit Feedback", use_container_width=True):
                    st.session_state.feedback_text = fb_text
                    st.session_state.session_feedback.append({"score":"negative","text":fb_text})
                    st.rerun()
        else:
            if st.session_state.feedback_score == "positive":
                st.success("Thanks for the feedback!")
            else:
                st.warning("Thanks — noted for improvement.")
            if st.session_state.feedback_count_neg >= 2:
                st.info("💡 Consider switching role or uploading a different dataset.")

        st.divider()
        if st.button("↩ Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            # Re-initialise defaults so no stale comparison data remains
            st.rerun()

# ═══════════════════════════════════════════════════════
# STEP 1 — LANDING
# ═══════════════════════════════════════════════════════
if st.session_state.step == "landing":
    n1, n2 = st.columns([1, 5])
    with n1:
        st.markdown("### 🦁 SIMBA AI")
    with n2:
        st.caption("AI Business Intelligence Platform · v1.0")
    st.markdown("---")

    _, hero, _ = st.columns([1, 3, 1])
    with hero:
        st.markdown(
            "<h1 style='text-align:center;font-size:2.4rem;font-weight:600;line-height:1.25;'>"
            "Business Intelligence<br>"
            "<span style='background:linear-gradient(135deg,#3b82f6,#6366f1);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "background-clip:text;'>Built for Your Role</span></h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;color:#64748b;font-size:1rem;"
            "line-height:1.7;margin-top:12px;'>"
            "Upload any dataset. State your role.<br>"
            "Get a complete AI-generated dashboard — tailored specifically "
            "to how you make decisions.</p>",
            unsafe_allow_html=True)
        st.markdown("")

    _, pn, _ = st.columns([1, 3, 1])
    with pn:
        with st.container(border=True):
            st.markdown("##### 🔒 Data Privacy Notice")
            st.markdown("""
Your data is processed in this session only. Nothing is stored permanently on any server.
Analysis is powered by OpenAI — please ensure your data complies with your organisation's
data sharing and privacy policy before uploading.

- **Session only** — data cleared when session ends
- **Secure** — all transmission over HTTPS
- **PII detection** — personal data flagged before analysis
- **Subprocessor** — OpenAI processes data under their API terms
            """)
            st.markdown("")
            if st.button("✅ I Understand — Get Started", type="primary",
                         use_container_width=True):
                st.session_state.step = "data"
                st.rerun()
        st.caption("No login required · Built by Surya · SIMBA AI v1.0")

# ═══════════════════════════════════════════════════════
# STEP 2 — DATA SOURCE
# ═══════════════════════════════════════════════════════
elif st.session_state.step == "data":
    st.markdown("## 📂 Select Your Data Source")
    st.caption("Upload a file, connect to a database, or upload two datasets to compare.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📁 Upload a File", "🗄️ Connect to Database", "🔀 Comparison Mode"])

    with tab1:
        st.markdown("#### Upload CSV or Excel")
        st.caption(f"Supported: .csv · .xlsx · .xls — Max {MAX_FILE_SIZE_MB}MB")
        uploaded_file = st.file_uploader("Drop your file here",
                                         type=["csv","xlsx","xls"],
                                         label_visibility="collapsed")
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File is {file_size_mb:.1f}MB. Maximum allowed is {MAX_FILE_SIZE_MB}MB.")
            else:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") \
                         else pd.read_excel(uploaded_file)
                    if df.empty or len(df.columns) == 0:
                        st.error("This file appears to be empty or has no readable columns.")
                    elif len(df) == 0:
                        st.error("This file contains column headers but no data rows.")
                    else:
                        st.success(f"✅ **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
                        st.dataframe(df.head(5), use_container_width=True)
                        if st.button("Continue with this file →", type="primary",
                                     use_container_width=True):
                            st.session_state.df              = df
                            st.session_state.data_source     = "file"
                            st.session_state.data_label      = uploaded_file.name
                            st.session_state.comparison_mode = False
                            st.session_state.step            = "quality"
                            st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    with tab2:
        sub = st.session_state.db_sub_step
        if sub == "credentials":
            st.markdown("#### Connect to Supabase")
            st.caption("Credentials used for this session only — never stored.")
            with st.form("sb_form"):
                sb_url    = st.text_input("Project URL", placeholder="https://xxxx.supabase.co")
                sb_key    = st.text_input("API Key", type="password")
                st.caption("Found in Supabase → Settings → API Keys")
                submitted = st.form_submit_button("🔗 Connect & Discover Tables",
                                                  type="primary", use_container_width=True)
            if submitted:
                if not sb_url.strip() or not sb_key.strip():
                    st.error("Please enter both URL and API Key.")
                else:
                    with st.spinner("Connecting..."):
                        tables, err = sb_get_tables(sb_url.strip(), sb_key.strip())
                    if err:
                        st.error(f"Connection failed: {err}")
                    elif not tables:
                        st.error("Connected but no tables found.")
                    else:
                        st.session_state.db_sb_url   = sb_url.strip()
                        st.session_state.db_sb_key   = sb_key.strip()
                        st.session_state.db_tables   = tables
                        st.session_state.db_sub_step = "select_table"
                        st.rerun()
        elif sub == "select_table":
            st.success(f"✅ Connected · {len(st.session_state.db_tables)} table(s) found")
            selected = st.selectbox("Available tables",
                                    options=st.session_state.db_tables,
                                    label_visibility="collapsed")
            if st.button("🔍 Preview Table", use_container_width=True):
                with st.spinner("Loading..."):
                    df_prev, err = sb_load_table(st.session_state.db_sb_url,
                                                 st.session_state.db_sb_key, selected)
                if err:
                    st.error(f"Could not load: {err}")
                else:
                    if df_prev is None or df_prev.empty:
                        st.error("Table is empty.")
                    else:
                        st.success(f"**{selected}** — {len(df_prev):,} rows × {len(df_prev.columns)} cols")
                        st.dataframe(df_prev.head(5), use_container_width=True)
                        st.session_state["db_preview_df"]    = df_prev
                        st.session_state["db_preview_table"] = selected
            if "db_preview_df" in st.session_state:
                if st.button("Continue with this table →", type="primary",
                             use_container_width=True):
                    st.session_state.df              = st.session_state["db_preview_df"]
                    st.session_state.data_source     = "database"
                    st.session_state.data_label      = f"Supabase · {st.session_state['db_preview_table']}"
                    st.session_state.comparison_mode = False
                    st.session_state.step            = "quality"
                    st.rerun()
            if st.button("← Change credentials"):
                st.session_state.db_sub_step = "credentials"
                st.session_state.db_tables   = []
                if "db_preview_df" in st.session_state:
                    del st.session_state["db_preview_df"]
                st.rerun()

    with tab3:
        st.markdown("#### Comparison Mode")
        st.caption("Upload two datasets to compare performance — this month vs last month, Team A vs Team B, Branch 1 vs Branch 2.")
        st.markdown("")
        st.markdown("**Dataset A** (baseline — e.g. last month, Team A)")
        file_a = st.file_uploader("Upload Dataset A", type=["csv","xlsx","xls"],
                                  key="comp_file_a", label_visibility="collapsed")
        st.markdown("**Dataset B** (comparison — e.g. this month, Team B)")
        file_b = st.file_uploader("Upload Dataset B", type=["csv","xlsx","xls"],
                                  key="comp_file_b", label_visibility="collapsed")

        if file_a and file_b:
            try:
                df_a = pd.read_csv(file_a) if file_a.name.endswith(".csv") else pd.read_excel(file_a)
                df_b = pd.read_csv(file_b) if file_b.name.endswith(".csv") else pd.read_excel(file_b)
                ca, cb = st.columns(2)
                with ca:
                    st.success(f"✅ A: {file_a.name} — {len(df_a):,} rows")
                    st.dataframe(df_a.head(3), use_container_width=True)
                with cb:
                    st.success(f"✅ B: {file_b.name} — {len(df_b):,} rows")
                    st.dataframe(df_b.head(3), use_container_width=True)
                if st.button("🔀 Compare These Datasets →", type="primary",
                             use_container_width=True):
                    st.session_state.df              = df_a
                    st.session_state.df2             = df_b
                    st.session_state.data_label      = file_a.name
                    st.session_state.data_label2     = file_b.name
                    st.session_state.data_source     = "file"
                    st.session_state.comparison_mode = True
                    st.session_state.step            = "quality"
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read files: {e}")

# ═══════════════════════════════════════════════════════
# STEP 3 — DATA QUALITY
# ═══════════════════════════════════════════════════════
elif st.session_state.step == "quality":
    df = st.session_state.df
    if df is None:
        st.error("Session data lost. Please start over.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.stop()

    st.markdown("## 📋 Data Quality Report")
    if st.session_state.comparison_mode:
        st.caption(f"Checking Dataset A: {st.session_state.data_label}")
    else:
        st.caption(f"Source: {st.session_state.data_label}")
    st.markdown("---")

    total_rows    = len(df)
    missing       = df.isnull().sum()
    total_missing = int(missing.sum())
    duplicates    = int(df.duplicated().sum())
    pii_cols      = detect_pii(df)
    date_issues   = check_date_format_issues(df)
    type_issues   = check_type_inconsistencies(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{total_rows:,}")
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Values", total_missing)
    c4.metric("Duplicate Rows", duplicates)
    st.markdown("---")

    if total_missing > 0:
        st.markdown("**⚠️ Missing Values by Column**")
        m         = missing[missing > 0].reset_index()
        m.columns = ["Column", "Missing Count"]
        m["% of Rows"] = (m["Missing Count"] / total_rows * 100).round(1).astype(str) + "%"
        st.dataframe(m, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values detected.")

    if duplicates > 0:
        st.warning(f"⚠️ {duplicates} duplicate row(s) detected.")
    else:
        st.success("✅ No duplicate rows detected.")

    if date_issues:
        for issue in date_issues:
            st.warning(f"⚠️ Date format issue: {issue}")
    else:
        st.success("✅ No date format issues detected.")

    if type_issues:
        for issue in type_issues:
            st.warning(f"⚠️ Data type issue: {issue}")
    else:
        st.success("✅ No data type inconsistencies detected.")

    if pii_cols:
        st.error(f"🔴 PII Detected in: **{', '.join(pii_cols)}**")
        st.warning("**Data Protection Notice:** Personal information detected. "
                   "Ensure you have the legal right to process this data before proceeding.")
    else:
        st.success("✅ No PII detected.")

    st.markdown("**Column Overview**")
    st.dataframe(pd.DataFrame({
        "Column":        df.columns,
        "Type":          [str(df[c].dtype) for c in df.columns],
        "Non-Null":      [int(df[c].count()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns]
    }), use_container_width=True, hide_index=True)

    issues = sum([total_missing > 0, duplicates > 0, bool(pii_cols) * 2,
                  bool(date_issues), bool(type_issues)])
    qs     = max(0, 100 - (issues * 15))
    st.markdown(f"**Overall Data Quality Score: {qs}/100**")
    st.progress(qs / 100)
    st.markdown("---")

    st.markdown("**Please confirm you have reviewed this report before proceeding.**")
    acknowledged = st.checkbox("I have reviewed this data quality report and wish to proceed.")
    if acknowledged:
        if st.button("✅ Proceed to Analysis →", type="primary", use_container_width=True):
            st.session_state.quality_acknowledged = True
            st.session_state.step = "industry"
            st.rerun()
    else:
        st.info("Please check the box above to confirm you have reviewed the report.")

# ═══════════════════════════════════════════════════════
# STEP 3b — INDUSTRY DETECTION
# ═══════════════════════════════════════════════════════
elif st.session_state.step == "industry":
    df = st.session_state.df
    if df is None:
        st.error("Session data lost. Please start over.")
        st.stop()

    st.markdown("## 🏭 Industry & Domain Detection")
    st.caption("SIMBA AI has analysed your dataset and detected the following business domains.")
    st.markdown("---")

    if not st.session_state.detected_domains:
        with st.spinner("Detecting business domains from your data..."):
            domains = detect_industry_from_data(df)
            st.session_state.detected_domains = domains

    detected = st.session_state.detected_domains
    st.markdown("**We detected the following domain(s) in your dataset:**")
    for d in detected:
        st.markdown(f"- ✅ {d}")

    st.markdown("")
    st.markdown("**Is this correct?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, this is correct", type="primary", use_container_width=True):
            st.session_state.industry           = ", ".join(detected)
            st.session_state.industry_confirmed = True
            st.session_state.step               = "role"
            st.rerun()
    with col2:
        if st.button("✏️ No, let me specify", use_container_width=True):
            st.session_state.industry_confirmed = False

    if st.session_state.get("industry_confirmed") is False:
        st.markdown("")
        override = st.text_input("Describe your industry or domain",
                                 placeholder="e.g. Healthcare, Logistics, Retail Banking...")
        if st.button("Confirm and Continue →", use_container_width=True):
            if override.strip():
                st.session_state.industry           = override.strip()
                st.session_state.industry_confirmed = True
                st.session_state.step               = "role"
                st.rerun()
            else:
                st.warning("Please enter your industry before continuing.")

# ═══════════════════════════════════════════════════════
# STEP 4 — ROLE INPUT
# ═══════════════════════════════════════════════════════
elif st.session_state.step == "role":
    if st.session_state.session_locked:
        st.error("🔒 Session locked due to repeated invalid inputs. Please refresh to start a new session.")
        st.stop()

    st.markdown("## 👤 Who Are You?")
    st.markdown("Tell SIMBA AI your role — the entire analysis is built specifically for you.")
    st.markdown("---")

    st.markdown("**Your Role**")
    st.caption("Type your exact role. Any role works — CFO, Regional Sales Manager, Data Scientist, Store Manager...")
    role_input = st.text_input("Role",
        placeholder="e.g. CFO, Regional Sales Manager, Head of Finance, Data Analyst...",
        label_visibility="collapsed")

    if role_input.strip():
        # Preview role profile before submission
        preview_profile = detect_role_profile(role_input.strip(),
                                              list(st.session_state.df.columns) if st.session_state.df is not None else [])
        if preview_profile.get("error") == "injection":
            st.session_state.injection_attempts += 1
            remaining = 3 - st.session_state.injection_attempts
            st.error(f"⚠️ This input cannot be processed. Please enter a valid role description. "
                     f"({remaining} attempt(s) remaining before session lock.)")
            if st.session_state.injection_attempts >= 3:
                st.session_state.session_locked = True
                st.rerun()
        elif preview_profile.get("error") == "gibberish":
            st.warning("We couldn't recognise that role. Could you describe what you do in a sentence? "
                       "For example: I oversee regional sales, or I manage the finance team.")
        else:
            # Show role interpretation preview
            with st.container(border=True):
                st.caption("**Role interpretation preview:**")
                rc1, rc2, rc3 = st.columns(3)
                rc1.markdown(f"**Level:** {preview_profile.get('level','—')}")
                rc2.markdown(f"**Function:** {preview_profile.get('function','—')}")
                rc3.markdown(f"**Scope:** {preview_profile.get('scope','—')}")
                if preview_profile.get("interpretation_note"):
                    st.caption(f"ℹ️ {preview_profile['interpretation_note']}")

            if st.session_state.analysis_count >= MAX_ANALYSES_PER_SESSION:
                st.warning(f"Session limit of {MAX_ANALYSES_PER_SESSION} analyses reached.")
            else:
                if st.button("🚀 Generate My Dashboard", type="primary",
                             use_container_width=True):
                    st.session_state.role         = role_input.strip()
                    st.session_state.role_profile = preview_profile
                    st.session_state.analysis     = None
                    st.session_state.verified_facts = None
                    st.session_state.step         = "dashboard"
                    st.rerun()

# ═══════════════════════════════════════════════════════
# STEP 5 — DASHBOARD
# ═══════════════════════════════════════════════════════
elif st.session_state.step == "dashboard":
    df = st.session_state.df
    if df is None:
        st.error("Session data lost. Please start over.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.stop()

    role         = st.session_state.role
    role_profile = st.session_state.role_profile
    # Safety: if role_profile is empty (e.g. session restored mid-flow), rebuild it
    if not role_profile.get("level"):
        role_profile = detect_role_profile(role, list(df.columns) if df is not None else [])
        st.session_state.role_profile = role_profile
    is_comparison = st.session_state.comparison_mode

    # ── COMPUTE VERIFIED FACTS ──
    if st.session_state.verified_facts is None:
        facts, col_classifications           = build_verified_facts(df)
        st.session_state.verified_facts      = facts
        st.session_state.col_classifications = col_classifications
    else:
        facts               = st.session_state.verified_facts
        col_classifications = st.session_state.col_classifications

    # ── COMPARISON MODE ──
    if is_comparison:
        df2 = st.session_state.df2
        if df2 is not None and st.session_state.verified_facts2 is None:
            facts2, col_class2                    = build_verified_facts(df2)
            st.session_state.verified_facts2      = facts2
            st.session_state.col_classifications2 = col_class2

        facts2     = st.session_state.verified_facts2
        label_a    = st.session_state.data_label
        label_b    = st.session_state.data_label2

        # Top bar
        st.markdown("### 🦁 SIMBA AI — Comparison Mode")
        st.caption(
            f"Role: **{role_profile.get('title', role)}** · "
            f"{role_profile.get('level','')} · {role_profile.get('function','')} · "
            f"Comparing **{label_a}** vs **{label_b}**"
        )
        st.markdown("---")

        # ── PYTHON COMPARISON METRICS (always accurate) ──
        improved, declined, stable = compute_comparison_metrics(facts, facts2)

        # ── AI COMPARISON ──
        if st.session_state.analysis is None:
            with st.spinner("🧠 SIMBA AI is comparing your datasets..."):
                try:
                    comp_result = generate_comparison(
                        facts, facts2, label_a, label_b, role_profile
                    )
                    st.session_state.analysis        = comp_result
                    st.session_state.analysis_count += 1
                    add_audit_entry(role, f"{label_a} vs {label_b}",
                                    st.session_state.analysis_count)
                except Exception as e:
                    st.error(f"Comparison analysis failed: {e}")
                    st.stop()

        comp = st.session_state.analysis

        # ── COMPARISON SUMMARY ──
        cs = comp.get("comparison_summary", {})
        with st.container(border=True):
            st.markdown(f"#### 📌 Comparison Summary — {role_profile.get('title', role)}")
            for i, key in enumerate(["sentence_1","sentence_2","sentence_3"], 1):
                s = clean_ai_text(cs.get(key, ""))
                if s:
                    st.markdown(f"**{i}.** {s}")

        # ── PYTHON-COMPUTED CHANGES ──
        st.markdown("#### 📊 Python-Verified Changes")
        st.caption("All change values computed by Python — not AI.")

        if improved or declined:
            imp_col, dec_col = st.columns(2)
            with imp_col:
                st.markdown("**📈 Improved**")
                for item in improved[:5]:
                    with st.container(border=True):
                        st.markdown(f"🟢 **{item['metric'].replace('_',' ').title()}**")
                        st.markdown(f"A: {item['value_a']:,.2f} → B: {item['value_b']:,.2f}")
                        st.caption(f"+{item['change_pct']:.1f}% change")
            with dec_col:
                st.markdown("**📉 Declined**")
                for item in declined[:5]:
                    with st.container(border=True):
                        st.markdown(f"🔴 **{item['metric'].replace('_',' ').title()}**")
                        st.markdown(f"A: {item['value_a']:,.2f} → B: {item['value_b']:,.2f}")
                        st.caption(f"{item['change_pct']:.1f}% change")

        if stable:
            with st.expander(f"➡️ Stable metrics ({len(stable)} found)", expanded=False):
                for item in stable:
                    st.caption(f"**{item['metric'].replace('_',' ').title()}** — "
                               f"A: {item['value_a']:,.2f} | B: {item['value_b']:,.2f} | "
                               f"Change: {item['change_pct']:+.1f}%")

        # ── AI SIGNIFICANT CHANGES + RECOMMENDATIONS ──
        sig_changes = comp.get("significant_changes", [])
        if sig_changes:
            st.markdown("#### 🔍 Significant Changes — AI Analysis")
            for sc in sig_changes:
                with st.container(border=True):
                    st.markdown(f"**{sc.get('metric','').replace('_',' ').title()}** — {sc.get('finding','')}")
                    if sc.get("root_cause_hypothesis"):
                        st.caption(f"💭 Hypothesis: {sc['root_cause_hypothesis']}")

        recs = comp.get("recommendations", [])
        if recs:
            st.markdown("#### ✅ Recommendations")
            for r in recs:
                p_icon = {1:"🔴",2:"🟡",3:"🟢"}.get(r.get("priority",3),"🔵")
                with st.container(border=True):
                    st.markdown(f"{p_icon} **P{r.get('priority','')} · {r.get('action','')}**")
                    if r.get("evidence"):
                        st.caption(f"📊 Evidence: {r['evidence']}")
                    st.caption(f"{r.get('timeframe','')}")

        # ── NARRATIVE ──
        narrative = comp.get("narrative", {})
        if narrative:
            with st.expander("📝 Comparison Narrative", expanded=True):
                if narrative.get("opening"):
                    st.markdown(narrative["opening"])
                for p in narrative.get("body", []):
                    st.markdown(p)
                if narrative.get("close"):
                    st.markdown(narrative["close"])

        # ── SIDE BY SIDE STATS ──
        with st.expander("📐 Side-by-Side Statistical Summary", expanded=False):
            st.caption("Python-computed. All numbers verified.")
            stat_a = build_stat_summary_table(df,  col_classifications)
            stat_b = build_stat_summary_table(df2, st.session_state.col_classifications2)
            sa_col, sb_col = st.columns(2)
            with sa_col:
                st.markdown(f"**{label_a}**")
                st.dataframe(stat_a, use_container_width=True, hide_index=True)
            with sb_col:
                st.markdown(f"**{label_b}**")
                st.dataframe(stat_b, use_container_width=True, hide_index=True)

        if st.button("🔄 Re-run Comparison", use_container_width=True):
            st.session_state.analysis = None
            st.rerun()

        st.stop()

    # ══════════════════════════════════════════
    # STANDARD DASHBOARD (non-comparison)
    # ══════════════════════════════════════════

    # ── GENERATE ANALYSIS ──
    if st.session_state.analysis is None:
        with st.spinner("🧠 SIMBA AI is analysing your data..."):
            try:
                result = generate_analysis(
                    df, role, role_profile, facts, col_classifications,
                    st.session_state.industry, st.session_state.data_label
                )
                st.session_state.analysis        = result
                st.session_state.analysis_count += 1
                add_audit_entry(role, st.session_state.data_label,
                                st.session_state.analysis_count)
                # Validate AI suggestions — replace generic ones with Python-generated role-specific ones
                raw_suggestions = result.get("chat_suggestions", [])
                st.session_state.chat_suggestions = validate_and_fix_suggestions(
                    raw_suggestions, facts, role_profile, df
                )
            except Exception as e:
                st.warning("⚠️ AI analysis is temporarily unavailable. Showing verified statistical summary.")
                stat_df = build_stat_summary_table(df, col_classifications)
                st.dataframe(stat_df, use_container_width=True, hide_index=True)
                st.stop()

    analysis = st.session_state.analysis

    # ── TOP BAR ──
    tb1, tb2, tb3 = st.columns([4, 2, 1])
    with tb1:
        st.markdown("### 🦁 SIMBA AI Dashboard")
        st.caption(
            f"Role: **{analysis.get('role_interpreted', role)}** · "
            f"{analysis.get('level','')} · {analysis.get('function','')} · "
            f"{st.session_state.data_label}"
        )
    with tb2:
        if analysis.get("interpretation_note"):
            st.info(f"ℹ️ {analysis['interpretation_note']}")
    with tb3:
        label = "💬 Hide Chat" if st.session_state.show_chat else "💬 Ask AI"
        if st.button(label, use_container_width=True):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun()

    st.markdown("---")

    if st.session_state.show_chat:
        dash_col, chat_col = st.columns([7, 3])
    else:
        dash_col = st.container()
        chat_col = None

    with dash_col:

        # ── EXECUTIVE SUMMARY ──
        es = analysis.get("executive_summary", {})
        with st.container(border=True):
            st.markdown(f"#### 📌 Executive Summary — {analysis.get('role_interpreted', role)}")
            for i, key in enumerate(["sentence_1","sentence_2","sentence_3"], 1):
                s = clean_ai_text(es.get(key, ""))
                if s:
                    st.markdown(f"**{i}.** {s}")

        # ── EF COVERAGE + BIAS ──
        cov_info = analysis.get("_py_coverage", {})
        if cov_info.get("msg"):
            if cov_info["pct"] >= 90:
                st.success(cov_info["msg"])
            elif cov_info["pct"] >= 60:
                st.info(cov_info["msg"])
            else:
                st.warning(cov_info["msg"])

        bias_info = analysis.get("_py_bias", {})
        if bias_info.get("status") == "IMBALANCED":
            st.warning(bias_info.get("msg", ""))

        # ── TRAFFIC LIGHTS ──
        tl_list = analysis.get("traffic_lights", [])
        if tl_list:
            st.markdown("#### 🚦 Key Metrics")
            cols_n  = min(len(tl_list), 3)
            tl_cols = st.columns(cols_n)
            for i, tl in enumerate(tl_list):
                icon = {"GREEN":"🟢","AMBER":"🟡","RED":"🔴"}.get(tl.get("status",""),"⚪")
                with tl_cols[i % cols_n]:
                    with st.container(border=True):
                        st.markdown(f"{icon} **{tl.get('metric','')}**")
                        raw_val = tl.get("value","")
                        # Try to format as number if possible
                        try:
                            nums = re.findall(r"[\d]+\.?\d*", str(raw_val).replace(",",""))
                            if nums:
                                num_val = float(nums[0])
                                # Detect if percentage from metric name or value
                                is_pct_val = "%" in str(raw_val)
                                fmt_val = format_number(num_val, is_pct=is_pct_val)
                                # Preserve any suffix text after the number
                                suffix_text = re.sub(r"[\d,\.]+", "", str(raw_val)).strip()
                                display_val = fmt_val + (f" {suffix_text}" if suffix_text and suffix_text != "%" else "")
                            else:
                                display_val = raw_val
                        except Exception:
                            display_val = raw_val
                        st.markdown(f"### {display_val}")
                        st.caption(clean_ai_text(tl.get("reason","")))
                        if tl.get("target_note"):
                            st.caption(f"ℹ️ {tl['target_note']}")

        # ── ANOMALIES ──
        py_anomalies = facts.get("python_detected_anomalies", [])
        ai_anomalies = analysis.get("anomalies", [])

        # Deduplicate by column name — more reliable than text prefix matching
        py_cols    = {a.get("column","").lower() for a in py_anomalies}
        deduped_ai = [a for a in ai_anomalies
                      if a.get("metric","").lower() not in py_cols]
        all_anomalies = py_anomalies + deduped_ai

        if all_anomalies:
            high   = [a for a in all_anomalies if a.get("severity") == "HIGH"]
            medium = [a for a in all_anomalies if a.get("severity") == "MEDIUM"]
            low    = [a for a in all_anomalies if a.get("severity") == "LOW"]
            st.markdown("#### ⚠️ Anomaly Alerts")
            for a in high:
                text = clean_ai_text(a.get("finding") or a.get("description",""))
                with st.container(border=True):
                    st.markdown(f"🔴 **Requires Attention** — {text}")
            for a in medium:
                text = clean_ai_text(a.get("finding") or a.get("description",""))
                with st.container(border=True):
                    st.markdown(f"🟡 **Monitor Closely** — {text}")
            for a in low:
                text = clean_ai_text(a.get("finding") or a.get("description",""))
                with st.container(border=True):
                    st.markdown(f"🔵 **Note** — {text}")

        # ── CHARTS ──
        charts      = analysis.get("charts", [])
        acc_results = analysis.get("_py_accuracy", {})

        if charts:
            st.markdown("#### 📊 AI-Generated Charts")
            for i in range(0, len(charts), 2):
                cl, cr = st.columns(2)
                for j, container in enumerate([cl, cr]):
                    idx = i + j
                    if idx < len(charts):
                        ch       = charts[idx]
                        fig, err = render_chart(ch, df, col_classifications)
                        with container:
                            with st.container(border=True):
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True,
                                                    config={"displayModeBar": False})
                                else:
                                    st.info(f"📊 {err}")
                                # Relevance label if present
                                rel_label = ch.get("_relevance_label","")
                                if rel_label:
                                    st.caption(rel_label)
                                st.caption(f"💡 {clean_ai_text(ch.get('caption',''))}")
                                b1, b2, b3 = st.columns(3)
                                with b1:
                                    sent = ch.get("sentiment","")
                                    icon = {"POSITIVE":"📈","NEGATIVE":"📉",
                                            "URGENT":"⚡","NEUTRAL":"➡️"}.get(sent,"➡️")
                                    st.caption(f"{icon} {sent}")
                                with b2:
                                    conf = ch.get("confidence","")
                                    icon = {"HIGH":"🔵","MEDIUM":"🟡","INDICATIVE":"⚪"}.get(conf,"⚪")
                                    st.caption(f"{icon} {conf}")
                                with b3:
                                    # Verified badge — check if any accuracy result passed for this chart's columns
                                    chart_cols = [ch.get("x_field","").lower(), ch.get("y_field","").lower()]
                                    verified = any(
                                        v.get("passed") for k, v in acc_results.items()
                                        if any(cc in k.lower() for cc in chart_cols if cc)
                                    )
                                    if verified:
                                        st.caption("✅ Verified")

        # ── EF METHOD 5 — CONFIDENCE OVERALL ──
        conf_overall = analysis.get("_py_confidence","")
        if conf_overall:
            conf_labels = {
                "HIGH":       "🔵 High Confidence — Strong statistical evidence. Act with confidence.",
                "MEDIUM":     "🟡 Medium Confidence — Worth monitoring. Investigate before acting.",
                "INDICATIVE": "⚪ Indicative Only — Early signal. Gather more data before acting."
            }
            st.caption(f"**Overall Confidence:** {conf_labels.get(conf_overall, conf_overall)}")

        # ── STATISTICAL SUMMARY ──
        stat_df = build_stat_summary_table(df, col_classifications)
        with st.expander("📐 Statistical Summary — Python Verified", expanded=False):
            st.caption("All numbers computed directly by Python from your data. Not AI-generated.")
            st.dataframe(stat_df, use_container_width=True, hide_index=True)

        # ── TARGET ANALYSIS ──
        if facts.get("has_targets") and facts.get("target_gaps"):
            with st.expander("🎯 Target vs Actual — Python Verified", expanded=False):
                st.caption("Computed by Python from your actual and target columns.")
                for key, gap in facts["target_gaps"].items():
                    if isinstance(gap, dict) and "gap_pct" in gap:
                        status = gap.get("status","")
                        icon   = {"GREEN":"🟢","AMBER":"🟡","RED":"🔴"}.get(status,"⚪")
                        agg    = gap.get("agg_label","total")
                        st.markdown(
                            f"{icon} **{key.replace('_',' ')}** — "
                            f"Actual ({agg}): {gap['actual']:,.2f} | "
                            f"Target ({agg}): {gap['target']:,.2f} | "
                            f"Gap: {gap['gap_pct']:+.1f}%"
                        )

        # ── RECOMMENDATIONS ──
        recs = analysis.get("recommendations", [])
        if recs:
            st.markdown("#### ✅ Recommendations")
            for r in recs:
                p      = r.get("priority", 3)
                p_icon = {1:"🔴",2:"🟡",3:"🟢"}.get(p,"🔵")
                with st.container(border=True):
                    st.markdown(f"{p_icon} **P{p} · {clean_ai_text(r.get('action',''))}**")
                    if r.get("evidence"):
                        st.caption(f"📊 Evidence: {clean_ai_text(r['evidence'])}")
                    if r.get("hypothesis"):
                        # Strip the AI's own label if it included it — avoid duplication
                        hyp = r['hypothesis']
                        hyp = re.sub(r'^[Pp]ossible cause\s*\([^)]*\)\s*:', '', hyp).strip()
                        hyp = re.sub(r'^[Pp]ossible cause\s*:', '', hyp).strip()
                        if hyp:
                            st.caption(f"💭 Possible cause: {hyp}")
                    st.caption(f"Owner: {r.get('owner','')} · {r.get('timeframe','')}")

        # ── NARRATIVE — promoted out of expander per PRD ──
        narrative = analysis.get("narrative", {})
        if narrative:
            st.markdown("#### 📝 Narrative Report")
            with st.container(border=True):
                if narrative.get("opening"):
                    st.markdown(clean_ai_text(narrative["opening"]))
                for p in narrative.get("body", []):
                    st.markdown(clean_ai_text(p))
                if narrative.get("close"):
                    st.markdown(clean_ai_text(narrative["close"]))

        # ── EVALUATION METADATA ──
        ev = analysis.get("evaluation", {})
        if ev:
            with st.expander("🔬 Evaluation Metadata", expanded=False):
                st.caption("Quality assurance results — 6 evaluation methods.")
                e1, e2, e3 = st.columns(3)
                e1.metric("Relevance (AI scored)", f"{analysis.get('_relevance_avg', ev.get('relevance_score',0))}/10")
                e2.metric("Accuracy (Python)",     analysis.get("_py_acc_status", ev.get("accuracy_validated","—")))
                e3.metric("Coverage",              analysis.get("_py_coverage",{}).get("str", ev.get("coverage","—")))
                e4, e5, e6 = st.columns(3)
                e4.metric("Confidence (Python)",   analysis.get("_py_confidence", ev.get("confidence_overall","—")))
                e5.metric("Bias Check (Python)",   analysis.get("_py_bias",{}).get("status", ev.get("bias_check","—")))
                e6.metric("Eval Status",           ev.get("evaluation_status","—"))

        # ── AUDIT LOG ──
        if st.session_state.audit_log:
            with st.expander("📋 Session Audit Log", expanded=False):
                st.caption("In-session record of analyses run. Not stored permanently.")
                for entry in st.session_state.audit_log:
                    st.caption(
                        f"{entry['timestamp']} · Analysis #{entry['analysis_no']} · "
                        f"Role: {entry['role']} · Source: {entry['data_source']}"
                    )

        # ════════════════════════════════════════
        # SELF-SERVICE VISUAL BUILDER — PRD Feature 13
        # ════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🛠️ Build Your Own Analysis")
        st.caption("Select any columns, choose a chart type, and apply a filter. No code required — mirrors Tableau and Power BI self-service capability.")

        with st.container(border=True):
            # Row 1 — column selectors and chart type
            sb1, sb2, sb3 = st.columns(3)

            all_cols     = list(df.columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_sb  = df.select_dtypes(include=["object"]).columns.tolist()

            # Clean display labels for dropdowns
            col_labels     = {c: c.replace("_"," ").title() for c in all_cols}
            num_labels     = {c: c.replace("_"," ").title() for c in numeric_cols}

            with sb1:
                x_options   = all_cols
                x_display   = [col_labels[c] for c in x_options]
                x_selected_label = st.selectbox(
                    "X Axis (Group By)",
                    options=x_display,
                    key="sb_x"
                )
                x_selected = x_options[x_display.index(x_selected_label)]

            with sb2:
                y_options   = numeric_cols
                y_display   = [num_labels[c] for c in y_options]
                if y_options:
                    y_selected_label = st.selectbox(
                        "Y Axis (Metric)",
                        options=y_display,
                        key="sb_y"
                    )
                    y_selected = y_options[y_display.index(y_selected_label)]
                else:
                    st.caption("No numeric columns available.")
                    y_selected = None

            with sb3:
                chart_options  = ["Bar", "Line", "Pie", "Scatter"]
                sb_chart_type  = st.selectbox("Chart Type", options=chart_options, key="sb_chart")

            # Row 2 — optional filter
            sf1, sf2, sf3 = st.columns([2, 2, 1])
            with sf1:
                filter_col_options = ["None"] + cat_cols_sb
                filter_col_display = ["No filter"] + [c.replace("_"," ").title() for c in cat_cols_sb]
                filter_col_label   = st.selectbox("Filter by (optional)", options=filter_col_display, key="sb_filter_col")
                filter_col = None if filter_col_label == "No filter" else cat_cols_sb[filter_col_display.index(filter_col_label) - 1]

            with sf2:
                filter_vals_selected = []
                if filter_col:
                    available_vals = sorted(df[filter_col].dropna().unique().astype(str).tolist())
                    filter_vals_selected = st.multiselect(
                        f"{filter_col.replace('_',' ').title()} values",
                        options=available_vals,
                        placeholder="Select one or more values...",
                        key="sb_filter_val"
                    )

            with sf3:
                st.markdown("")
                st.markdown("")
                generate_custom = st.button("📊 Generate", type="primary", use_container_width=True, key="sb_generate")

            # Render custom chart
            if generate_custom and y_selected:
                try:
                    # Apply filter if set
                    df_filtered = df.copy()
                    if filter_col and filter_vals_selected:
                        df_filtered = df_filtered[df_filtered[filter_col].astype(str).isin(filter_vals_selected)]

                    if df_filtered.empty:
                        st.warning("No data matches this filter. Try a different value.")
                    else:
                        ctype    = col_classifications.get(y_selected, "value")
                        use_mean = ctype == "percentage"
                        x_label  = x_selected.replace("_"," ").title()
                        y_agg    = "Avg " if use_mean else "Total "
                        y_label  = y_agg + y_selected.replace("_"," ").title()
                        filter_note = f" — {filter_col.replace('_',' ').title()}: {', '.join(filter_vals_selected)}" if filter_vals_selected else ""
                        chart_title = f"{y_label} by {x_label}{filter_note}"

                        sb_chart_type_lower = sb_chart_type.lower()

                        if sb_chart_type_lower == "scatter":
                            if x_selected not in numeric_cols:
                                st.warning("Scatter chart requires a numeric X axis. Please select a numeric column for X.")
                            else:
                                sample = df_filtered[[x_selected, y_selected]].dropna().head(500)
                                fig    = px.scatter(
                                    sample, x=x_selected, y=y_selected,
                                    title=chart_title,
                                    color_discrete_sequence=["#3b82f6"],
                                    labels={x_selected: x_label, y_selected: y_label}
                                )
                                fig.update_traces(marker=dict(size=7, opacity=0.7))
                                fig.update_layout(**CHART_THEME)
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                        elif sb_chart_type_lower == "pie":
                            g = df_filtered.groupby(x_selected)[y_selected].mean().reset_index() if use_mean                                 else df_filtered.groupby(x_selected)[y_selected].sum().reset_index()
                            g = g[g[y_selected] > 0]
                            if g.empty:
                                st.warning("No positive values to display in pie chart.")
                            else:
                                fig = px.pie(g, names=x_selected, values=y_selected,
                                             title=chart_title, color_discrete_sequence=COLOURS)
                                fig.update_layout(**CHART_THEME)
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                        elif sb_chart_type_lower == "line":
                            g = df_filtered.groupby(x_selected)[y_selected].mean().reset_index() if use_mean                                 else df_filtered.groupby(x_selected)[y_selected].sum().reset_index()
                            try:
                                g = g.sort_values(x_selected)
                            except Exception:
                                pass
                            g.columns = [x_selected, y_label]
                            fig = px.line(g, x=x_selected, y=y_label, title=chart_title,
                                          markers=True, color_discrete_sequence=["#3b82f6"],
                                          labels={x_selected: x_label, y_label: y_label})
                            fig.update_layout(**CHART_THEME)
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                        else:
                            # Bar (default)
                            g = df_filtered.groupby(x_selected)[y_selected].mean().reset_index() if use_mean                                 else df_filtered.groupby(x_selected)[y_selected].sum().reset_index()
                            g.columns = [x_selected, y_label]
                            g         = g.sort_values(y_label, ascending=False)
                            fig = px.bar(g, x=x_selected, y=y_label, title=chart_title,
                                         color_discrete_sequence=["#3b82f6"],
                                         labels={x_selected: x_label, y_label: y_label})
                            fig.update_layout(**CHART_THEME)
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                        # Show what aggregation was used so user understands the numbers
                        agg_note = "Average" if use_mean else "Total (Sum)"
                        st.caption(f"📐 Aggregation method: **{agg_note}** — computed by Python from your data.")

                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

            elif generate_custom and not y_selected:
                st.warning("Please select a numeric column for the Y axis.")

    # ════════════════════════════════════════
    # CHAT PANEL
    # ════════════════════════════════════════
    if st.session_state.show_chat and chat_col is not None:
        with chat_col:
            with st.container(border=True):
                st.markdown("#### 💬 Ask SIMBA AI")
                st.caption(f"Answering as: **{analysis.get('role_interpreted', role)}**")
                st.markdown("---")

                if not st.session_state.chat_history:
                    st.markdown("**Suggested questions for your role:**")
                    for sug in st.session_state.chat_suggestions:
                        if st.button(sug, use_container_width=True,
                                     key=f"sug_{stable_hash(sug)}"):
                            st.session_state.chat_history.append({"role":"user","content":sug})
                            st.rerun()

                if st.session_state.chat_history:
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.markdown(f"**You:** {msg['content']}")
                            st.markdown("---")
                        else:
                            st.markdown(f"**SIMBA AI:** {clean_ai_text(msg['content'])}")
                            st.markdown("---")

                    last = st.session_state.chat_history[-1]

                    if last["role"] == "user":
                        with st.spinner("Thinking..."):
                            try:
                                chat_facts = build_chat_facts(df, col_classifications)
                                cat_values = {}
                                for _c in df.select_dtypes(include=["object"]).columns[:6]:
                                    cat_values[_c] = df[_c].dropna().unique().tolist()[:15]

                                history_context = ""
                                if len(st.session_state.chat_history) > 1:
                                    prev_msgs = st.session_state.chat_history[-5:-1]
                                    history_context = "Previous conversation:\n"
                                    for m in prev_msgs:
                                        prefix = "User" if m["role"] == "user" else "SIMBA AI"
                                        history_context += f"{prefix}: {m['content'][:200]}\n"
                                    history_context += "\n"

                                chat_system = (
                                    f"You are SIMBA AI answering for a "
                                    f"{analysis.get('role_interpreted', role)} "
                                    f"(Level: {analysis.get('level','L2')}).\n\n"
                                    "ABSOLUTE RULES:\n"
                                    "1. Use ONLY numbers from pre-computed facts. Zero exceptions.\n"
                                    "2. ONLY reference entities in actual_values_in_data.\n"
                                    "3. If answer not in facts: 'That information is not available in this dataset.'\n"
                                    "4. Never use training knowledge to fill gaps.\n"
                                    "5. Answer in under 150 words. Direct for this role.\n"
                                    "6. Percentage columns: describe as averages, never totals."
                                )
                                chat_user = (
                                    f"{history_context}"
                                    f"Pre-computed facts:\n{chat_facts}\n\n"
                                    f"actual_values_in_data: {json.dumps(cat_values, default=str)}\n\n"
                                    f"Executive summary: {json.dumps(analysis.get('executive_summary',{}))}\n\n"
                                    f"Question from {analysis.get('role_interpreted', role)}: {last['content']}"
                                )
                                answer    = call_groq(chat_system, chat_user, max_tokens=400)
                                followups = generate_followup_questions(
                                    role=analysis.get('role_interpreted', role),
                                    level=analysis.get('level','L2'),
                                    question=last['content'],
                                    answer=answer,
                                    df_columns=list(df.columns),
                                    cat_values=cat_values
                                )
                                st.session_state.chat_history.append({
                                    "role": "assistant", "content": answer, "followups": followups
                                })
                                st.rerun()
                            except Exception as e:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"Sorry, could not answer that. Error: {e}",
                                    "followups": []
                                })
                                st.rerun()

                    elif last["role"] == "assistant":
                        followups = last.get("followups", [])
                        if followups:
                            st.markdown("*You could also ask:*")
                            history_len = len(st.session_state.chat_history)
                            for fq_idx, fq in enumerate(followups):
                                btn_key = f"fq_{history_len}_{fq_idx}_{stable_hash(fq)}"
                                if st.button(fq, use_container_width=True, key=btn_key):
                                    st.session_state.chat_history.append({"role":"user","content":fq})
                                    st.rerun()

                st.markdown("---")
                user_q = st.text_input("Ask anything about your data...",
                                       key="chat_input", label_visibility="collapsed")
                ac, cc = st.columns([3, 1])
                with ac:
                    if st.button("Send →", use_container_width=True, type="primary"):
                        if user_q.strip():
                            st.session_state.chat_history.append({"role":"user","content":user_q.strip()})
                            st.rerun()
                with cc:
                    if st.button("Clear", use_container_width=True):
                        st.session_state.chat_history = []
                        # Regenerate role-specific suggestions on clear
                        if st.session_state.verified_facts and st.session_state.role_profile:
                            st.session_state.chat_suggestions = validate_and_fix_suggestions(
                                [], st.session_state.verified_facts,
                                st.session_state.role_profile, df
                            )
                        st.rerun()