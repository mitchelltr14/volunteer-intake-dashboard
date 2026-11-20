"""Volunteer Intake & Scheduling Automation Dashboard.

Structured volunteer questionnaire -> Claude-powered role matching ->
coordinator review -> searchable Volunteer Directory with welcome emails.
Also: Quick Add parses unstructured sign-up messages (emails, texts, forms).

Run with:  streamlit run app.py
"""

import json

import anthropic
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-5-5"

CHANNELS = ["Email", "Phone call", "Text / SMS", "WhatsApp", "Signal"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIME_WINDOWS = ["Weekday Mornings", "Weekday Afternoons", "Weekday Evenings", "Weekends"]
FREQUENCIES = ["Recurring (weekly / monthly)", "One-time / event-based", "Either"]

ABILITY_AREAS = [
    "Event & Logistics Support",
    "Food / Supply Sorting & Prep",
    "Outreach & Community Tabling",
    "Graphic Design / Social Media",
    "Admin / Data Entry",
    "Translation / Language Support",
]
NOT_SELECTED = "Not interested"
COMFORT_LEVELS = [
    "Beginner / Willing to Learn",
    "Comfortable / Hands-On",
    "Experienced / Can Lead",
]

WORK_SETTINGS = ["On-site", "Remote", "Either"]

ROLE_CATALOG = [
    "Event Setup & Logistics Crew",
    "Event Day Team Lead",
    "Food Pantry Sorter / Packer",
    "Meal Prep Assistant",
    "Community Outreach Ambassador",
    "Tabling & Info Booth Volunteer",
    "Social Media & Design Volunteer",
    "Office Admin & Data Entry Volunteer",
    "Interpreter / Translation Volunteer",
    "Delivery Driver",
]

SAMPLE_VOLUNTEER = {
    "q_name": "Maria Delgado",
    "q_contact": "maria.delgado@example.org · (555) 214-0198",
    "q_channel": "Text / SMS",
    "q_days": ["Tue", "Thu", "Sat"],
    "q_win_Weekday Mornings": False,
    "q_win_Weekday Afternoons": False,
    "q_win_Weekday Evenings": True,
    "q_win_Weekends": True,
    "q_frequency": "Recurring (weekly / monthly)",
    "q_ability_Event & Logistics Support": "Comfortable / Hands-On",
    "q_ability_Food / Supply Sorting & Prep": "Experienced / Can Lead",
    "q_ability_Outreach & Community Tabling": "Beginner / Willing to Learn",
    "q_ability_Graphic Design / Social Media": NOT_SELECTED,
    "q_ability_Admin / Data Entry": NOT_SELECTED,
    "q_ability_Translation / Language Support": "Experienced / Can Lead",
    "q_setting": "On-site",
    "q_lifting": "Can lift up to 25 lbs; no heavy pallets.",
    "q_standing": "Fine standing for 2–3 hours with short breaks.",
    "q_transport": "Has a car and a valid license; can do local deliveries.",
    "q_other_constraints": "Prefers not to work outdoors in extreme heat.",
    "q_experience": (
        "Fluent in Spanish and English. Volunteered 2 years at a church food "
        "bank, coordinating a Saturday sorting shift of ~8 people. Helped "
        "translate intake forms for families."
    ),
}

# JSON schema the model must follow (enforced via structured outputs).
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "contact": {"type": "string"},
        "preferred_channel": {"type": "string"},
        "skills_summary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string"},
                    "level": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["area", "level", "note"],
                "additionalProperties": False,
            },
        },
        "availability_summary": {"type": "string"},
        "matched_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["role", "reason"],
                "additionalProperties": False,
            },
        },
        "preferred_role": {"type": "string"},
        "draft_welcome_email": {"type": "string"},
    },
    "required": [
        "name",
        "contact",
        "preferred_channel",
        "skills_summary",
        "availability_summary",
        "matched_roles",
        "preferred_role",
        "draft_welcome_email",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""You are a volunteer coordinator assistant for a community nonprofit.
Given one volunteer's intake questionnaire, produce a profile for the coordinator.

Rules:
- Suggest the top 2-3 roles from this catalog: {", ".join(ROLE_CATALOG)}.
- Only suggest roles that respect the volunteer's constraints (lifting, standing,
  remote vs. on-site, transport) and fit their stated availability.
- Each reason is one or two concise sentences tying abilities, comfort level,
  constraints and availability to the role.
- Put the best match first and set preferred_role to it.
- skills_summary lists only the areas the volunteer selected, with their level.
- availability_summary is one short sentence.
- draft_welcome_email: a warm, personalized email (subject line first, then body,
  under 200 words) that names their matched role, reflects their skills and
  comfort level, and confirms their availability. Sign off as "The Volunteer Team".
- Do not invent facts that are not in the questionnaire."""


# Free-text sign-up message parsing (Quick Add tab).
SAMPLE_MESSAGE = (
    "Hi there! My name is Jordan Lee and I saw your post about needing help at "
    "the food pantry. I'm a retired teacher, great with organizing, and I've "
    "done a lot of spreadsheet work. I also speak some Mandarin. I'm free most "
    "Tuesday and Thursday mornings and the occasional Saturday. I'd love to help "
    "with sorting donations or keeping your records tidy. You can reach me at "
    "jordan.lee@example.com. Thanks!"
)

MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "availability": {"type": "string"},
        "preferred_role": {"type": "string"},
        "draft_welcome_email": {"type": "string"},
    },
    "required": ["name", "skills", "availability", "preferred_role", "draft_welcome_email"],
    "additionalProperties": False,
}

MESSAGE_SYSTEM_PROMPT = f"""You are a volunteer coordinator assistant for a community nonprofit.
Extract a volunteer profile from an unstructured sign-up message (email, text or form).

Rules:
- name: the volunteer's full name, or "Unknown" if not given.
- skills: short skill phrases taken from the message.
- availability: one short sentence summarizing when they can help.
- preferred_role: the role they asked for, otherwise the best fit from:
  {", ".join(ROLE_CATALOG)}.
- draft_welcome_email: a warm, personalized email (subject line first, then body,
  under 200 words) that references their specific skills and availability.
  Sign off as "The Volunteer Team".
- Do not invent facts that are not in the message."""


# ---------------------------------------------------------------------------
# State & helpers
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("directory", [])
    st.session_state.setdefault("pending_profile", None)


def fill_sample() -> None:
    """Callback: populate every questionnaire widget with sample answers."""
    st.session_state.update(SAMPLE_VOLUNTEER)


def get_api_key() -> str | None:
    """Prefer st.secrets; fall back to a password field in the sidebar."""
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        key = None
    if key:
        st.sidebar.success("API key loaded from secrets.")
        return key
    return st.sidebar.text_input("Anthropic API key", type="password") or None


def collect_responses() -> dict:
    """Read questionnaire answers from session state into a clean dict."""
    ss = st.session_state
    abilities = {
        area: ss.get(f"q_ability_{area}")
        for area in ABILITY_AREAS
        if ss.get(f"q_ability_{area}", NOT_SELECTED) != NOT_SELECTED
    }
    return {
        "full_name": ss.get("q_name", "").strip(),
        "contact": ss.get("q_contact", "").strip(),
        "preferred_channel": ss.get("q_channel"),
        "availability": {
            "days": ss.get("q_days", []),
            "time_windows": [w for w in TIME_WINDOWS if ss.get(f"q_win_{w}")],
            "frequency": ss.get("q_frequency"),
        },
        "abilities": abilities,
        "constraints": {
            "work_setting": ss.get("q_setting"),
            "lifting": ss.get("q_lifting", ""),
            "standing": ss.get("q_standing", ""),
            "driving_transport": ss.get("q_transport", ""),
            "other": ss.get("q_other_constraints", ""),
        },
        "prior_experience_notes": ss.get("q_experience", ""),
    }


def validate(responses: dict) -> list[str]:
    errors = []
    if not responses["full_name"]:
        errors.append("Full name is required.")
    if not responses["contact"]:
        errors.append("Email / contact is required.")
    if not (responses["availability"]["days"] or responses["availability"]["time_windows"]):
        errors.append("Select at least one day or time window.")
    if not responses["abilities"]:
        errors.append("Select a comfort level for at least one ability area.")
    return errors


# ---------------------------------------------------------------------------
# Claude evaluation
# ---------------------------------------------------------------------------

def call_claude(api_key: str, system: str, content: str, schema: dict) -> dict:
    """Send one request to Claude and return its JSON output as a dict.

    Raises RuntimeError with a user-friendly message on any failure.
    """
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Invalid Anthropic API key.")
    except anthropic.RateLimitError:
        raise RuntimeError("Rate limited by the Anthropic API — try again shortly.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Anthropic API — check your connection.")

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to process this request.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The response was cut off before completing. Please retry.")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("Could not parse the model's response as JSON. Please retry.")
    if not isinstance(data, dict):
        raise RuntimeError("The model's response was not a JSON object. Please retry.")
    missing = [k for k in schema["required"] if k not in data]
    if missing:
        raise RuntimeError(f"Model response is missing fields: {', '.join(missing)}")
    return data


def evaluate_volunteer(api_key: str, responses: dict) -> dict:
    """Questionnaire -> full profile with matched roles."""
    content = "Volunteer questionnaire (JSON):\n" + json.dumps(responses, indent=2)
    return call_claude(api_key, SYSTEM_PROMPT, content, PROFILE_SCHEMA)


def extract_from_message(api_key: str, message: str) -> dict:
    """Free-text sign-up message -> directory entry (same shape as questionnaire profiles)."""
    data = call_claude(api_key, MESSAGE_SYSTEM_PROMPT, f"Sign-up message:\n{message}", MESSAGE_SCHEMA)
    return {
        "name": data["name"],
        "contact": "",
        "preferred_channel": "—",
        "skills_summary": [{"area": s, "level": "", "note": ""} for s in data["skills"]],
        "availability_summary": data["availability"],
        "matched_roles": [],
        "preferred_role": data["preferred_role"],
        "draft_welcome_email": data["draft_welcome_email"],
    }


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def render_questionnaire() -> bool:
    """Render the intake form. Returns True when submitted."""
    st.header("1 · Volunteer Ability & Intake Questionnaire")
    st.button("✨ Fill Sample Volunteer", on_click=fill_sample)

    with st.form("intake_form"):
        st.subheader("Basic info")
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.text_input("Full name *", key="q_name")
        c2.text_input("Email / contact *", key="q_contact")
        c3.selectbox("Preferred channel", CHANNELS, key="q_channel")

        st.subheader("Availability")
        st.pills("Days of the week", DAYS, selection_mode="multi", key="q_days")
        cols = st.columns(len(TIME_WINDOWS))
        for col, w in zip(cols, TIME_WINDOWS):
            col.checkbox(w, key=f"q_win_{w}")
        st.radio("Commitment type", FREQUENCIES, horizontal=True, key="q_frequency")

        st.subheader("Abilities & comfort levels")
        options = [NOT_SELECTED, *COMFORT_LEVELS]
        grid = st.columns(2)
        for i, area in enumerate(ABILITY_AREAS):
            grid[i % 2].selectbox(area, options, key=f"q_ability_{area}")

        st.subheader("Constraints & notes")
        st.radio("Work setting", WORK_SETTINGS, horizontal=True, key="q_setting")
        c1, c2 = st.columns(2)
        c1.text_input("Lifting", key="q_lifting", placeholder="e.g. up to 25 lbs")
        c2.text_input("Standing / mobility", key="q_standing", placeholder="e.g. 2 hrs max")
        c1.text_input("Driving / transport", key="q_transport", placeholder="e.g. has car")
        c2.text_input("Other preferences / constraints", key="q_other_constraints")
        st.text_area("Prior experience / notes", key="q_experience", height=90)

        return st.form_submit_button("🔍 Evaluate & Match Roles", type="primary")


def render_review(profile: dict) -> None:
    """Coordinator review: confirm/override role, edit email, add to directory."""
    st.header("2 · Role Matching Review")
    st.markdown(f"**{profile['name']}** · {profile['contact']} · prefers *{profile['preferred_channel']}*")
    st.caption(f"🗓️ {profile['availability_summary']}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Skills & comfort levels**")
        for s in profile["skills_summary"]:
            note = f" — {s['note']}" if s.get("note") else ""
            st.markdown(f"- **{s['area']}**: {s['level']}{note}")
    with c2:
        st.markdown("**Suggested roles**")
        for i, m in enumerate(profile["matched_roles"], 1):
            st.markdown(f"{i}. **{m['role']}** — {m['reason']}")

    matched = [m["role"] for m in profile["matched_roles"]]
    choices = list(dict.fromkeys([profile["preferred_role"], *matched, *ROLE_CATALOG]))
    role = st.selectbox("Confirm or override preferred role", choices, index=0)
    email = st.text_area("Welcome email draft (editable)", profile["draft_welcome_email"], height=260)

    c1, c2 = st.columns([1, 4])
    if c1.button("✅ Add to Directory", type="primary"):
        st.session_state.directory.append({**profile, "preferred_role": role, "draft_welcome_email": email})
        st.session_state.pending_profile = None
        st.success(f"{profile['name']} added to the directory.")
        st.rerun()
    if c2.button("Discard"):
        st.session_state.pending_profile = None
        st.rerun()


def render_quick_add(api_key: str | None) -> None:
    """Paste an unstructured sign-up message and add the parsed volunteer directly."""
    st.header("Quick Add from Sign-up Message")
    st.caption("Paste an email, text or form submission. Claude extracts the details.")
    message = st.text_area("Volunteer message", placeholder=SAMPLE_MESSAGE, height=180)
    if st.button("➕ Extract & Add Volunteer", type="primary"):
        if not message.strip():
            st.error("Paste a volunteer message first.")
        elif not api_key:
            st.error("Add an Anthropic API key in the sidebar (or in .streamlit/secrets.toml).")
        else:
            with st.spinner("Extracting volunteer details…"):
                try:
                    profile = extract_from_message(api_key, message)
                except RuntimeError as e:
                    st.error(str(e))
                    return
            st.session_state.directory.append(profile)
            st.success(f"{profile['name']} added to the directory. See the Directory tab.")


def format_skills(skills: list[dict]) -> str:
    """'Area (Level)' for questionnaire skills; plain text for message-extracted skills."""
    return "; ".join(
        f"{s['area']} ({s['level'].split(' /')[0]})" if s.get("level") else s["area"]
        for s in skills
    )


def directory_frame(directory: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Name": p["name"],
            "Contact": p["contact"],
            "Channel": p["preferred_channel"],
            "Preferred Role": p["preferred_role"],
            "Skills": format_skills(p["skills_summary"]),
            "Availability": p["availability_summary"],
        }
        for p in directory
    ])


def render_directory() -> None:
    st.header("3 · Volunteer Directory")
    directory = st.session_state.directory
    if not directory:
        st.info("No volunteers yet. Submit a questionnaire to add one.")
        return

    df = directory_frame(directory)
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search (name, contact, skills, availability)", key="dir_search")
    roles = c2.multiselect("Role", sorted(df["Preferred Role"].unique()))
    channels = c3.multiselect("Channel", sorted(df["Channel"].unique()))

    mask = pd.Series(True, index=df.index)
    if query:
        mask &= df.apply(lambda r: query.lower() in " ".join(map(str, r)).lower(), axis=1)
    if roles:
        mask &= df["Preferred Role"].isin(roles)
    if channels:
        mask &= df["Channel"].isin(channels)
    filtered = df[mask]

    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} volunteers.")

    if filtered.empty:
        return
    st.subheader("Welcome email preview")
    idx = st.selectbox(
        "Volunteer",
        filtered.index,
        format_func=lambda i: f"{directory[i]['name']} — {directory[i]['preferred_role']}",
    )
    volunteer = directory[idx]
    # Edits are saved back to the directory entry as you type.
    volunteer["draft_welcome_email"] = st.text_area(
        "Edit email", volunteer["draft_welcome_email"], height=260, key=f"email_edit_{idx}"
    )
    # st.code provides a one-click copy button in its top-right corner.
    st.code(volunteer["draft_welcome_email"], language=None, wrap_lines=True)
    st.download_button(
        "Download as .txt",
        volunteer["draft_welcome_email"],
        file_name=f"welcome_{volunteer['name'].replace(' ', '_').lower()}.txt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Volunteer Intake Dashboard", page_icon="🤝", layout="wide")
    init_state()
    st.title("🤝 Volunteer Intake & Scheduling Automation Dashboard")

    st.sidebar.header("Settings")
    api_key = get_api_key()
    st.sidebar.metric("Volunteers in directory", len(st.session_state.directory))

    tab_intake, tab_quick, tab_directory = st.tabs(
        ["📝 Intake & Matching", "✉️ Quick Add from Message", "📇 Directory"]
    )

    with tab_quick:
        render_quick_add(api_key)

    with tab_intake:
        if render_questionnaire():
            responses = collect_responses()
            errors = validate(responses)
            if errors:
                for e in errors:
                    st.error(e)
            elif not api_key:
                st.error("Add an Anthropic API key in the sidebar (or in .streamlit/secrets.toml).")
            else:
                with st.spinner("Matching roles and drafting welcome email…"):
                    try:
                        st.session_state.pending_profile = evaluate_volunteer(api_key, responses)
                    except RuntimeError as e:
                        st.error(str(e))

        if st.session_state.pending_profile:
            st.divider()
            render_review(st.session_state.pending_profile)

    with tab_directory:
        render_directory()


if __name__ == "__main__":
    main()
