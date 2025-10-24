# app.py
import json
import os
import uuid
from pathlib import Path

import streamlit as st
from datetime import date, datetime, timezone
import pytz

try:
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    HAS_GOOGLE_AUTH = True
except ImportError:  # pragma: no cover
    HAS_GOOGLE_AUTH = False

from position_statement_renderer import (
    extract_json_from_response,
    render_position_statement_pdf,
)
from vox_helpers import compute_background_summary
from vox_extract import extract_all, generate_position_statement

SHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


def _load_service_account_info():
    if "google_service_account" in st.secrets:
        info = st.secrets["google_service_account"]
        if isinstance(info, str):
            return json.loads(info)
        return dict(info)
    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                f"Google service account file not found at {path}."
            )
        return json.loads(path.read_text(encoding="utf-8"))
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")
    if raw:
        return json.loads(raw)
    raise RuntimeError(
        "Google service account credentials are not configured. "
        "Set st.secrets['google_service_account'], define the GOOGLE_SERVICE_ACCOUNT_FILE, "
        "or provide GOOGLE_SERVICE_ACCOUNT_INFO environment variable."
    )


def _get_spreadsheet_id():
    # Check if it's nested in google_service_account section
    if "google_service_account" in st.secrets:
        service_account_info = st.secrets["google_service_account"]
        if isinstance(service_account_info, dict) and "google_sheets_spreadsheet_id" in service_account_info:
            return service_account_info["google_sheets_spreadsheet_id"]
    
    # Check if it's at root level
    if "google_sheets_spreadsheet_id" in st.secrets:
        return st.secrets["google_sheets_spreadsheet_id"]
    
    sheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
    if sheet_id:
        return sheet_id
    raise RuntimeError(
        "Google Sheets spreadsheet ID is missing. "
        "Set st.secrets['google_service_account']['google_sheets_spreadsheet_id'], "
        "st.secrets['google_sheets_spreadsheet_id'], or the GOOGLE_SHEETS_SPREADSHEET_ID environment variable."
    )


def _get_spreadsheet_range():
    # Check if it's nested in google_service_account section
    if "google_service_account" in st.secrets:
        service_account_info = st.secrets["google_service_account"]
        if isinstance(service_account_info, dict) and "google_sheets_range" in service_account_info:
            return service_account_info["google_sheets_range"]
    
    # Check if it's at root level
    if "google_sheets_range" in st.secrets:
        return st.secrets["google_sheets_range"]
    
    return os.getenv("GOOGLE_SHEETS_RANGE", "Feedback!A:J")


def append_feedback_to_sheet(feedback):
    if not HAS_GOOGLE_AUTH:
        raise ImportError(
            "google-auth is required to talk to the Google Sheets API. "
            "Install it with `pip install google-auth`."
        )

    service_info = _load_service_account_info()
    credentials = service_account.Credentials.from_service_account_info(
        service_info,
        scopes=SHEETS_SCOPES,
    )
    authed_session = AuthorizedSession(credentials)
    spreadsheet_id = _get_spreadsheet_id()
    range_name = _get_spreadsheet_range()

    values = [[
        feedback["run_id"],
        feedback["timestamp_utc"],
        feedback["stage"],
        feedback["accuracy"],
        feedback["relevance"],
        feedback["writing_style"],
        feedback["presentation"],
        feedback["ease_of_use"],
        feedback["remarks"],
        feedback.get("pdf_filename", ""),
    ]]

    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/"
        f"{range_name}:append?valueInputOption=USER_ENTERED"
    )
    
    response = authed_session.post(url, json={"values": values})
    
    if response.status_code >= 300:
        raise RuntimeError(
            f"Sheets API returned {response.status_code}: {response.text}"
        )

st.set_page_config(page_title="Segmented Chatbot", page_icon="💬", layout="centered")

SEGMENTS = [
    {"id": "about_exclusion", "title": "About the Exclusion", "color": "#E3F2FD"},  # light blue
    {"id": "about_young_person", "title": "About the Young Person", "color": "#E8F5E9"},  # light green
    {"id": "about_procedure", "title": "About the Procedure", "color": "#FFF3E0"},  # light orange
    {"id": "document_details", "title": "Document Details", "color": "#F3E5F5"},  # light purple
]

# Define questions per segment.
# Each question has:
#   key: variable name used to store the answer
#   prompt: text shown to the user
#   type: "text" | "textarea" | "number" | "select" | "radio" | "checkbox"
#   options: (optional) for select/radio
#   condition: (optional) callable(answers) -> bool to show conditionally
QUESTIONS = {
    "about_exclusion": [
        {"key": "is_permanently_excluded", "prompt": "Has your child been permanently excluded?", "type": "radio", "options": ["Yes", "No"]},
        {"key": "exclusion_letter_content", "prompt": "Was a letter written to confirm your child's exclusion? Please provide the content of the exclusion letter and the reasons given by the school.", "type": "textarea"},
        {"key": "school_version_events", "prompt": "What does the school say happened to lead to the exclusion? Please describe the school's version of events.", "type": "textarea"},
        {"key": "school_evidence", "prompt": "What evidence does the school have to support the exclusion?", "type": "textarea"},
        {"key": "student_agrees_with_school", "prompt": "Does the young person agree with the school's version of events?", "type": "radio", "options": ["Yes", "No"]},
        {"key": "student_version_events", "prompt": "What is the young person's version of events?", "type": "textarea"},
        {"key": "witnesses_details", "prompt": "Are there witnesses that can support the young person's version of events? Please provide details.", "type": "textarea"},
        {"key": "student_voice_heard_details", "prompt": "Did the school speak with the young person and take their version of events before excluding them? Please provide details.", "type": "textarea"},
    ],
    "about_young_person": [
        {"key": "is_send", "prompt": "Does the young person have special educational needs or disabilities (SEND)?", "type": "radio", "options": ["Yes", "No"]},
        {"key": "send_details", "prompt": "Please describe the SEND and how the school have made adjustments to address this SEND.", "type": "textarea", "condition": lambda a: a.get("is_send") == "Yes"},
        {"key": "ehcp_details", "prompt": "Does the young person have an EHCP? Please provide details about how the school has implemented it.", "type": "textarea", "condition": lambda a: a.get("is_send") == "Yes"},
        {"key": "is_ethnic_minority", "prompt": "Is the young person from an ethnic minority background?", "type": "radio", "options": ["Yes", "No"]},
        {"key": "previous_suspensions_details", "prompt": "Has the young person been previously suspended? Please provide details and rough dates.", "type": "textarea"},
        {"key": "family_awareness_details", "prompt": "Were the family aware of behavioural issues, or the risk of exclusion before it happened? Please provide details.", "type": "textarea"},
        {"key": "personal_issues_details", "prompt": "Are there any personal issues that the young person is facing that may have contributed to the exclusion?", "type": "textarea"},
    ],
    "about_procedure": [
        {"key": "stage", "prompt": "What stage of procedure is the exclusion at?", "type": "radio", "options": ["Governors Panel", "Independent Review Panel"]},
        {"key": "governor_procedure_info", "prompt": "Please provide details about any procedural issues during the Governors meeting. Did you have any concerns about fairness, time limits, or anything else that seemed odd?", "type": "textarea", "condition": lambda a: a.get("stage") == "Independent Review Panel"},
        {"key": "other_information_provided", "prompt": "Are there any other information that you would like to provide?", "type": "textarea"},
    ],
    "document_details": [
        {"key": "child_name", "prompt": "What is the name of the young person?", "type": "text"},
        {"key": "parent_name", "prompt": "What is the name of the parent?", "type": "text"},
        {"key": "school_name", "prompt": "What is the name of the school?", "type": "text"},
        {"key": "exclusion_date", "prompt": "What is the date of the exclusion?", "type": "date"},
        {"key": "exclusion_letter_date", "prompt": "What is the date of the exclusion letter?", "type": "date"},
    ],
}

# ---------------------------
# Helpers
# ---------------------------
def heading_bubble(text, color):
    st.markdown(
        f"""
        <style>
        .heading-bubble h4 {{
            color: #000000 !important;
        }}
        .heading-bubble {{
            color: #000000 !important;
        }}
        </style>
        <div class="heading-bubble" style="
            background:{color};
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            margin: 8px 0;
            color: #000000 !important;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def ask_question(q, answers):
    """Render a single question and return possibly updated answer for this key."""
    key = q["key"]
    prompt = q["prompt"]
    qtype = q["type"]
    options = q.get("options")

    # Only render if no condition or condition passes
    cond = q.get("condition")
    if cond and not cond(answers):
        return answers.get(key, None)  # skip but preserve previous value if any

    # Render appropriate widget
    if qtype == "text":
        val = st.text_input(prompt, key=key, value=answers.get(key, ""), help="")
    elif qtype == "textarea":
        val = st.text_area(prompt, key=key, value=answers.get(key, ""), help="")
    elif qtype == "number":
        val = st.number_input(prompt, key=key, value=answers.get(key, 0), step=1)
    elif qtype == "select":
        val = st.selectbox(prompt, options, key=key, index=(options.index(answers[key]) if answers.get(key) in options else 0))
    elif qtype == "radio":
        # Show radio with no default selection unless a previous answer exists
        prev_value = answers.get(key)
        widget_value = st.session_state.get(key)

        if widget_value in options:
            default_index = None  # let Streamlit use the widget state
        elif prev_value in options:
            default_index = options.index(prev_value)
        else:
            default_index = None

        val = st.radio(
            prompt,
            options,
            key=key,
            horizontal=True,
            index=default_index,
        )
    elif qtype == "checkbox":
        val = st.checkbox(prompt, key=key, value=bool(answers.get(key, False)))
    elif qtype == "date":
        prev_value = answers.get(key)
        if isinstance(prev_value, str):
            try:
                prev_value = datetime.fromisoformat(prev_value).date()
            except ValueError:
                prev_value = None
        if isinstance(prev_value, datetime):
            prev_value = prev_value.date()
        val = st.date_input(
            prompt,
            key=key,
            value=prev_value if isinstance(prev_value, date) else None,
        )
    else:
        st.warning(f"Unknown question type: {qtype}")
        val = answers.get(key)

    return val

def segment_done(segment_id, answers):
    """Check if all visible questions in a segment have been answered (non-empty)."""
    for q in QUESTIONS[segment_id]:
        if q["key"] in {"other_information_provided", "exclusion_letter_date"}:
            continue
        cond = q.get("condition")
        if cond and not cond(answers):
            continue
        v = answers.get(q["key"])
        if v in (None, ""):
            # For checkbox False is a valid answer; treat None only as missing
            if q["type"] == "checkbox":
                if answers.get(q["key"]) is None:
                    return False
            else:
                return False
    return True

# ---------------------------
# State
# ---------------------------
if "step" not in st.session_state:
    st.session_state.step = 0  # which segment we are on
if "answers" not in st.session_state:
    st.session_state.answers = {}

answers = st.session_state.answers

st.title("Vox")

# Progress header
steps_total = len(SEGMENTS)
st.progress((st.session_state.step + 1) / steps_total)

current = SEGMENTS[st.session_state.step]
heading_bubble(f"<h4 style='margin:0'>{current['title']}</h4>", current["color"])

for q in QUESTIONS[current["id"]]:
    # Evaluate and render question (handles conditionals)
    val = ask_question(q, answers)
    answers[q["key"]] = val

    if q["key"] == "is_permanently_excluded" and val == "No":
        st.error("⚠️ We currently only support permanent exclusions, not suspensions.")
        st.stop()

col1, _ = st.columns([1, 1])
if col1.button("Save answers"):
    st.success("Answers saved.")

# Navigation
if st.session_state.step < steps_total - 1:
    nav_cols = st.columns([1, 1, 1])
    if nav_cols[0].button("⬅ Previous", disabled=st.session_state.step == 0):
        st.session_state.step -= 1
        st.rerun()

    can_go_next = segment_done(current["id"], answers)
    if nav_cols[1].button("Next ➡", disabled=not can_go_next):
        st.session_state.step += 1
        st.rerun()
else:
    back_col, _ = st.columns([1, 1])
    if back_col.button("⬅ Previous", disabled=st.session_state.step == 0):
        st.session_state.step -= 1
        st.rerun()

# When finished, show a summary and expose variables
if st.session_state.step == steps_total - 1:
    st.divider()
    st.subheader("Collected answers")
    st.json(answers)

    is_permanently_excluded = answers.get("is_permanently_excluded")
    exclusion_letter_content = answers.get("exclusion_letter_content")
    school_version_events = answers.get("school_version_events")
    school_evidence = answers.get("school_evidence")
    student_agrees_with_school = answers.get("student_agrees_with_school")
    student_version_events = answers.get("student_version_events")
    witnesses_details = answers.get("witnesses_details")
    student_voice_heard_details = answers.get("student_voice_heard_details")

    background_summary_lines = compute_background_summary(answers)
    st.subheader("Background summary")
    for line in background_summary_lines:
        st.markdown(f"- {line}")
    background_summary_text = "\n".join(background_summary_lines)

    stage = answers.get("stage")
    governor_procedure_info = answers.get("governor_procedure_info")
    other_information_provided = answers.get("other_information_provided") or ""

    child_name = answers.get("child_name")
    parent_name = answers.get("parent_name")
    school_name = answers.get("school_name")
    exclusion_date = answers.get("exclusion_date")
    exclusion_letter_date = answers.get("exclusion_letter_date")

    # "Submit" button demonstrating you can now process/store/send the data
    if st.button("Submit all data"):
        # Store the processed data in session state
        school_facts, exclusion_reason, student_perspective = extract_all(
            exclusion_letter_content,
            school_version_events,
            school_evidence,
            student_agrees_with_school,
            student_version_events,
            witnesses_details,
            student_voice_heard_details,
        )

        st.success("Summaries extracted.")
        st.success(f"School facts: {school_facts}")
        st.success(f"Exclusion reason: {exclusion_reason}")
        st.success(f"Student perspective: {student_perspective}")

        # Store summaries in session state for persistent display
        st.session_state["extracted_summaries"] = {
            "school_facts": school_facts,
            "exclusion_reason": exclusion_reason,
            "student_perspective": student_perspective,
        }

        stage_info_parts = []
        if stage:
            stage_info_parts.append(f"Stage: {stage}.")
        if governor_procedure_info:
            stage_info_parts.append(f"Governor procedure information: {governor_procedure_info}")
        stage_info_text = " ".join(stage_info_parts) if stage_info_parts else "Stage information not provided."

        other_info_text = other_information_provided

        exclusion_date_value = exclusion_date
        if isinstance(exclusion_date_value, datetime):
            exclusion_date_value = exclusion_date_value.date()
        if isinstance(exclusion_date_value, date):
            exclusion_date_text = exclusion_date_value.strftime("%d %B %Y")
        elif isinstance(exclusion_date_value, str):
            exclusion_date_text = exclusion_date_value
        else:
            exclusion_date_text = ""

        exclusion_letter_date_value = exclusion_letter_date
        if isinstance(exclusion_letter_date_value, datetime):
            exclusion_letter_date_value = exclusion_letter_date_value.date()
        if isinstance(exclusion_letter_date_value, date):
            exclusion_letter_date_text = exclusion_letter_date_value.strftime("%d %B %Y")
        elif isinstance(exclusion_letter_date_value, str):
            exclusion_letter_date_text = exclusion_letter_date_value
        else:
            exclusion_letter_date_text = ""

        if stage == "Independent Review Panel":
            specific_instructions = (
                "Focus the grounds on procedural and substantive issues with the governing board decision. "
                "Emphasise errors during the governors' meeting and why the IRP should overturn that decision."
            )
        else:
            specific_instructions = ""

        grounds_path = (
            Path("documents/irp_arguments.json") if stage == "Independent Review Panel"
            else Path("documents/governors_panel_arguments.json")
        )
        try:
            position_statement_grounds = grounds_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            position_statement_grounds = "{}"
            st.warning(f"Could not load position statement grounds from {grounds_path}. Using empty JSON object.")

        position_statement, guidance_context = generate_position_statement(
            exclusion_reason=exclusion_reason,
            school_facts=school_facts,
            student_perspective=student_perspective,
            background_summary=background_summary_text,
            stage_info=stage_info_text,
            other_information_provided=other_info_text,
            exclusion_letter_date=exclusion_letter_date_text,
            specific_instructions=specific_instructions,
            position_statement_grounds=position_statement_grounds,
        )

        try:
            position_payload = extract_json_from_response(position_statement)
        except ValueError as exc:
            st.error(f"Could not parse the position statement JSON: {exc}")
            st.stop()

        user_details = {
            "child_name": child_name or "",
            "parent_name": parent_name or "",
            "school_name": school_name or "",
            "stage": stage or "",
            "exclusion_date": exclusion_date_text,
            "exclusion_letter_date": exclusion_letter_date_text,
        }

        try:
            rendered_statement = render_position_statement_pdf(
                position_payload,
                user_details=user_details,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.success("Position statement PDF generated successfully.")

            st.subheader("Guidance context used for RAG")
            st.write("Behaviour in Schools excerpts:")
            st.text_area("Behaviour Guidance Context", guidance_context["behaviour_in_schools"], height=200, help="")
            st.write("Suspensions guidance excerpts:")
            st.text_area("Suspensions Guidance Context", guidance_context["suspensions"], height=200, help="")

            st.subheader("Parsed position statement JSON")
            st.json(rendered_statement.json_payload)

            pdf_bytes = rendered_statement.pdf_path.read_bytes()
            download_name = rendered_statement.pdf_path.name
            st.download_button(
                label="Download Position Statement PDF",
                data=pdf_bytes,
                file_name=download_name,
                mime="application/pdf",
            )


            st.caption(f"LaTeX source: {rendered_statement.tex_path}")
            st.caption(f"Compilation log: {rendered_statement.log_path}")

            run_id = uuid.uuid4().hex
            st.session_state["latest_run"] = {
                "run_id": run_id,
                "stage": stage or "",
                "child_name": child_name or "",
                "parent_name": parent_name or "",
                "school_name": school_name or "",
                "exclusion_date": exclusion_date_text,
                "exclusion_letter_date": exclusion_letter_date_text,
                "pdf_filename": download_name,
            }
            st.session_state["latest_pdf_bytes"] = pdf_bytes
            st.session_state["latest_pdf_name"] = download_name

            st.subheader("Reviewer Evaluation")
            with st.form("evaluation_form"):
                st.markdown(f"Run ID: `{run_id}`")
                
                # Reviewer information
                st.markdown("**Reviewer Information**")
                reviewer_name = st.text_input("Your name", placeholder="Enter your full name", help="")
                reviewer_email = st.text_input("Your email", placeholder="Enter your email address", help="")
                
                st.markdown("**Evaluation Scores**")
                accuracy_score = st.slider("Accuracy of factual content", 0, 10, 5)
                relevance_score = st.slider("Relevance of arguments", 0, 10, 5)
                writing_score = st.slider("Writing style and clarity", 0, 10, 5)
                presentation_score = st.slider("Presentation of document", 0, 10, 5)
                ease_score = st.slider("Ease of using this tool", 0, 10, 5)
                reviewer_remarks = st.text_area("Additional comments or specific issues", height=120, help="")
                submitted_feedback = st.form_submit_button("Submit feedback")

            if submitted_feedback:
                # Use UK timezone instead of UTC
                uk_tz = pytz.timezone('Europe/London')
                timestamp_uk = datetime.now(uk_tz).isoformat()
                feedback_payload = {
                    "run_id": run_id,
                    "timestamp_utc": timestamp_uk,  # Note: keeping field name as timestamp_utc for compatibility
                    "reviewer_name": reviewer_name,
                    "reviewer_email": reviewer_email,
                    "stage": stage or "",
                    "accuracy": accuracy_score,
                    "relevance": relevance_score,
                    "writing_style": writing_score,
                    "presentation": presentation_score,
                    "ease_of_use": ease_score,
                    "remarks": reviewer_remarks,
                    "pdf_filename": download_name,
                }
                st.write("Feedback payload:", feedback_payload)

                try:
                    append_feedback_to_sheet(feedback_payload)
                except Exception as exc:  # pragma: no cover - runtime feedback
                    st.error(f"Could not record feedback: {exc}")
                else:
                    st.success("Feedback recorded successfully.")
                    st.balloons()

    # Show PDF and review form if data has been submitted
    if "latest_run" in st.session_state:
        st.divider()
        
        # Display the extracted summaries
        if "extracted_summaries" in st.session_state:
            st.subheader("Extracted Summaries")
            summaries = st.session_state["extracted_summaries"]
            st.success(f"School facts: {summaries['school_facts']}")
            st.success(f"Exclusion reason: {summaries['exclusion_reason']}")
            st.success(f"Student perspective: {summaries['student_perspective']}")
        
        st.subheader("Generated Position Statement")
        
        # Display the stored PDF data
        if "latest_pdf_bytes" in st.session_state and "latest_pdf_name" in st.session_state:
            st.download_button(
                label="Download Position Statement PDF",
                data=st.session_state["latest_pdf_bytes"],
                file_name=st.session_state["latest_pdf_name"],
                mime="application/pdf",
            )
            
            st.caption(f"PDF: {st.session_state['latest_pdf_name']}")
        
        # Display the review form
        st.subheader("Reviewer Evaluation")
        with st.form("evaluation_form"):
            run_id = st.session_state["latest_run"]["run_id"]
            st.markdown(f"Run ID: `{run_id}`")
            
            # Reviewer information
            st.markdown("**Reviewer Information**")
            reviewer_name = st.text_input("Your name", placeholder="Enter your full name")
            reviewer_email = st.text_input("Your email", placeholder="Enter your email address")
            
            st.markdown("**Evaluation Scores**")
            accuracy_score = st.slider("Accuracy of factual content", 0, 10, 5)
            relevance_score = st.slider("Relevance of arguments", 0, 10, 5)
            writing_score = st.slider("Writing style and clarity", 0, 10, 5)
            presentation_score = st.slider("Presentation of document", 0, 10, 5)
            ease_score = st.slider("Ease of using this tool", 0, 10, 5)
            reviewer_remarks = st.text_area("Additional comments or specific issues", height=120)
            submitted_feedback = st.form_submit_button("Submit feedback")

        if submitted_feedback:
            # Use UK timezone instead of UTC
            uk_tz = pytz.timezone('Europe/London')
            timestamp_uk = datetime.now(uk_tz).isoformat()
            feedback_payload = {
                "run_id": run_id,
                "timestamp_utc": timestamp_uk,  # Note: keeping field name as timestamp_utc for compatibility
                "reviewer_name": reviewer_name,
                "reviewer_email": reviewer_email,
                "stage": st.session_state["latest_run"]["stage"],
                "accuracy": accuracy_score,
                "relevance": relevance_score,
                "writing_style": writing_score,
                "presentation": presentation_score,
                "ease_of_use": ease_score,
                "remarks": reviewer_remarks,
                "pdf_filename": st.session_state["latest_run"]["pdf_filename"],
            }
            st.write("Feedback payload:", feedback_payload)

            try:
                append_feedback_to_sheet(feedback_payload)
            except Exception as exc:  # pragma: no cover - runtime feedback
                st.error(f"Could not record feedback: {exc}")
            else:
                st.success("Feedback recorded successfully.")
                st.balloons()