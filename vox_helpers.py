import os

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def compute_background_summary(responses):
    """Build a background summary list similar to the front-end helper."""
    summary = []

    is_send = str(responses.get("is_send", "")).strip().lower() == "yes"
    send_details = (responses.get("send_details") or "").strip()
    ehcp_details = (responses.get("ehcp_details") or "").strip()
    is_ethnic_minority = str(responses.get("is_ethnic_minority", "")).strip().lower() == "yes"
    previous_suspensions_details = (responses.get("previous_suspensions_details") or "").strip()
    family_awareness_details = (responses.get("family_awareness_details") or "").strip()
    personal_issues_details = (responses.get("personal_issues_details") or "").strip()

    if is_send:
        summary.append("Young person has SEND.")
        if send_details:
            summary.append(f"SEND Details: {send_details}.")
        if ehcp_details:
            summary.append(f"EHCP Details: {ehcp_details}.")
    else:
        summary.append("Young person does NOT have SEND.")

    if is_ethnic_minority:
        summary.append("Young person is from ethnic minority background.")
    else:
        summary.append("Young person is NOT from ethnic minority background.")

    if previous_suspensions_details:
        summary.append(f"Previous suspensions: {previous_suspensions_details}.")
    else:
        summary.append("No previous suspensions mentioned.")

    if family_awareness_details:
        summary.append(
            "Family awareness of behavioural issues, or the risk of exclusion before it happened: "
            f"{family_awareness_details}."
        )
    else:
        summary.append(
            "No family awareness of behavioural issues, or the risk of exclusion before it happened details provided."
        )

    if personal_issues_details:
        summary.append(f"Personal issues: {personal_issues_details}.")
    else:
        summary.append("No personal issues mentioned.")

    return summary

def _get_ollama_api_key():
    """Load the Ollama API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        return api_key
        
    OLLAMA_API_KEY = st.secrets["ollama"]["api_key"]
    if OLLAMA_API_KEY:
        return OLLAMA_API_KEY
        
    if HAS_STREAMLIT and "ollama_api_key" in st.secrets:
        secret_value = st.secrets["ollama_api_key"]
        if isinstance(secret_value, str):
            return secret_value
        raise TypeError("Streamlit secret 'ollama_api_key' must be a string.")
    raise RuntimeError(
        "Ollama API key is not configured. Set the OLLAMA_API_KEY environment variable "
        "or define st.secrets['ollama_api_key']."
    )

def _get_openai_api_key():
    """Load the OpenAI API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    if HAS_STREAMLIT:
        try:
            openai_section = st.secrets["openai"]["api_key"]
        except Exception:
            openai_section = None
        if openai_section:
            if isinstance(openai_section, str):
                return openai_section
            raise TypeError("Streamlit secret 'openai.api_key' must be a string.")

        if "openai_api_key" in st.secrets:
            secret_value = st.secrets["openai_api_key"]
            if isinstance(secret_value, str):
                return secret_value
            raise TypeError("Streamlit secret 'openai_api_key' must be a string.")

    raise RuntimeError(
        "OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable, "
        "define st.secrets['openai']['api_key'], or define st.secrets['openai_api_key']."
    )

def _normalise_context(context):
    """Ensure all prompt variables are strings to keep str.format happy."""
    normalised = {}
    for key, value in context.items():
        if value is None:
            normalised[key] = "Not provided"
        else:
            normalised[key] = str(value)
    return normalised


def _compose_guidance_query(exclusion_reason, school_facts, student_perspective, background_summary, stage_info, other_information_provided, exclusion_letter_date, specific_instructions):
    parts = [
        exclusion_reason,
        school_facts,
        student_perspective,
        background_summary,
        stage_info,
        other_information_provided,
        exclusion_letter_date,
        specific_instructions,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
