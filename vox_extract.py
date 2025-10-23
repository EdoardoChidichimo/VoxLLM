from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from rag_index import GuidanceRetriever, format_behaviour_block, format_suspensions_block


PROMPTS_DIR = Path("prompts")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

try:
    import streamlit as st  # type: ignore

    HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    HAS_STREAMLIT = False


system_messages = {
    "extract_school_facts": (
        "You are a legal expert specialising in UK school exclusion law. "
        "Your role is to synthesise information from multiple sources to create a clear, factual summary of the school's position in an exclusion case. "
        "Be objective, accurate, and focus on identifying key facts and evidence."
    ),
    "extract_exclusion_reason": (
        "You are a legal expert specialising in UK school exclusion law. "
        "Your role is to extract and clearly state the specific reason(s) given for a school exclusion from official documentation. "
        "Be precise and identify all stated reasons."
    ),
    "extract_student_perspective": (
        "You are a legal expert specialising in UK school exclusion law. "
        "Your role is to synthesise information about the student's perspective and whether proper procedures were followed."
    ),
    "generate_position_statement": (
        "You are a legal advocate specialising in UK school exclusion appeals. "
        "Draft a position statement grounded strictly in the provided facts, statutory guidance excerpts, and procedural information. "
        "Follow the template and citation requirements exactly."
    ),
}

_guidance_retriever: Optional[GuidanceRetriever] = None


def get_guidance_retriever() -> GuidanceRetriever:
    global _guidance_retriever
    if _guidance_retriever is None:
        _guidance_retriever = GuidanceRetriever()
    return _guidance_retriever


def _get_ollama_api_key() -> str:
    """Load the Ollama API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        return api_key
    if HAS_STREAMLIT and "ollama_api_key" in st.secrets:
        secret_value = st.secrets["ollama_api_key"]
        if isinstance(secret_value, str):
            return secret_value
        raise TypeError("Streamlit secret 'ollama_api_key' must be a string.")
    raise RuntimeError(
        "Ollama API key is not configured. Set the OLLAMA_API_KEY environment variable "
        "or define st.secrets['ollama_api_key']."
    )


def call_llm(system_message: str, prompt: str) -> str:
    """Call Ollama chat endpoint with a simple system/user payload."""
    api_key = _get_ollama_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    text = response.json()["message"]["content"]
    return text


def _normalise_context(context: Dict[str, Any]) -> Dict[str, str]:
    """Ensure all prompt variables are strings to keep str.format happy."""
    normalised: Dict[str, str] = {}
    for key, value in context.items():
        if value is None:
            normalised[key] = "Not provided"
        else:
            normalised[key] = str(value)
    return normalised


def build_prompt(template_filename: str, **template_context: Any) -> str:
    """Load a prompt template and interpolate the provided variables."""
    template_path = PROMPTS_DIR / template_filename
    template_text = template_path.read_text(encoding="utf-8")
    context = _normalise_context(template_context)

    # Protect placeholder tokens before escaping braces in the template.
    placeholder_tokens: Dict[str, str] = {}
    for key in context.keys():
        token = f"__PLACEHOLDER_{key.upper()}__"
        placeholder_tokens[f"{{{key}}}"] = token
        template_text = template_text.replace(f"{{{key}}}", token)

    template_text = template_text.replace("{", "{{").replace("}", "}}")

    for original, token in placeholder_tokens.items():
        template_text = template_text.replace(token, original)

    return template_text.format(**context)


def extract_school_facts(exclusion_letter_content: str, school_version_events: str, school_evidence: str) -> str:
    system_message = system_messages["extract_school_facts"]
    prompt = build_prompt(
        "extract_school_facts.txt",
        exclusion_letter_content=exclusion_letter_content,
        school_version_events=school_version_events,
        school_evidence=school_evidence,
    )
    return call_llm(system_message, prompt)


def extract_exclusion_reason(exclusion_letter_content: str) -> str:
    system_message = system_messages["extract_exclusion_reason"]
    prompt = build_prompt(
        "extract_exclusion_reason.txt",
        exclusion_letter_content=exclusion_letter_content,
    )
    return call_llm(system_message, prompt)


def extract_student_perspective(
    student_agrees_with_school: str,
    student_version_events: str,
    witnesses_details: str,
    student_voice_heard_details: str,
) -> str:
    system_message = system_messages["extract_student_perspective"]
    prompt = build_prompt(
        "extract_parents_facts.txt",
        student_agrees_with_school=student_agrees_with_school,
        student_version_events=student_version_events,
        witnesses_details=witnesses_details,
        student_voice_heard_details=student_voice_heard_details,
    )
    return call_llm(system_message, prompt)


def extract_all(
    exclusion_letter_content: str,
    school_version_events: str,
    school_evidence: str,
    student_agrees_with_school: str,
    student_version_events: str,
    witnesses_details: str,
    student_voice_heard_details: str,
) -> Tuple[str, str, str]:
    school_facts = extract_school_facts(exclusion_letter_content, school_version_events, school_evidence)
    exclusion_reason = extract_exclusion_reason(exclusion_letter_content)
    student_perspective = extract_student_perspective(
        student_agrees_with_school,
        student_version_events,
        witnesses_details,
        student_voice_heard_details,
    )
    return school_facts, exclusion_reason, student_perspective


def _compose_guidance_query(
    exclusion_reason: str,
    school_facts: str,
    student_perspective: str,
    background_summary: str,
    stage_info: str,
    other_information_provided: str,
    exclusion_letter_date: str,
    specific_instructions: str,
) -> str:
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


def build_guidance_context(
    exclusion_reason: str,
    school_facts: str,
    student_perspective: str,
    background_summary: str,
    stage_info: str,
    other_information_provided: str,
    exclusion_letter_date: str,
    specific_instructions: str,
    behaviour_top_k: int = 4,
    suspensions_top_k: int = 6,
) -> Dict[str, str]:
    retriever = get_guidance_retriever()
    query = _compose_guidance_query(
        exclusion_reason,
        school_facts,
        student_perspective,
        background_summary,
        stage_info,
        other_information_provided,
        exclusion_letter_date,
        specific_instructions,
    )

    blocks = retriever.build_context_blocks(
        query=query,
        behaviour_top_k=behaviour_top_k,
        suspensions_top_k=suspensions_top_k,
    )

    behaviour_blocks = [format_behaviour_block(block) for block in blocks["behaviour"]]
    suspensions_blocks = [format_suspensions_block(block) for block in blocks["suspensions"]]

    behaviour_text = "\n\n---\n\n".join(behaviour_blocks) if behaviour_blocks else "No relevant Behaviour in Schools passages were retrieved."
    suspensions_text = "\n\n---\n\n".join(suspensions_blocks) if suspensions_blocks else "No relevant Suspensions guidance passages were retrieved."

    return {
        "behaviour_in_schools": behaviour_text,
        "suspensions": suspensions_text,
    }


def generate_position_statement(exclusion_reason, school_facts, student_perspective, background_summary, stage_info, other_information_provided, exclusion_letter_date, specific_instructions, position_statement_grounds):
    guidance_context = build_guidance_context(
        exclusion_reason=exclusion_reason,
        school_facts=school_facts,
        student_perspective=student_perspective,
        background_summary=background_summary,
        stage_info=stage_info,
        other_information_provided=other_information_provided,
        exclusion_letter_date=exclusion_letter_date,
        specific_instructions=specific_instructions,
    )

    prompt = build_prompt(
        "create_position_statement.txt",
        exclusion_reason=exclusion_reason,
        school_facts=school_facts,
        student_perspective=student_perspective,
        background_summary=background_summary,
        suspensions_guidance=guidance_context["suspensions"],
        behaviour_in_schools_guidance=guidance_context["behaviour_in_schools"],
        position_statement_grounds=position_statement_grounds,
        stage_info=stage_info,
        other_information_provided=other_information_provided,
        exclusion_letter_date=exclusion_letter_date,
        specific_instructions=specific_instructions,
    )

    system_message = system_messages["generate_position_statement"]
    position_statement = call_llm(system_message, prompt)

    # Log guidance context for audit purposes.
    print("=== Guidance Context: Behaviour in Schools ===")
    print(guidance_context["behaviour_in_schools"])
    print("=== Guidance Context: Suspensions Guidance ===")
    print(guidance_context["suspensions"])

    return position_statement, guidance_context
