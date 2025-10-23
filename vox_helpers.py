from __future__ import annotations

from typing import Any, Dict, List


def compute_background_summary(responses: Dict[str, Any]) -> List[str]:
    """Build a background summary list similar to the front-end helper."""
    summary: List[str] = []

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
