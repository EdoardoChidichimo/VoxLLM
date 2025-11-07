import json
import re
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

PLACEHOLDER_NAMES = (
    "CHILD_NAME",
    "PARENT_NAME",
    "SCHOOL_NAME",
    "EXCLUSION_DATE",
    "EXCLUSION_LETTER_DATE",
    "STAGE",
)
BRACKET_PLACEHOLDER_PATTERN = re.compile(r"\[(" + "|".join(PLACEHOLDER_NAMES) + r")\]")
ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<<(" + "|".join(PLACEHOLDER_NAMES) + r")>>")
AT_PLACEHOLDER_PATTERN = re.compile(r"@@(" + "|".join(PLACEHOLDER_NAMES) + r")@@")
OPENING_SINGLE_QUOTE_PATTERN = re.compile(r"(^|[\s(\[{<])'(?=\S)")


def fix_opening_single_quotes(text):
    """Replace straight opening single quotes with LaTeX-friendly backticks."""
    if not text:
        return text
    return OPENING_SINGLE_QUOTE_PATTERN.sub(lambda match: match.group(1) + "`", text)


@dataclass
class RenderedPositionStatement:
    json_payload: dict
    tex_path: Path
    pdf_path: Path
    log_path: Path


def escape_latex(value):
    """Escape LaTeX special characters in the provided string."""
    if value is None:
        return ""
    escaped = []
    for char in value:
        if char in LATEX_SPECIAL_CHARS:
            escaped.append(LATEX_SPECIAL_CHARS[char])
        else:
            escaped.append(char)
    return "".join(escaped)


def replace_newlines(value):
    """Convert newlines to LaTeX line breaks."""
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\\ ")


def resolve_placeholders(text, values):
    """Replace placeholder tokens such as [CHILD_NAME] or <<CHILD_NAME>>."""

    def replace_match(match):
        placeholder = match.group(1)
        replacement = values.get(placeholder) or ""
        return replacement

    text = BRACKET_PLACEHOLDER_PATTERN.sub(replace_match, text)
    text = ANGLE_PLACEHOLDER_PATTERN.sub(replace_match, text)
    text = AT_PLACEHOLDER_PATTERN.sub(replace_match, text)
    return text


def extract_json_from_response(raw_response):
    """Extract a JSON object from an LLM response that may use fenced code blocks."""
    if not raw_response:
        raise ValueError("Empty response from language model.")

    code_block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    if code_block_match:
        json_text = code_block_match.group(1)
    else:
        json_text = raw_response.strip()

    # Try to parse the JSON as-is first
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        # If parsing fails, try to fix common issues
        print(f"JSON parsing failed: {exc}")
        print(f"Raw JSON text (first 500 chars): {json_text[:500]}")
        
        # Try to fix common JSON issues
        fixed_json = _attempt_json_fixes(json_text)
        if fixed_json != json_text:
            print("Attempting to fix JSON issues...")
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as fixed_exc:
                print(f"Fixed JSON still failed: {fixed_exc}")
        
        # If all else fails, show the problematic JSON for debugging
        raise ValueError(f"Unable to parse JSON position statement: {exc}\n\nRaw JSON text:\n{json_text}")


def _attempt_json_fixes(json_text):
    """Attempt to fix common JSON issues in LLM-generated JSON."""
    fixed = json_text
    
    # Fix missing commas between array elements
    # Look for patterns like "}" followed by "{" without a comma
    fixed = re.sub(r'}\s*\n\s*{', '},\n{', fixed)
    
    # Fix missing commas between object properties
    # Look for patterns like '"key": value' followed by '"key": value' without comma
    fixed = re.sub(r'("\s*:\s*[^,}\]]+)\s*\n\s*(")', r'\1,\n\2', fixed)
    
    # Fix specific comma delimiter issues - look for missing commas before closing quotes
    # Pattern: "value" followed by "key" without comma
    fixed = re.sub(r'("\s*)\n\s*(")', r'\1,\n\2', fixed)
    
    # Fix missing commas after string values in arrays/objects
    # Pattern: "string" followed by "string" without comma
    fixed = re.sub(r'("\s*)\s*\n\s*(")', r'\1,\n\2', fixed)
    
    # Fix trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Fix missing quotes around keys (basic cases)
    fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)
    
    return fixed


def _format_ground_titles(grounds, placeholder_values):
    items = []
    for ground in grounds:
        title = ground.get("ground_title", "")
        resolved_title = resolve_placeholders(title, placeholder_values)
        normalized_title = fix_opening_single_quotes(resolved_title)
        escaped_title = replace_newlines(escape_latex(normalized_title))
        items.append(f"\\item {escaped_title}")
    return "\n".join(items)


def _format_ground_content(grounds, placeholder_values):
    sections = []
    for idx, ground in enumerate(grounds):
        number = ground.get("ground_number")
        title = ground.get("ground_title", "")
        bullets = ground.get("bullets") or []

        resolved_title = resolve_placeholders(title, placeholder_values)
        normalized_title = fix_opening_single_quotes(resolved_title)
        escaped_title = replace_newlines(escape_latex(normalized_title))
        heading = f"\\section*{{\\raggedright Ground {number}: {escaped_title}}}"

        enumerate_options = (
            "[label=\\arabic*., leftmargin=6ex, start=4, series=main]"
            if idx == 0
            else "[label=\\arabic*., leftmargin=6ex, resume*=main]"
        )

        bullet_lines = []
        for bullet in bullets:
            if isinstance(bullet, Mapping):
                bullet_data = bullet
            elif isinstance(bullet, str):
                bullet_data = {"type": "text", "content": bullet}
            else:
                bullet_data = {}

            content = bullet_data.get("content", "")
            reference = bullet_data.get("reference")
            bullet_type = (bullet_data.get("type") or "").lower()

            resolved_content = resolve_placeholders(content, placeholder_values).strip()
            if bullet_type == "quote":
                if resolved_content.startswith('"') and resolved_content.endswith('"'):
                    inner = resolved_content[1:-1].strip()
                    resolved_content = f"``{inner}''"
                elif resolved_content.startswith("'") and resolved_content.endswith("'"):
                    inner = resolved_content[1:-1].strip()
                    resolved_content = f"`{inner}'"
                else:
                    resolved_content = f"``{resolved_content}''"

            resolved_content = fix_opening_single_quotes(resolved_content)
            escaped_content = replace_newlines(escape_latex(resolved_content))
            line = escaped_content

            if reference:
                resolved_reference = resolve_placeholders(reference, placeholder_values)
                resolved_reference = fix_opening_single_quotes(resolved_reference)
                escaped_reference = replace_newlines(escape_latex(resolved_reference))
                line = f"{line} {escaped_reference}"

            if not line:
                line = " "

            bullet_lines.append(f"\\item {line}")

        if not bullet_lines:
            bullet_lines.append("\\item ")

        bullet_block = "\n".join(bullet_lines)
        sections.append(
            "\n".join(
                [
                    heading,
                    f"\\begin{{enumerate}}{enumerate_options}",
                    bullet_block,
                    "\\end{enumerate}",
                ]
            )
        )

    return "\n\n".join(sections)


def _load_template(template_path):
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _write_tex_file(tex_content, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    return tex_path


def _compile_tex_to_pdf(tex_path, output_dir):
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        str(tex_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    log_path = output_dir / f"{tex_path.stem}.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"LaTeX compilation failed. Review the log at: {log_path}"
        )

    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("LaTeX compilation reported success but PDF was not created.")

    return pdf_path, log_path


def render_position_statement_pdf(
    position_statement,
    user_details,
    template_path=Path("documents/position_statement_output_template.tex"),
    output_dir=Path("output") / "position_statements",
):
    template_text = _load_template(template_path)

    placeholder_values = {
        "CHILD_NAME": user_details.get("child_name") or "",
        "PARENT_NAME": user_details.get("parent_name") or "",
        "SCHOOL_NAME": user_details.get("school_name") or "",
        "EXCLUSION_DATE": user_details.get("exclusion_date") or "",
        "EXCLUSION_LETTER_DATE": user_details.get("exclusion_letter_date") or "",
        "STAGE": user_details.get("stage") or "",
    }
    
    # Debug output to help identify placeholder substitution issues
    print("=== PLACEHOLDER SUBSTITUTION DEBUG ===")
    print(f"User details received: {user_details}")
    print(f"Placeholder values: {placeholder_values}")
    for key, value in placeholder_values.items():
        if not value:
            print(f"WARNING: {key} is empty!")

    escaped_placeholders = {
        key: replace_newlines(escape_latex(fix_opening_single_quotes(value)))
        for key, value in placeholder_values.items()
    }

    grounds = position_statement.get("grounds", [])
    grounds_titles = _format_ground_titles(grounds, placeholder_values)
    grounds_content = _format_ground_content(grounds, placeholder_values)

    replacements = {
        "@@CHILD_NAME@@": escaped_placeholders["CHILD_NAME"],
        "@@PARENT_NAME@@": escaped_placeholders["PARENT_NAME"],
        "@@SCHOOL_NAME@@": escaped_placeholders["SCHOOL_NAME"],
        "@@STAGE@@": escaped_placeholders["STAGE"],
        "@@EXCLUSION_DATE@@": escaped_placeholders["EXCLUSION_DATE"],
        "@@EXCLUSION_LETTER_DATE@@": escaped_placeholders["EXCLUSION_LETTER_DATE"],
        "@@GROUNDS_TITLES@@": grounds_titles,
        "@@GROUNDS_CONTENT@@": grounds_content,
    }

    tex_content = template_text
    for placeholder, value in replacements.items():
        tex_content = tex_content.replace(placeholder, value)
    
    # Debug: Check if any placeholders remain
    remaining_placeholders = []
    for placeholder in ["@@CHILD_NAME@@", "@@PARENT_NAME@@", "@@SCHOOL_NAME@@", 
                       "@@EXCLUSION_DATE@@", "@@EXCLUSION_LETTER_DATE@@", "@@STAGE@@"]:
        if placeholder in tex_content:
            remaining_placeholders.append(placeholder)
    
    if remaining_placeholders:
        print(f"ERROR: Placeholders not substituted: {remaining_placeholders}")
    else:
        print("SUCCESS: All placeholders successfully substituted")

    unique_stem = f"position_statement_{uuid.uuid4().hex[:8]}"
    tex_path = _write_tex_file(tex_content, output_dir, unique_stem)
    pdf_path, log_path = _compile_tex_to_pdf(tex_path, output_dir)

    return RenderedPositionStatement(
        json_payload=position_statement,
        tex_path=tex_path,
        pdf_path=pdf_path,
        log_path=log_path,
    )
