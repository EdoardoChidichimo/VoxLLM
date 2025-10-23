## Vox Prototype

Streamlit-based assistant that helps lawyers prepare position statements by combining structured data capture, Retrieval Augmented Generation (RAG) guidance, and PDF rendering.

### Prerequisites
- Python 3.10+
- Access to an Ollama-compatible hosted model (HTTP API)
- Google Workspace project with Sheets API enabled and a service account that can edit the feedback spreadsheet
- `pdflatex` executable available on the machine for PDF generation

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Secret & Credential Management
All secrets are loaded from either environment variables or `st.secrets`. **Never** commit real keys or service account files to version control.

#### Streamlit secrets (recommended for Streamlit Cloud/Sharing)
Create `.streamlit/secrets.toml` (ignored by Git):
```toml
[general]

[google_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"

google_sheets_spreadsheet_id = "your-sheet-id"
google_sheets_range = "Feedback!A:J"  # optional override
ollama_api_key = "sk-your-ollama-key"
```

#### Environment variables (works locally / Docker)
```bash
export OLLAMA_API_KEY="sk-your-ollama-key"
export OLLAMA_API_URL="https://ollama.com/api/chat"   # optional override
export OLLAMA_MODEL="gpt-oss:120b"                    # optional override
export GOOGLE_SHEETS_SPREADSHEET_ID="your-sheet-id"
export GOOGLE_SERVICE_ACCOUNT_FILE="/path/to/service-account.json"
# alternatively:
# export GOOGLE_SERVICE_ACCOUNT_INFO="$(cat /path/to/service-account.json)"
```

Keep the service-account JSON outside the repository (the `.gitignore` excludes common filenames like `perspective-*.json`). Rotate keys and re-deploy if a secret ever leaks.

### Running the App
```bash
streamlit run vox.py
```
The UI will prompt for exclusion case information; after submission it will:
1. Call Ollama for structured summaries
2. Retrieve supporting guidance from the local TF-IDF index
3. Render a PDF position statement via LaTeX
4. Append feedback responses to Google Sheets

Generated PDFs and LaTeX artifacts are stored under a timestamped directory in `./data/`.

### Deployment Checklist
- Populate environment variables or Streamlit secrets on the target hosting platform
- Ensure the Google service account email has edit access to the target spreadsheet
- Confirm outbound HTTPS access is allowed to both Google APIs and `OLLAMA_API_URL`
- Provide reviewers with a secure channel to submit feedback; monitor the Sheet for responses

### Local Utilities
- `guidance_document_prep.py` regenerates the JSONL guidance chunks from source documents
- `position_statement_renderer.py` contains the LaTeX rendering workflow; ensure `pdflatex` remains available

### Security Notes
- Keys are fetched at runtime; if a secret is missing the app raises a clear error instead of failing silently
- Always audit git history before distribution to confirm secrets were never committed
