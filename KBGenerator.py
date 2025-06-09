#!/usr/bin/env python3
"""
Generate FAQs in XML from Zendesk tickets and store them in Google Sheets.

Main fixes compared to the original version:
- **Security**: All secrets (Zendesk, OpenAI, Google) are read exclusively from environment variables; nothing hard‑coded.
- **Logging**: Uses the standard `logging` module instead of `print` for structured logs.
- **Retries & Back‑off**: HTTP requests and the OpenAI call are wrapped with exponential back‑off using `requests.adapters.Retry` and `tenacity`.
- **HTTP Connection Re‑use**: A single `requests.Session` is shared across calls.
- **Type Hints & Docstrings**: Improves readability and IDE support.
- **PDF Extraction**: Migrated to `pypdf` (modern fork of PyPDF2).
- **Concurrency**: Optionally processes tickets in parallel via `ThreadPoolExecutor` (configurable).
- **Configurable**: Spreadsheet IDs, sheet names, CSV path, limits, etc., are all parametrised.
- **Resource Cleanup**: Temporary PDFs are always deleted, even on errors.
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import requests
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI
from pypdf import PdfReader
from requests.adapters import HTTPAdapter, Retry
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------- Logging ------------------------------------ #

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("zendesk_faq")

# ---------------------------- Config ------------------------------------- #

ZENDESK_BASE_URL = os.getenv("ZENDESK_BASE_URL", "https://britech.zendesk.com")
ZENDESK_AUTH_B64 = os.getenv("ZENDESK_AUTH_B64")  # email/token Base64
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_CREDS_JSON_B64")  # service‑account JSON, Base64‑encoded
GOOGLE_SPREADSHEET_ID = os.getenv(
    "GOOGLE_SPREADSHEET_ID", "1bsVVg2kCWWcA7tfAK7qEJXnFLqIC-E7MYmlZ5vj1xog"
)
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Página1")

CSV_FILE = os.getenv("CSV_FILE", "RESOLVIDOS.CSV")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
PDF_TOKEN_LIMIT = int(os.getenv("PDF_TOKEN_LIMIT", "7000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ------------------------- HTTP session ---------------------------------- #


def _create_session() -> requests.Session:
    """Return a requests Session pre‑configured with retry/back‑off."""

    session = requests.Session()
    retry_cfg = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    return session


session = _create_session()

# ------------------------- Google Sheets --------------------------------- #


def _load_google_credentials() -> Credentials:
    """Load service‑account credentials from Base64 JSON."""

    if not GOOGLE_CREDENTIALS_B64:
        raise EnvironmentError("Google credentials missing (GOOGLE_CREDS_JSON_B64).")
    creds_dict = json.loads(base64.b64decode(GOOGLE_CREDENTIALS_B64).decode())
    return Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )


def _get_worksheet() -> gspread.Worksheet:
    creds = _load_google_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SPREADSHEET_ID)
    return sheet.worksheet(GOOGLE_SHEET_NAME)


# --------------------------- Zendesk ------------------------------------- #


def _zendesk_headers() -> dict[str, str]:
    if not ZENDESK_AUTH_B64:
        raise EnvironmentError("Zendesk credentials missing (ZENDESK_AUTH_B64).")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {ZENDESK_AUTH_B64}",
    }


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)

def fetch_comments(ticket_id: str) -> dict | None:
    """Return JSON with comments for the given ticket or None on error."""

    url = f"{ZENDESK_BASE_URL}/api/v2/tickets/{ticket_id}/comments"
    try:
        resp = session.get(url, headers=_zendesk_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("Failed to fetch comments for ticket %s: %s", ticket_id, exc)
        return None


def extract_comment_texts(comments_json: dict) -> List[str]:
    """Extract raw body text from Zendesk comments JSON."""

    return [c["body"] for c in comments_json.get("comments", []) if c.get("body")]


def download_first_pdf(comments_json: dict, ticket_id: str) -> Optional[Path]:
    """Download the first PDF attachment found and return its Path."""

    for comment in comments_json.get("comments", []):
        for att in comment.get("attachments", []):
            if att.get("content_type") == "application/pdf":
                content_url = att["content_url"]
                file_name = att.get("file_name") or f"{ticket_id}.pdf"
                target = Path(tempfile.gettempdir()) / f"{ticket_id}_{file_name}"
                try:
                    with session.get(content_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(target, "wb") as fp:
                            for chunk in r.iter_content(chunk_size=8192):
                                fp.write(chunk)
                    logger.debug("PDF saved to %s", target)
                    return target
                except requests.RequestException as exc:
                    logger.warning(
                        "Failed to download PDF for ticket %s: %s", ticket_id, exc
                    )
                    return None
    return None


def extract_pdf_text(pdf_path: Path, token_limit: int = PDF_TOKEN_LIMIT) -> str:
    """Extract text from PDF limited by token count."""

    if not pdf_path or not pdf_path.exists():
        return ""

    chars_limit = token_limit * 4  # rough conversion tokens→chars
    collected: List[str] = []
    total = 0

    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                collected.append(text)
                total += len(text)
                if total > chars_limit:
                    break
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)

    return "\n".join(collected)


# ------------------------- OpenAI ---------------------------------------- #

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "Você é um assistente de IA que recebe comentários de tickets e gera uma FAQ em XML.\n"
    "Para cada comentário fornecido, gere um par de <item> com uma pergunta (question) e uma resposta (resposta)\n"
    "baseando-se somente no conteúdo do comentário.\n"
    "A saída deve seguir este formato e conter um <item> para cada comentário recebido:\n"
    "<Ticket>\n"
    "  <idTicket>{{ticket_id}}</idTicket>\n"
    "  <knowledge>\n"
    "    <item>\n"
    "      <question>Pergunta baseada no comentário 1</question>\n"
    "      <resposta>Resposta baseada no comentário 1</resposta>\n"
    "    </item>\n"
    "    <!-- ... -->\n"
    "  </knowledge>\n"
    "</Ticket>\n"
    "Nunca use nomes reais de pessoas; utilize usuário1, usuário2, etc.\n"
    "Não invente perguntas ou respostas; apenas utilize o que está nos comentários do ticket."
)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)

def generate_faq_xml(ticket_id: str, comments: List[str], pdf_text: str = "") -> str:
    """Generate FAQ XML using the OpenAI Chat Completions API."""

    user_parts = [f"Comentário {i + 1}: {c}" for i, c in enumerate(comments)]
    content = f"ticket_id: {ticket_id}\n\n" + "\n".join(user_parts)
    if pdf_text:
        content += f"\n\nCONTEÚDO DO PDF (texto extraído):\n{pdf_text[:PDF_TOKEN_LIMIT * 4]}"

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{{ticket_id}}", ticket_id)},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------- Google Sheets Write ----------------------------- #


def append_to_sheet(worksheet: gspread.Worksheet, ticket_id: str, faq_xml: str) -> None:
    try:
        worksheet.append_row([ticket_id, faq_xml], value_input_option="RAW")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to append ticket %s to sheet: %s", ticket_id, exc)
        raise


# ------------------------------ Main ------------------------------------ #


def process_ticket(ticket_id: str, worksheet: gspread.Worksheet) -> None:
    logger.info("Processing ticket %s", ticket_id)

    comments_json = fetch_comments(ticket_id)
    if not comments_json:
        return

    comments_texts = extract_comment_texts(comments_json)
    if not comments_texts:
        logger.warning("No comments found for ticket %s", ticket_id)
        return

    pdf_path = download_first_pdf(comments_json, ticket_id)
    pdf_text = extract_pdf_text(pdf_path) if pdf_path else ""

    try:
        faq_xml = generate_faq_xml(ticket_id, comments_texts, pdf_text)
        append_to_sheet(worksheet, ticket_id, faq_xml)
        logger.info("Ticket %s processed successfully.", ticket_id)
    finally:
        if pdf_path and pdf_path.exists():
            pdf_path.unlink(missing_ok=True)


def main() -> None:
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        logger.error("CSV file %s not found.", csv_path)
        return

    worksheet = _get_worksheet()

    with csv_path.open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = [f.strip().lstrip("\ufeff") for f in reader.fieldnames or []]
        id_column = next((name for name in ("ID", "\ufeffID") if name in fieldnames), None)
        if not id_column:
            logger.error("ID column not found in CSV. Fields: %s", fieldnames)
            return

        tickets = [row[id_column] for row in reader if row.get(id_column)]

    if not tickets:
        logger.warning("No tickets found in CSV.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_ticket, tid, worksheet): tid for tid in tickets}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Processing failed for ticket %s: %s", tid, exc)


if __name__ == "__main__":
    main()
