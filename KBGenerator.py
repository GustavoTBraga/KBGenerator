#!/usr/bin/env python3
"""
Generate FAQs in XML from Zendesk tickets (with attachments) and – optionally – store
them in Google Sheets.

Revision 2025-06-14 c
─────────────────────────────────────────────────────────────────────────────
✔ Corrige erro 400 quando a extensão do anexo está em maiúsculas (.PNG → .png).
  - Filtra anexos pelo sufixo (lista oficial de extensões aceitas pela OpenAI).
  - Normaliza a extensão para minúsculas antes do upload.
  - Mantém logs claros para anexos enviados (✔) ou ignorados (⏭).

⚠ Não houve outras mudanças funcionais.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import gspread
import openai
import requests
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("faq")

# ---------------------------------------------------------------------------
# Zendesk configuration (token ainda hard-coded; securizar depois!)
# ---------------------------------------------------------------------------
ZENDESK_AUTH_B64 = os.getenv("ZENDESK_AUTH_B64")  # "YXRs…Q=="

if not ZENDESK_AUTH_B64:
    logger.error("ZENDESK_AUTH_B64 não configurada")
    raise SystemExit(1)

ZENDESK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Basic {ZENDESK_AUTH_B64}",
}
ZENDESK_BASE_URL = "https://britech.zendesk.com/api/v2"

# ---------------------------------------------------------------------------
# Extensões aceitas pela OpenAI (minúsculas, sem ponto)
# Fonte: https://platform.openai.com/docs/guides/file-input
# ---------------------------------------------------------------------------
OPENAI_FILE_EXTS = {
    "c", "cpp", "css", "csv", "doc", "docx", "gif", "go", "html", "java",
    "jpeg", "jpg", "js", "json", "md", "pdf", "php", "pkl", "png", "pptx",
    "py", "rb", "tar", "tex", "ts", "txt", "webp", "xlsx", "xml", "zip",
}


# ---------------------------------------------------------------------------
# Zendesk helpers
# ---------------------------------------------------------------------------
def buscar_comentarios_zendesk(ticket_id: str) -> Dict[str, Any] | None:
    """Fetches all comments from a single Zendesk ticket."""
    url = f"{ZENDESK_BASE_URL}/tickets/{ticket_id}/comments"
    try:
        resp = requests.get(url, headers=ZENDESK_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("[%s] Zendesk: %s", ticket_id, exc)
        return None


def extrair_comentarios(json_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts text and attachments list from each comment."""
    return [
        {
            "texto": c.get("body", ""),
            "attachments": c.get("attachments", []),
        }
        for c in json_resp.get("comments", [])
    ]


# ---------------------------------------------------------------------------
# Helpers de anexos
# ---------------------------------------------------------------------------
def _is_supported(att: Dict[str, Any]) -> bool:
    """True se o anexo tiver extensão suportada pelo endpoint /files."""
    ext = Path(att.get("file_name", "")).suffix.lstrip(".").lower()
    return ext in OPENAI_FILE_EXTS


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------
def upload_attachments_openai(
    attachments: List[Dict[str, Any]],
    client: openai.OpenAI,
) -> List[str]:
    """Faz upload dos anexos compatíveis e devolve a lista de file_id."""
    file_ids: List[str] = []

    for att in attachments:
        if not _is_supported(att):
            logger.info(
                "⏭  Ignorando anexo não suportado (%s)",
                att.get("file_name"),
            )
            continue

        url = att.get("content_url")
        if not url:
            continue

        original_name = att.get("file_name", "attachment")
        base, ext = os.path.splitext(original_name)
        safe_name = f"{base}{ext.lower()}"  # garante .png, .pdf, etc.

        tmp_path: Path | None = None
        try:
            # 1) Cria arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_name}") as tmp:
                tmp_path = Path(tmp.name)

            # 2) Faz download stream
            with requests.get(url, headers=ZENDESK_HEADERS, stream=True, timeout=60) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)

            # 3) Upload para a OpenAI
            with tmp_path.open("rb") as fh:
                upload = client.files.create(file=fh, purpose="assistants")

            logger.info(
                "✔ %s enviado (id=%s, %d bytes)",
                safe_name, upload.id, upload.bytes,
            )
            file_ids.append(upload.id)

        except Exception as exc:
            logger.warning("Falha anexo %s – %s", url, exc)

        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    return file_ids


def build_multimodal_content(texto: str, file_ids: List[str]) -> List[Dict[str, Any]]:
    """Creates the multimodal content array."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": texto}]
    for fid in file_ids:
        parts.append({"type": "file", "file": {"file_id": fid}})
    return parts


def gerar_faq_openai(comentarios: List[Dict[str, Any]], ticket_id: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY não configurada!")
        return None

    client = openai.OpenAI(api_key=api_key)

    system_prompt = (
        "Você é um assistente de IA que recebe comentários de tickets e gera uma FAQ em XML.\n"
        "Para cada comentário, crie um <item> com <question>/<resposta> usando SOMENTE o conteúdo "
        "do comentário e/ou anexos.\n"
        "Formato final:\n<Ticket>\n  <idTicket>{{ticket_id}}</idTicket>\n  <knowledge>...\n"
        "</knowledge>\n</Ticket>\n"
        "Substitua nomes reais por usuário1, usuário2…\n"
        "Não invente fatos além do ticket."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt.replace("{{ticket_id}}", str(ticket_id))}
    ]

    for item in comentarios:
        fids = upload_attachments_openai(item["attachments"], client)
        messages.append({"role": "user", "content": build_multimodal_content(item["texto"], fids)})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=10000,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("OpenAI: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Google Sheets helper (opcional)
# ---------------------------------------------------------------------------
def registrar_google_sheets(ticket_id: str, faq: str) -> bool:
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if not creds_json:
        logger.error("GOOGLE_CREDS_JSON ausente.")
        return False
    try:
        creds = Credentials.from_service_account_file(
            creds_json,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        ws = (
            gspread.authorize(creds)
            .open_by_key("1bsVVg2kCWWcA7tfAK7qEJXnFLqIC-E7MYmlZ5vj1xog")
            .worksheet("Página1")
        )
        ws.append_row([ticket_id, faq])
        return True
    except Exception as exc:
        logger.error("Sheets: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    csv_path = Path("RESOLVIDOS.CSV")
    if not csv_path.exists():
        logger.error("CSV %s não encontrado.", csv_path)
        return

    with csv_path.open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = [f.lstrip("\ufeff").strip() for f in reader.fieldnames]
        if "ID" not in fieldnames:
            logger.error("Coluna ID ausente. Headers: %s", fieldnames)
            return

        for row in reader:
            tid = row.get("ID") or row.get("\ufeffID")
            if not tid:
                continue
            logger.info("Processando ticket %s…", tid)
            raw = buscar_comentarios_zendesk(tid)
            if raw is None:
                continue
            comentarios = extrair_comentarios(raw)
            if not comentarios:
                logger.warning("%s sem comentários.", tid)
                continue
            faq_xml = gerar_faq_openai(comentarios, tid)
            print(faq_xml)
            # if faq_xml and registrar_google_sheets(tid, faq_xml):
            #     logger.info("✔ Ticket %s gravado.", tid)
            # else:
            #     logger.error("✘ Falha ticket %s.", tid)


if __name__ == "__main__":
    main()
