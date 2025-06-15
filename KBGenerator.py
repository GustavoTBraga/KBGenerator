#!/usr/bin/env python3
"""
Generate FAQs in XML from Zendesk tickets – agora enviando imagens como `image_url`
diretamente ao Chat-Completions (sem download/upload).
"""

from __future__ import annotations
import csv, logging, os
from pathlib import Path
from typing import Any, Dict, List

import gspread, openai, requests
from google.oauth2.service_account import Credentials

# ─────────────────── Logging ───────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("faq")

# ─────────────────── Zendesk auth ───────────────
ZENDESK_AUTH_B64 = os.getenv("ZENDESK_AUTH_B64")
if not ZENDESK_AUTH_B64:
    logger.error("ZENDESK_AUTH_B64 não configurada")
    raise SystemExit(1)

ZENDESK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Basic {ZENDESK_AUTH_B64}",
}
ZENDESK_BASE_URL = "https://britech.zendesk.com/api/v2"

# ─────────────────── Extensões suportadas ───────
OPENAI_IMG_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}  # vision-enabled

# ─────────────────── Zendesk helpers ────────────
def buscar_comentarios_zendesk(ticket_id: str) -> Dict[str, Any] | None:
    url = f"{ZENDESK_BASE_URL}/tickets/{ticket_id}/comments"
    try:
        r = requests.get(url, headers=ZENDESK_HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.error("[%s] Zendesk: %s", ticket_id, exc)
        return None

def extrair_comentarios(json_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"texto": c.get("body", ""), "attachments": c.get("attachments", [])}
        for c in json_resp.get("comments", [])
    ]

# ─────────────────── Attachment → parts ─────────
def _make_parts_from_attachments(texto: str, atts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cria partes multimodais usando somente URLs públicas."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": texto}]
    for att in atts:
        url = att.get("content_url")
        if not url:
            continue

        ext = Path(att.get("file_name", "")).suffix.lstrip(".").lower()
        if ext in OPENAI_IMG_EXTS:
            # envia como image_url
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
            logger.info("➜ imagem enviada via URL (%s)", att.get("file_name"))
        else:
            # outros tipos: coloca link no texto
            parts.append({"type": "text", "text": f"[Arquivo disponível]({url})"})
            logger.info("⏭ anexo não-imagem referenciado via link (%s)", att.get("file_name"))
    return parts

# ─────────────────── OpenAI helper ──────────────
def gerar_faq_openai(comentarios: List[Dict[str, Any]], ticket_id: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY não configurada!")
        return None

    client = openai.OpenAI(api_key=api_key)

    system_prompt = (
        "Você é um assistente de IA cuja função é gerar perguntas e respostas para uma base de conhecimento.\n"
        "Você recebe um ticket de Zendesk com as respectivas trocas de mensagens entre o usuário e o agente de atendimento de suporte.\n"
        "Você analisa a troca de mensagens considerando os respectivos anexos. \n"
        "Como resultado da análise você deve produzir um resumo geral e as perguntas e respostas (Q&A) que agreguem à base de conhecimento.\n"
        "Formato final:\n"
        "<Ticket>\n"
        "  <idTicket>{{ticket_id}}</idTicket>\n"
        "  <Summary>\n"
        "  </Summary>\n"
        "  <question>\n"
        "  </question>\n"
        "  <answer>\n"
        "  </answer>\n"
        "  <question>\n"
        "  </question>\n"
        "  <answer>\n"
        "  </answer>\n"
        "  ..."       
        "</Ticket>\n"
        "Não invente fatos. Usar somente o que está descrito nas mensagens."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt.replace("{{ticket_id}}", str(ticket_id))}
    ]

    for item in comentarios:
        messages.append(
            {"role": "user", "content": _make_parts_from_attachments(item["texto"], item["attachments"])}
        )

    try:
        comp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=10_000,
        )
        resposta = comp.choices[0].message.content.strip()
        return resposta
    except Exception as exc:
        logger.error("OpenAI: %s", exc)
        return None

# ─────────────────── Google Sheets (opcional) ──
def registrar_google_sheets(ticket_id: str, faq: str) -> bool:
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if not creds_json:
        logger.error("GOOGLE_CREDS_JSON ausente.")
        return False
    try:
        creds = Credentials.from_service_account_file(
            creds_json, scopes=["https://www.googleapis.com/auth/spreadsheets"]
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

# ─────────────────── Main ───────────────────────
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
