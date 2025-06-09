import os
import requests
import openai
import gspread
from google.oauth2.service_account import Credentials
import csv

def buscar_comentarios_zendesk(ticket_id):
    url = f"https://britech.zendesk.com/api/v2/tickets/{ticket_id}/comments"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic <YXRsYXNyaXNrQGJyaXRlY2guY29tLmJyL3Rva2VuOjJQbGRLS1hja0NjdmpPbUg0aWlzc1lWQ1NVQlI1RDhyYUt0SGIxeVQ=>',
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'[ERRO][{ticket_id}] Zendesk:', e)
        return None

def extrair_textos_comentarios(comentarios_json):
    try:
        return [c['body'] for c in comentarios_json['comments']]
    except Exception as e:
        print('[ERRO] Ao extrair comentários:', e)
        return []

def gerar_faq_openai(comentarios_formatados, ticket_id):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system_prompt = (
        "Você é um assistente de IA que recebe comentários de tickets e gera uma FAQ em XML.\n"
        "Para cada comentário fornecido, gere um par de <item> com uma pergunta (question) e uma resposta (resposta) baseando-se somente no conteúdo do comentário.\n"
        "A saída deve ter o formato abaixo e conter um <item> para cada comentário recebido:\n"
        "\n"
        "<Ticket>\n"
        "  <idTicket>{{ticket_id}}</idTicket>\n"
        "  <knowledge>\n"
        "    <item>\n"
        "      <question>Pergunta baseada no comentário 1</question>\n"
        "      <resposta>Resposta baseada no comentário 1</resposta>\n"
        "    </item>\n"
        "    <item>\n"
        "      <question>Pergunta baseada no comentário 2</question>\n"
        "      <resposta>Resposta baseada no comentário 2</resposta>\n"
        "    </item>\n"
        "    <!-- ... -->\n"
        "  </knowledge>\n"
        "</Ticket>\n"
        "Nunca use nomes reais de pessoas; utilize usuário1, usuário2, etc.\n"
        "Não invente perguntas ou respostas; apenas utilize o que está nos comentários do ticket."
    )
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt.replace("{{ticket_id}}", str(ticket_id))},
                {"role": "user", "content": f"ticket_id: {ticket_id}\n\n{comentarios_formatados}"},
            ],
            temperature=0,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print('[ERRO] OpenAI:', e)
        return None

def registrar_google_sheets(ticket_id, faq):
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    if not creds_json:
        print('[ERRO] Credenciais Google não configuradas.')
        return False
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    try:
        creds = Credentials.from_service_account_file(creds_json, scopes=scope)
        client = gspread.authorize(creds)
        spreadsheet_id = '1bsVVg2kCWWcA7tfAK7qEJXnFLqIC-E7MYmlZ5vj1xog'
        sheet_name = 'Página1'
        sh = client.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)
        worksheet.append_row([ticket_id, faq])
        return True
    except Exception as e:
        print('[ERRO] Sheets:', e)
        return False

def main():
    arquivo_csv = 'RESOLVIDOS.CSV'
    if not os.path.isfile(arquivo_csv):
        print(f'[ERRO] Arquivo {arquivo_csv} não encontrado.')
        return

    with open(arquivo_csv, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = [name.lstrip('\ufeff').strip() for name in reader.fieldnames]
        if 'ID' not in fieldnames:
            print(f'[ERRO] Coluna ID não encontrada no CSV! Fieldnames lidos: {fieldnames}')
            return
        for row in reader:
            ticket_id = row.get('ID') or row.get('\ufeffID')
            print(f'Processando ticket {ticket_id}...')

            comentarios = buscar_comentarios_zendesk(ticket_id)
            if comentarios is None:
                continue
            comentarios_textos = extrair_textos_comentarios(comentarios)
            if not comentarios_textos:
                print(f"[ERRO] Nenhum comentário encontrado para o ticket {ticket_id}.")
                continue
            comentarios_formatados = "\n".join(
                [f"Comentário {i+1}: {c}" for i, c in enumerate(comentarios_textos)]
            )
            faq = gerar_faq_openai(comentarios_formatados, ticket_id)
            if faq is None:
                continue
            print(faq)
            sucesso = registrar_google_sheets(ticket_id, faq)
            if sucesso:
                print(f'✔ Ticket {ticket_id} registrado com sucesso.')
            else:
                print(f'✘ Falha ao registrar ticket {ticket_id}.')

if __name__ == '__main__':
    main()
