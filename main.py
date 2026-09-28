import requests
import os
import json
import pandas as pd
import re
import time

from dotenv import load_dotenv
from yandex_gpt import YandexGPTConfigManagerForAPIKey, YandexGPT
from datasets import load_dataset
from typing import Optional
from tqdm import tqdm

def load_legal_dataset(split: str = "train", limit: int = 7) -> list:

    dataset = load_dataset("TryDotAtwo/russian-legal-ner", split=split)
    texts = dataset["text"][:limit]
    return texts

def call_yandexgpt(system_prompt: str, user_prompt: str) -> str:
   
    headers = {
        "Authorization": f"Api-Key {YC_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 2000
        },
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if "result" not in result or "alternatives" not in result["result"]:
        raise RuntimeError(f"Unexpected API response: {result}")

    return result["result"]["alternatives"][0]["message"]["text"]
def parse_json_response(raw_response: str) -> dict:
    cleaned = re.sub(r"```json\s*", "", raw_response)  
    cleaned = re.sub(r"```\s*$", "", cleaned)           

    cleaned = cleaned.strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    
def truncate_text(text: str, max_length: int = 8000) -> str:
    if len(text) > max_length:
        return text[:max_length]
    return text


def main():
    texts = load_legal_dataset(split="train", limit=7)
    #Текст без указания судьи и сторон
    test_texts1= """ПОСТАНОВЛЕНИЕ
    о возбуждении дела об административном правонарушении
    Дело № 5-412/2024
    «12» марта 2024 года

    Рассмотрев материалы проверки по факту нарушения правил дорожного движения,
    установил:

    Водитель транспортного средства совершил выезд на полосу встречного движения
    в нарушение требований дорожной разметки. Указанные действия образуют состав
    административного правонарушения, предусмотренного частью 4 статьи 12.15
    Кодекса Российской Федерации об административных правонарушениях.

    На основании изложенного и руководствуясь статьями 28.1, 29.9 КоАП РФ,
    постановил:

    Признать виновным в совершении административного правонарушения и назначить
    административное наказание в виде административного штрафа в размере
    5 000 (пяти тысяч) рублей.

    Постановление может быть обжаловано в течение десяти суток со дня вручения
    или получения копии постановления.

    Срок исполнения постановления — 60 дней с момента вступления в законную силу."
    """
    #Текст с опечатками
    test_texts2= """ПОСТАНОВЛЕНИЕ
    по делу об административном правонарушении

    Дело № 4-12З/202З

    «О5» апреля 202З года
    г. Екатиринбург

    Судья Левченко Ирина Петровнa, рассмотрев материалы дела об административном
    правонарушении, предусмотренном статьёй 14.16 КоАП РФ, в отношении
    гражданина Смирнова Алексея Викторовича,

    установил:

    ООО «Ромашка-Торг» (ИНН 7701234567) допустило реализацию алкогольной
    продукции без сопроводительных документов, подтверждающих легальность её
    производства и оборота. Факт нарушения выявлен 27.О3.202З в помещении
    магазина, расположенного по адресу: г. Екатиринбург, ул. Ленина, д. 15.

    В судебном заседании представитель ООО «Ромашка-Торг» вину признал.

    На основании изложенного, руководствуясь ст. 29.9, 29.1О КоАП РФ,

    постановил:

    Признать ООО «Ромашка-Торг» виновным в совершении административного
    правонарушения и назначить наказание в виде административного штрафа
    в размере 3ОО ООО (трёхсот тысяч) рублей.

    Постановление может быть обжаловано в течение 1О суток со дня вручения
    копии постановления.

    Срок оплаты штрафа — 6О дней со дня вступления постановления в законную силу."""
    
    results = []
    results1 = []
    results2 = []
    for idx, raw_text in enumerate(texts):
        truncated = truncate_text(raw_text)
        user_prompt = USER_PROMPT_TEMPLATE.format(text=truncated)
        raw_response = call_yandexgpt(SYSTEM_PROMPT, user_prompt)
        extracted = parse_json_response(raw_response)
        record = {
            "doc_index": idx,          
            "extracted": extracted  
        }
        results.append(record)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    
    truncated = truncate_text(test_texts1)
    user_prompt = USER_PROMPT_TEMPLATE.format(text=truncated)
    raw_response = call_yandexgpt(SYSTEM_PROMPT, user_prompt)
    extracted = parse_json_response(raw_response)
    record = {
        "doc_index": idx,          
        "extracted": extracted  
    }
    results1.append(record)
    with open("results_test1.json", "w", encoding="utf-8") as f:
        json.dump(results1, f, ensure_ascii=False, indent=2)
        
    truncated = truncate_text(test_texts2)
    user_prompt = USER_PROMPT_TEMPLATE.format(text=truncated)
    raw_response = call_yandexgpt(SYSTEM_PROMPT, user_prompt)
    extracted = parse_json_response(raw_response)
    record = {
        "doc_index": idx,          
        "extracted": extracted  
    }
    results2.append(record)
    with open("results_test2.json", "w", encoding="utf-8") as f:
        json.dump(results2, f, ensure_ascii=False, indent=2)    
    
        

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")


API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
MODEL_URI = f"gpt://{YC_FOLDER_ID}/yandexgpt-lite"

SYSTEM_PROMPT = """Ты — строгий и точный AI-помощник юриста.
Твоя задача — извлекать факты из юридического текста без искажений. 
Не выдумывай данные. Если сведения отсутствуют, верни null. Отвечай только валидным JSON-объектом. 
Не добавляй пояснения, Markdown, комментарии или текст до/после JSON"""

USER_PROMPT_TEMPLATE = """Извлеки из текста юридического документа следующие сущности:

- Тип дела: к какому судопроизводству относится дело
- Номер дела:
- ФИО судьи: кто вынес решение
- Стороны судебного процесса:
- Упомянутая локация
- Даты: дата подписания, сроки действия

Пример желаемого ответа (few-shot):
{{
  "type of case: ["Административное"],
  "case_number: [2-412/2012]
  "Judge": ["Иванов И.И."],
  "litigator":[Бобр И.И]
  "location": "Москва"
  "date": "2 марта 2026",
}}  
Текст документа для анализа:
{text}
"""

if __name__ == "__main__":
    main()

