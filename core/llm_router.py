"""Работа с Ollama и роутинг команд (Мозг-Роутер)"""
import json
import re
import base64
import cv2
import requests
from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_NUM_PREDICT,
    VISION_CROP_RATIO, VISION_IMAGE_SIZE, VISION_IMAGE_QUALITY
)


class LLMRouter:
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base

    @staticmethod
    def encode_image(frame):
        """Кодирует кадр в base64 для отправки в LLM"""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        crop = frame[0:int(h * VISION_CROP_RATIO), :]
        small = cv2.resize(crop, VISION_IMAGE_SIZE)
        _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, VISION_IMAGE_QUALITY])
        return base64.b64encode(buffer).decode('utf-8')

    def ask(self, user_text, user_name="Неизвестный", entities_text="", chat_history=None, frame=None):
        """Отправляет запрос в LLM и возвращает JSON-команду"""
        print(f"  [LLM] Запрос: '{user_text[:60]}...'")

        name_info = f"\nИМЯ ТЕКУЩЕГО ПОЛЬЗОВАТЕЛЯ В КАДРЕ: {user_name}" if user_name != "Неизвестный" else ""
        entities_context = f"\n\nУЖЕ СУЩЕСТВУЮЩИЕ СУЩНОСТИ И СВЯЗИ В БАЗЕ:\n{entities_text}" if entities_text else "\n\n(База данных пуста или не содержит фактов)"

        # RAG поиск по документам
        rag_context = ""
        # 🔥 Расширенные триггеры — теперь RAG срабатывает на больше запросов
        rag_triggers = [
            "найди", "документ", "инструкц", "секрет", "код", "пароль",
            "что любит", "что знает", "расскажи", "что ты знаешь",
            "интересы", "хобби", "предпочт", "нравит"
        ]
        if self.knowledge_base and any(word in user_text.lower() for word in rag_triggers):
            search_results = self.knowledge_base.search(user_text, n_results=3)  # Было 2, стало 3
            if search_results:
                rag_context = "\n\nИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ:\n" + "\n".join(search_results)

        system_prompt = self._build_system_prompt(name_info, entities_context, rag_context)
        messages = self._build_messages(system_prompt, user_text, chat_history, frame)

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "top_p": OLLAMA_TOP_P,
                "num_predict": OLLAMA_NUM_PREDICT
            }
        }

        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            if r.status_code == 200:
                raw_answer = r.json().get("message", {}).get("content", "")
                print(f"  [LLM] Сырой: {raw_answer[:200]}")
                return self._parse_json_response(raw_answer)
            else:
                return {"action": "reply", "text": f"Ошибка API: {r.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"action": "reply", "text": "Ollama не запущена."}
        except requests.exceptions.Timeout:
            return {"action": "reply", "text": "Превышено время ожидания."}
        except Exception as e:
            return {"action": "reply", "text": f"Ошибка: {e}"}

    def _build_system_prompt(self, name_info, entities_context, rag_context):
        return f"""Ты — голосовой ассистент Кью. Ты общаешься тепло, как подруга. Отвечай от ЖЕНСКОГО лица.

Анализируй запрос пользователя и возвращай СТРОГО JSON:
{{"action": "...", "entity_name": "...", "entity_type": "...", "data_type": "...", "data_value": "...", "relationship": "...", "entity1_name": "...", "entity2_name": "...", "text": "...", "search_query": "...", "search_type": "general", "city": "...", "need_vision": false}}

ДЕЙСТВИЯ:
1. "create_entity" — запомнить человека/питомца (БЕЗ фото). Пример: "запомни жену зовут Катя"
2. "add_data" — добавить факт. Пример: "я люблю кофе"
3. "update_data" — исправить старый факт. Пример: "замени факт, я люблю чай"
4. "create_relationship" — связать двух людей.
5. "query_entity" — рассказать о ком-то или ответить на вопрос "кто я?", "что ты знаешь?".
   ВАЖНО: Если в базе знаний (RAG) есть факты о человеке — ОБЯЗАТЕЛЬНО положи их в поле "data_value" и "data_type": "fact".
6. "capture_face" — сделать ФОТО. ТОЛЬКО при явной просьбе: "запомни моё лицо", "сделай фото".
7. "web_search" — поиск в интернете с чтением сайтов.
   Используй, когда пользователь спрашивает о чём-то конкретном:
   - погода, новости, курсы валют
   - расписания, афиши, мероприятия
   - отключения воды, света, услуги ЖКХ
   - факты, которые ты не знаешь
   - запросы со словами "посмотри", "найди", "в интернете"
   
   В поле "search_query" укажи КОРОТКИЙ поисковый запрос (2-5 слов, суть вопроса).
   В поле "search_type" укажи: "general", "news" или "weather".
   В поле "city" укажи город ТОЛЬКО для погоды.
   
   Примеры search_query:
   - "отключения воды Воронеж" (НЕ "посмотри когда будут отключения холодной воды в воронеже")
   - "киноафиша Воронеж"
   - "мероприятия Воронеж июль"
   
   НЕ используй web_search для:
   - личной информации о пользователе (используй query_entity)
   - разговоров, приветствий
   - того, что ты уже знаешь из базы
   
   В поле "search_query" укажи ТОЧНЫЙ поисковый запрос на русском, без лишних слов.
   В поле "search_type" укажи: "general", "news" или "weather".
   В поле "city" укажи город ТОЛЬКО для погоды.
8. "reply" — обычный разговор.
9. "vision" — анализ изображения.
10. "exit" — выход.
11. "forget" — удалить сущность. ТОЛЬКО при словах: "забудь", "удали", "очисти память".

ПРАВИЛА ИСПОЛЬЗОВАНИЯ WEB_SEARCH:
- Если пользователь спрашивает про РЕЙТИНГ, ОЦЕНКУ, ОТЗЫВЫ фильма/книги/сериала → ВСЕГДА используй web_search
- Если пользователь говорит "этого", "этой", "этот" — смотри КОНТЕКСТ предыдущих вопросов
- Для фильмов добавляй в search_query слово "Кинопоиск" или "IMDb"
- Пример: "рейтинг фильма Доктор Ду" → search_query="фильм Доктор Ду рейтинг Кинопоиск"

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. РАЗЛИЧАЙ "МЕНЯ" и "ТЕБЯ":
   - "как МЕНЯ зовут", "что ты знаешь обо МНЕ" → это про ПОЛЬЗОВАТЕЛЯ. action="query_entity", entity_name=ИМЯ ИЗ name_info.
   - "как ТЕБЯ зовут" → это про ТЕБЯ. action="reply", text="Меня зовут Кью, я твоя подруга и помощница!"
2. ПРАВИЛА ЧЕЛОВЕЧНОЙ РЕЧИ (КРИТИЧНО ДЛЯ ОЗВУЧКИ):
   - НИКОГДА не отвечай как база данных: "Дима это пользователь, Катя жена". Это звучит как робот!
   - Говори естественно, как живая подруга. Используй вводные слова: "Ну,", "Кстати,", "Помнишь,".
   - Если действие "query_entity" — ОБЯЗАТЕЛЬНО заполни поле "text" готовой фразой для озвучки!
3. ЕСЛИ БАЗА ПУСТА → НЕ ВЫДУМЫВАЙ! Просто верни action="query_entity" без поля text.
4. ФОТО только по явной просьбе. "Запомни жену Катю" → create_entity. "Запомни моё лицо" → capture_face.
5. "Не надо запоминать" → НЕ forget! Ответь "Хорошо, не буду."
6. НИКОГДА не выдумывай пароли, коды, PIN-коды, секретные данные!
7. ВСЕГДА используй женский род. Возвращай ТОЛЬКО JSON.
8. ПРИ ПОИСКЕ В ИНТЕРНЕТЕ (web_search):
   - search_query должен быть КОРОТКИМ и ТОЧНЫМ (2-5 слов)
   - НЕ добавляй слова "посмотри", "найди", "информацию" — только суть запроса
   - Пример ПРАВИЛЬНОГО search_query: "отключения воды Воронеж"
   - Пример НЕПРАВИЛЬНОГО search_query: "посмотри в интернете когда будут отключения холодной воды в воронеже"

ПРИМЕРЫ:
- "что ты знаешь обо мне" → {{"action": "query_entity", "entity_name": "Дима", "text": "Ты — Дима, мой главный пользователь! Я помню, что мы с тобой уже много чего обсудили."}}
- "как тебя зовут" → {{"action": "reply", "text": "Меня зовут Кью, я твоя подруга и помощница!"}}
- "замени факт, Таисия любит какао" → {{"action": "update_data", "entity_name": "Таисия", "data_type": "fact", "data_value": "любит пить какао"}}
- "что ты знаешь о Кате" → {{"action": "query_entity", "entity_name": "Катя", "text": "Катя — твоя жена, ты о ней так тепло рассказывал!"}}
- "запомни жену зовут Катя" → {{"action": "create_entity", "entity_name": "Катя", "entity_type": "person", "relationship": "жена"}}
- "запомни моё лицо" → {{"action": "capture_face", "entity_name": "Дима"}}
- "не надо меня запоминать" → {{"action": "reply", "text": "Хорошо, не буду запоминать."}}
- "забудь всё про Катю" → {{"action": "forget", "entity_name": "Катя"}}
- "какая погода в Москве" → {{"action": "web_search", "search_type": "weather", "city": "Москва"}}
- "погода в Санкт-Петербурге" → {{"action": "web_search", "search_type": "weather", "city": "Санкт-Петербург"}}
- "какие новости сегодня" → {{"action": "web_search", "search_type": "news", "search_query": "главные новости сегодня"}}
- "новости технологий" → {{"action": "web_search", "search_type": "news", "search_query": "новости технологий"}}
- "кто президент Франции" → {{"action": "web_search", "search_type": "general", "search_query": "президент Франции 2026"}}
- "курс доллара" → {{"action": "web_search", "search_type": "general", "search_query": "курс доллара сегодня"}}

{name_info}
{entities_context}
{rag_context}"""

    def _build_messages(self, system_prompt, user_text, chat_history, frame):
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-10:]:
                messages.append(msg)
        if frame is not None:
            img = self.encode_image(frame)
            if img:
                messages.append({"role": "user", "content": user_text, "images": [img]})
                return messages
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _parse_json_response(raw_text):
        if not raw_text:
            return {"action": "reply", "text": "Не удалось получить ответ."}
        try:
            return json.loads(raw_text)
        except:
            pass
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        matches = re.findall(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if matches:
            results = []
            for m in matches:
                try: results.append(json.loads(m))
                except: pass
            if len(results) == 1: return results[0]
            elif len(results) > 1: return {"multi": results}
        raw_lower = raw_text.lower()
        if 'запомни' in raw_lower:
            return {"action": "add_data", "data_type": "fact", "data_value": raw_text}
        if 'пока' in raw_lower or 'до свидания' in raw_lower:
            return {"action": "exit", "text": "До свидания!"}
        if 'забудь' in raw_lower:
            return {"action": "forget", "text": "Забыла всё."}
        return {"action": "reply", "text": raw_text}