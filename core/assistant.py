"""Главный класс-оркестр, связывающий все модули"""
import cv2
import threading
import time
import re
from config import EMOTION_CONFIG, FACE_ANALYSIS_EVERY_N_FRAMES
from core.database import DatabaseManager
from core.entity_manager import EntityManager
from core.knowledge_base import KnowledgeBase
from core.llm_router import LLMRouter
from modules.tts import TextToSpeech
from modules.stt import SpeechToText
from modules.face_recognition import FaceRecognizer
from ui.display import Display


class Assistant:
    def __init__(self):
        print("[START] Инициализация системы...")

        self.last_distance = 999.0
        self.last_confidence = 0.0
        self.show_formulas = False
        self.show_help = False

        # Состояние
        self.running = True
        self.latest_data = []
        self.voice_locked = False
        self.last_detected_name = "Неизвестный"
        self.last_known_name = "Неизвестный"
        self.last_detected_emotion = "neutral"
        self.previous_emotion = None
        self.current_frame = None
        self.chat_history = []

        # Модули
        print("[START] Инициализация баз данных и модулей...")
        self.db = DatabaseManager()
        self.entity_manager = EntityManager(self.db)
        self.kb = KnowledgeBase()
        self.kb.load_documents_from_folder()
        self.llm = LLMRouter(self.kb)
        self.tts = TextToSpeech()
        self.stt = SpeechToText()
        self.faces = FaceRecognizer()
        from modules.web_search import WebSearcher
        self.web = WebSearcher()
        self.display = Display()

        # Поток слушания
        self.voice_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.voice_thread.start()

        print("[START] Система готова!")

    def _get_user_name(self):
        """Возвращает актуальное имя пользователя (нормализованное)"""
        name = self.last_known_name if self.last_known_name != "Неизвестный" else self.last_detected_name
        # Нормализуем имя
        return DatabaseManager.normalize_name(name)

    def _get_entities_context(self):
        """Собирает текст о существующих сущностях для промпта LLM"""
        try:
            # Получаем все сущности
            self.db.cursor.execute("SELECT id, name, type FROM entities")
            entities = self.db.cursor.fetchall()
            if not entities:
                return ""

            context = ""
            for ent_id, name, ent_type in entities:
                # Данные
                self.db.cursor.execute("SELECT data_type, data_value FROM entity_data WHERE entity_id = ?", (ent_id,))
                data = self.db.cursor.fetchall()
                data_str = ", ".join([f"{d[0]}: {d[1]}" for d in data]) if data else "нет данных"

                # Связи
                self.db.cursor.execute("""
                    SELECT e.name, r.relationship_type FROM relationships r
                    JOIN entities e ON (r.entity2_id = e.id AND r.entity1_id = ?)
                        OR (r.entity1_id = e.id AND r.entity2_id = ?)
                """, (ent_id, ent_id))
                rels = self.db.cursor.fetchall()
                rel_str = ", ".join([f"{r[0]} ({r[1]})" for r in rels]) if rels else "нет связей"

                context += f"- {name} ({ent_type}): данные=[{data_str}], связи=[{rel_str}]\n"
            return context
        except Exception as e:
            print(f"[DB Error] get_entities_context: {e}")
            return ""

    def _listen_loop(self):
        """Поток слушания микрофона"""
        print("[LISTEN] Слушатель запущен.")
        while self.running:
            if self.stt.stream is None:
                time.sleep(1)
                continue
            if self.voice_locked:
                time.sleep(0.02)
                continue

            data = self.stt.read_audio()
            if data is None:
                time.sleep(0.1)
                continue

            cmd = self.stt.process_audio(data)
            if cmd:
                print(f"\n[LISTEN] Распознано: '{cmd}'")
                try:
                    self.process_command(cmd)
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()

            time.sleep(0.02)

    def speak(self, text):
        """Озвучивает текст с блокировкой микрофона"""
        self.voice_locked = True
        try:
            self.tts.speak(text)
        finally:
            self.voice_locked = False

    def process_command(self, cmd):
        """Обрабатывает распознанную команду"""
        print(f"\n[CMD] Обработка: '{cmd}'")

        user_name = self._get_user_name()
        entities_text = self._get_entities_context()

        command = self.llm.ask(
            user_text=cmd,
            user_name=user_name,
            entities_text=entities_text,
            chat_history=self.chat_history,
            frame=self.current_frame
        )

        # Несколько команд
        if "multi" in command:
            for c in command["multi"]:
                self._execute_command(c, cmd)
            return

        self._execute_command(command, cmd)

    def _execute_command(self, command, original_cmd=None):
        """Выполняет одну команду от LLM"""
        action = command.get("action", "reply")
        entity_name = command.get("entity_name")
        entity_type = command.get("entity_type", "person")
        data_type = command.get("data_type")
        data_value = command.get("data_value", "")
        relationship = command.get("relationship")
        entity1_name = command.get("entity1_name")
        entity2_name = command.get("entity2_name")
        text = command.get("text", "")
        need_vision = command.get("need_vision", False)

        user_name = self._get_user_name()
        print(f"  [CMD] Action: {action}, Entity: {entity_name}")

        # 1. CREATE ENTITY
        if action == "create_entity":
            if entity_name:
                photo = self.current_frame if need_vision else None
                self.entity_manager.create_or_update_entity(
                    name=entity_name,
                    entity_type=entity_type,
                    photo=photo
                )
                if relationship and user_name != "Неизвестный":
                    self.entity_manager.create_relationship_safe(user_name, entity_name, relationship)

                self.speak(text if text else f"Запомнила: {entity_name}.")
            else:
                self.speak("Не поняла, кого запомнить.")
            return

        # 2. ADD DATA (новый факт)
        if action == "add_data":
            if entity_name and data_type:
                if data_type == "voice":
                    self.speak(f"Записываю голос {entity_name}... Говори 5 секунд.")
                    self.entity_manager.add_fact(entity_name, "Голос записан", category="voice")
                elif data_type == "photo":
                    photo = self.current_frame
                    if photo is not None:
                        ent_id = self.db.get_entity_by_name(entity_name)
                        if ent_id:
                            self.entity_manager.save_photo(ent_id, entity_name, photo)
                            self.speak(f"Фото {entity_name} сохранено.")
                    else:
                        self.speak("Сейчас нет кадра для фото.")
                else:
                    self.entity_manager.add_fact(entity_name, data_value)
                    self.speak(text if text else f"Запомнила: {data_value}.")
            else:
                self.speak("Не поняла, что запомнить.")
            return

        # 3. UPDATE DATA (замена старого факта)
        if action == "update_data":
            if entity_name and data_value:
                self.entity_manager.update_fact(entity_name, data_value)
                self.speak(text if text else f"Обновила: {data_value}.")
            else:
                self.speak("Не поняла, что обновить.")
            return

        # 4. CREATE RELATIONSHIP
        if action == "create_relationship":
            e1 = entity1_name or user_name
            e2 = entity2_name or entity_name
            if e1 and e2 and relationship:
                self.entity_manager.create_relationship_safe(e1, e2, relationship)
                self.speak(text if text else f"Запомнила связь: {e1} и {e2} — {relationship}.")
            else:
                self.speak("Не поняла, кого связать.")
            return

        # 5. QUERY ENTITY (Теперь отвечает ТОЛЬКО LLM, код не вмешивается)
        if action == "query_entity":
            if entity_name:
                # Если LLM уже сгенерировала живой ответ в поле text — просто озвучиваем!
                if text:
                    # 🔥 Защита от галлюцинаций: проверяем, не путает ли LLM имена
                    if user_name != "Неизвестный" and entity_name.lower() == user_name.lower():
                        # Если спрашивают про пользователя, но LLM говорит "тебя зовут [другое имя]"
                        if "тебя зовут" in text.lower():
                            # Проверяем, есть ли в тексте правильное имя пользователя
                            if user_name.lower() not in text.lower():
                                print(f"  [CMD] ⚠️ LLM перепутала имена, исправляю...")
                                text = f"Ты — {user_name}, мой главный пользователь!"

                    self.speak(text)
                else:
                    # Если LLM поленилась и не заполнила text (бывает с маленькими моделями)
                    # Делаем второй, быстрый запрос с упором на генерацию текста
                    print(f"  [CMD] LLM не заполнила text, запрашиваем генерацию ответа...")

                    # Собираем сырые данные для LLM
                    info = self.entity_manager.get_entity_full_info(entity_name)
                    raw_data = ""
                    if info:
                        facts = [d[1] for d in info["data"] if d[0] == "fact"]
                        rels = [f"{r[0]} ({r[1]})" for r in info["relationships"]]
                        raw_data = f"Факты из памяти: {', '.join(facts) if facts else 'нет'}. Связи: {', '.join(rels) if rels else 'нет'}."

                    # 🔥 Усиленный промпт: запрещаем выдумывать
                    follow_up_prompt = f"""Ты знаешь про {entity_name} следующее: {raw_data}.

    Скажи об этом по-человечески, одной фразой, как подруга. Не перечисляй факты через запятую.

    ВАЖНО:
    - Используй ТОЛЬКО факты из списка выше
    - НЕ выдумывай имена, связи или факты, которых нет в списке
    - Если фактов нет — скажи "Я пока мало знаю про {entity_name}"
    - Если спрашивают про пользователя (его зовут {user_name}) — используй его имя"""

                    follow_up = self.llm.ask(
                        user_text=follow_up_prompt,
                        user_name=user_name,
                        entities_text="",  # Не даем лишний контекст, чтобы не путать
                        chat_history=self.chat_history
                    )

                    # 🔥 Защита от ошибок типа
                    if isinstance(follow_up, dict):
                        final_text = follow_up.get("text", f"Я пока мало знаю про {entity_name}.")
                    elif isinstance(follow_up, str):
                        final_text = follow_up
                    else:
                        final_text = f"Я пока мало знаю про {entity_name}."

                    self.speak(final_text)
            return

        # 6. FORGET
        if action == "forget":
            if entity_name:
                if self.entity_manager.delete_entity(entity_name):
                    self.speak(text if text else f"Забыла всё про {entity_name}.")
                else:
                    self.speak(f"Я не знаю никого по имени {entity_name}.")
            else:
                self.speak("Кого именно забыть?")
            return

        # 7. EXIT
        if action == "exit":
            self.speak(text if text else "До свидания!")
            self.running = False
            return
        # 7.5. WEB SEARCH (поиск в интернете с Crawl4AI)
        if action == "web_search":
            search_type = command.get("search_type", "general")
            search_query = command.get("search_query", "")
            city = command.get("city", "Москва")

            print(f"  [WEB] Тип: {search_type}, запрос: {search_query}, город: {city}")

            # Выполняем поиск
            if search_type == "weather":
                self.speak(f"Сейчас посмотрю погоду в городе {city}...")
                result, sources = self.web.get_weather(city)
            elif search_type == "news":
                query = search_query or "главные новости сегодня"
                self.speak(f"Ищу новости по запросу: {query}...")
                result, sources = self.web.get_news(query, max_results=5)
            else:  # general
                if not search_query:
                    search_query = original_cmd or "вопрос"
                self.speak(f"Ищу в интернете и читаю сайты: {search_query}...")
                result, sources = self.web.deep_search(search_query, max_sites=2)

            # Передаём результаты LLM
            print(f"  [WEB] Результат получен ({len(result)} символов), передаю LLM...")

            follow_up_prompt = f"""Пользователь спросил: "{original_cmd}"

    Я нашла в интернете и прочитала следующие материалы:

    {result[:15000]}

    Сформулируй ПОДРОБНЫЙ и ПОЛЕЗНЫЙ ответ на основе ПРОЧИТАННОГО контента.
    Выбери самое важное и расскажи кратко (3-5 предложений).
    Если в материалах есть конкретные данные — обязательно упомяни их.

    ВАЖНО ДЛЯ ОЗВУЧКИ:
    - Числа пиши СЛОВАМИ: "двадцать четыре градуса", а НЕ "24°C"
    - НЕ используй символы °C, %, м/с — пиши словами
    - Говори как подруга, от женского лица

    Верни СТРОГО JSON: {{"action": "reply", "text": "твой ответ"}}"""

            follow_up = self.llm.ask(
                user_text=follow_up_prompt,
                user_name=user_name,
                entities_text="",
                chat_history=self.chat_history
            )

            # Защита от ошибок типа
            if isinstance(follow_up, dict):
                final_text = follow_up.get("text", "Не удалось сформулировать ответ.")
            elif isinstance(follow_up, str):
                final_text = follow_up
            else:
                final_text = "Не удалось сформулировать ответ."

            self.speak(final_text)
            return

            follow_up = self.llm.ask(
                user_text=follow_up_prompt,
                user_name=user_name,
                entities_text="",
                chat_history=self.chat_history
            )

            # Защита от ошибок типа
            if isinstance(follow_up, dict):
                final_text = follow_up.get("text", result)
            elif isinstance(follow_up, str):
                final_text = follow_up
            else:
                final_text = result

            self.speak(final_text)
            return

        # 8. VISION
        if action == "vision" or need_vision:
            if self.current_frame is not None and (not text or not need_vision):
                command = self.llm.ask(
                    user_text=original_cmd or text,
                    user_name=user_name,
                    entities_text=self._get_entities_context(),
                    chat_history=self.chat_history,
                    frame=self.current_frame
                )
                text = command.get("text", "")
            self.speak(text if text else "Не могу разглядеть.")
            return

        # 8.5. CAPTURE FACE (Запомнить лицо)
        if action == "capture_face":
            # Определяем, чьё лицо запомнить
            target_name = entity_name or user_name

            if target_name == "Неизвестный" or not target_name:
                # Если имя не определено — спрашиваем
                self.speak("Как тебя зовут? Назови своё имя, и я запомню твоё лицо.")
                return

            if self.current_frame is not None:
                self.speak(f"Секунду, запоминаю лицо {target_name}...")
                time.sleep(0.5)

                # Проверяем, есть ли уже такое лицо в базе
                if target_name in self.faces.known_faces_db:
                    self.speak(f"Обновляю фото для {target_name}...")

                success = self.faces.add_known_face(target_name, self.current_frame)

                if success:
                    # Если это новый человек — создаём сущность в БД
                    entity_id = self.db.get_entity_by_name(target_name)
                    if not entity_id:
                        self.entity_manager.create_or_update_entity(
                            name=target_name,
                            entity_type="person",
                            photo=self.current_frame
                        )
                        self.speak(f"Готово! Я запомнила {target_name}. Теперь я буду узнавать тебя.")
                    else:
                        self.speak(f"Обновила фото для {target_name}. Теперь я буду лучше тебя узнавать!")

                    # Если есть связь (жена, дочь и т.д.) — сохраняем
                    if relationship and user_name != "Неизвестный":
                        self.entity_manager.create_relationship_safe(user_name, target_name, relationship)
                        self.speak(f"И запомнила, что {target_name} — {relationship} {user_name}.")
                else:
                    self.speak("Не удалось распознать лицо на фото. Убедись, что лицо хорошо видно и смотрит в камеру.")
            else:
                self.speak("Сейчас нет кадра с камеры. Повернись к камере и попробуй снова.")
            return

        # 9. REPLY (по умолчанию)
        self.speak(text if text else "Не поняла.")

    def run(self):
        """Основной цикл работы с камерой"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Камера недоступна")
            return

        print("[RUN] Основной цикл запущен")
        fc = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            fc += 1

            # Анализ лица каждые N кадров
            if fc % FACE_ANALYSIS_EVERY_N_FRAMES == 0:
                self.latest_data = self.faces.analyze_frame(frame.copy())

            # Рисуем рамки вокруг лиц
            for d in self.latest_data:
                r = d.get('region', {})
                if r:
                    cfg = EMOTION_CONFIG.get(d['dominant'], EMOTION_CONFIG['neutral'])
                    x, y, w, h = r.get('x', 0), r.get('y', 0), r.get('w', 0), r.get('h', 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['color'], 2)

            # Обновляем состояние
            if self.latest_data:
                self.last_detected_name = self.latest_data[0].get('name', 'Неизвестный')
                self.last_detected_emotion = self.latest_data[0].get('dominant', 'neutral')
                self.last_distance = self.latest_data[0].get('distance', 999.0)
                self.last_confidence = self.latest_data[0].get('confidence', 0.0)

                if self.last_detected_name != "Неизвестный":
                    self.last_known_name = self.last_detected_name

            # Рендер UI (статус-бар, имя, эмоция)
            frame = self.display.render_frame(
                frame, self.voice_locked,
                self.last_detected_name, self.last_detected_emotion,
                self.previous_emotion
            )
            if self.last_detected_emotion != self.previous_emotion:
                self.previous_emotion = self.last_detected_emotion

            # Рисуем панель уверенности в правом нижнем углу
            frame = self.display.render_confidence_panel(
                frame,
                self.last_detected_name,
                self.last_confidence,
                self.last_distance
            )

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('й'):
                break
            elif key == ord('f') or key == ord('F') or key == ord('а') or key == ord('А'):
                self.show_formulas = not self.show_formulas
                self.show_help = False
            elif key == ord('h') or key == ord('H') or key == ord('р') or key == ord('Р'):
                self.show_help = not self.show_help
                self.show_formulas = False

            # Рисуем оверлеи поверх всего
            if self.show_help:
                frame = self.display.render_help_overlay(frame)
            elif self.show_formulas:
                frame = self.display.render_formula_overlay(
                    frame,
                    self.last_detected_name,
                    self.last_distance,
                    self.last_confidence,
                    self.faces.last_current_vector,
                    self.faces.last_matched_vector,
                    self.faces.last_matched_name
                )

            cv2.imshow("AI Vision", frame)

        print("[RUN] Завершение...")
        cap.release()
        cv2.destroyAllWindows()
        self.stt.cleanup()
        self.db.close()
        print("[RUN] Работа завершена")