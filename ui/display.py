"""Отрисовка интерфейса (OpenCV + PIL)"""
import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from config import FONT_PATHS, EMOTION_CONFIG


class Display:
    def __init__(self):
        self.font_large = self._load_font(28)
        self.font_status = self._load_font(18)
        self.font_small = self._load_font(16)
        self.font_tiny = self._load_font(14)

    @staticmethod
    def _load_font(size):
        for font_path in FONT_PATHS:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def draw_text_on_frame(frame, text, position, font, bg_color=(0, 0, 0),
                           text_color=(255, 255, 255), padding=6, alpha=0.75):
        """Рисует текст с подложкой через PIL (поддерживает кириллицу)"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x, y = position

            # 🔥 Рисуем подложку только если bg_color не None
            if bg_color is not None:
                overlay = img_pil.copy()
                draw_overlay = ImageDraw.Draw(overlay)
                draw_overlay.rectangle(
                    [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                    fill=bg_color
                )
                img_pil = Image.blend(img_pil, overlay, alpha=alpha)

            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), text, font=font, fill=text_color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return frame

    def render_frame(self, frame, voice_locked, detected_name, detected_emotion, previous_emotion):
        """Рендерит полный кадр с UI"""
        h, w = frame.shape[:2]

        # СТАТУС В ПРАВОМ ВЕРХНЕМ УГЛУ (без подложки)
        status_x = w - 220
        status_y = 10

        status_text = "СЛУШАЮ..." if not voice_locked else "ГОВОРЮ..."
        status_color = (0, 255, 0) if not voice_locked else (0, 0, 255)
        # 🔥 Передаём bg_color=None — подложки не будет
        frame = self.draw_text_on_frame(
            frame, status_text, (status_x, status_y),
            self.font_status, text_color=status_color, bg_color=None, padding=0
        )

        # ИМЯ + ЭМОЦИЯ (под статусом, тоже без подложки)
        info_y = status_y + 24

        if detected_name != "Неизвестный":
            info_text = detected_name

            if detected_emotion and detected_emotion != "neutral":
                emo_cfg = EMOTION_CONFIG.get(detected_emotion, EMOTION_CONFIG['neutral'])
                emo_ru = emo_cfg['ru']
                info_text += f" | {emo_ru}"
        else:
            info_text = "Неизвестный"

        # 🔥 Передаём bg_color=None — подложки не будет
        frame = self.draw_text_on_frame(
            frame, info_text, (status_x, info_y),
            self.font_small, bg_color=None, text_color=(255, 255, 255), padding=0
        )

        return frame

    def render_confidence_panel(self, frame, detected_name, confidence, distance):
        """Рисует панель уверенности в правом нижнем углу"""
        h, w = frame.shape[:2]

        panel_w = 280
        panel_h = 75
        panel_x = w - panel_w - 10
        panel_y = h - panel_h - 10

        if confidence >= 75:
            bar_color = (0, 220, 0)
            status_text = "ВЫСОКАЯ"
        elif confidence >= 50:
            bar_color = (0, 220, 220)
            status_text = "СРЕДНЯЯ"
        elif confidence >= 25:
            bar_color = (0, 140, 255)
            status_text = "НИЗКАЯ"
        else:
            bar_color = (0, 0, 220)
            status_text = "НЕ ОПРЕДЕЛЕНО"

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        frame = self.draw_text_on_frame(
            frame, "РАСПОЗНАВАНИЕ",
            (panel_x + 8, panel_y + 5),
            self.font_tiny, bg_color=None, text_color=(200, 200, 200), padding=0
        )

        name_text = detected_name if detected_name != "Неизвестный" else "—"
        frame = self.draw_text_on_frame(
            frame, f"Лицо: {name_text}",
            (panel_x + 8, panel_y + 22),
            self.font_tiny, bg_color=None, text_color=(255, 255, 255), padding=0
        )

        bar_x = panel_x + 8
        bar_y = panel_y + 40
        bar_w = panel_w - 100
        bar_h = 10

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        fill_w = int(bar_w * (confidence / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        conf_text = f"{confidence:.0f}%"
        frame = self.draw_text_on_frame(
            frame, conf_text,
            (bar_x + bar_w + 5, bar_y - 2),
            self.font_tiny, bg_color=None, text_color=bar_color, padding=0
        )

        dist_text = f"dist: {distance:.3f} | {status_text}"
        frame = self.draw_text_on_frame(
            frame, dist_text,
            (panel_x + 8, panel_y + 55),
            self.font_tiny, bg_color=None, text_color=(150, 150, 150), padding=0
        )

        return frame

    def render_formula_overlay(self, frame, detected_name, distance, confidence,
                               current_vector, matched_vector, matched_name):
        """Рисует оверлей с формулами расчёта распознавания"""
        h, w = frame.shape[:2]

        # 🔥 ЕЩЁ МЕНЬШЕ размеры панели
        panel_w = min(480, w - 40)
        panel_h = min(320, h - 40)
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2

        # 🔥 Ещё более прозрачный фон (60% панель, 40% камера)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # Очень тонкая рамка
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 255), 1)

        # Заголовок (мельче)
        title = "АЛГОРИТМ РАСПОЗНАВАНИЯ ЛИЦ (FaceNet)"
        frame = self.draw_text_on_frame(frame, title, (panel_x + 8, panel_y + 8),
                                        self.font_tiny, text_color=(100, 200, 255), padding=2)

        y = panel_y + 28
        line_h = 16  # 🔥 Ещё меньше высота строки

        # --- ЭТАП 1 ---
        frame = self.draw_text_on_frame(frame, "1. Embedding: f(лицо) → R^128", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 200, 100), padding=2)
        y += line_h

        # --- ЭТАП 2 ---
        frame = self.draw_text_on_frame(frame, "2. Нормализация: v = v/||v||", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 200, 100), padding=2)
        y += line_h

        # --- ЭТАП 3 ---
        frame = self.draw_text_on_frame(frame, "3. Косинусное расстояние: d = 1 - (v1·v2)", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 200, 100), padding=2)
        y += line_h

        # Показываем только 3 компоненты векторов
        if current_vector is not None and matched_vector is not None:
            preview1 = "[" + ", ".join(f"{x:.3f}" for x in current_vector[:3]) + ", ...]"
            frame = self.draw_text_on_frame(frame, f"  Текущий: {preview1}", (panel_x + 8, y),
                                            self.font_tiny, text_color=(100, 255, 100), padding=2)
            y += line_h

            preview2 = "[" + ", ".join(f"{x:.3f}" for x in matched_vector[:3]) + ", ...]"
            frame = self.draw_text_on_frame(frame, f"  '{matched_name}': {preview2}", (panel_x + 8, y),
                                            self.font_tiny, text_color=(255, 150, 100), padding=2)
            y += line_h

        # Текущее расстояние
        frame = self.draw_text_on_frame(frame, f"  → d = {distance:.4f}", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 255, 100), padding=2)
        y += line_h + 2

        # --- ЭТАП 4 ---
        frame = self.draw_text_on_frame(frame, "4. Уверенность: conf = (0.60 - d)/0.60 × 100%", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 200, 100), padding=2)
        y += line_h

        # Подставляем текущие значения
        calc_detail = f"  conf = (0.60 - {distance:.4f})/0.60 × 100% = {confidence:.1f}%"
        frame = self.draw_text_on_frame(frame, calc_detail, (panel_x + 8, y),
                                        self.font_tiny, text_color=(100, 255, 100), padding=2)
        y += line_h + 2

        # --- ПОРОГ ---
        frame = self.draw_text_on_frame(frame, "Порог: d < 0.60 → совпадение", (panel_x + 8, y),
                                        self.font_tiny, text_color=(255, 200, 100), padding=2)
        y += line_h

        threshold_color = (0, 255, 0) if distance < 0.6 else (0, 0, 255)
        if distance < 0.6:
            verdict = f"  d={distance:.4f} < 0.60 → ЭТО {detected_name.upper()}!"
        else:
            verdict = f"  d={distance:.4f} ≥ 0.60 → НЕИЗВЕСТНЫЙ"

        frame = self.draw_text_on_frame(frame, verdict, (panel_x + 8, y),
                                        self.font_small, text_color=threshold_color, padding=2)
        y += line_h + 4

        # Подсказка
        frame = self.draw_text_on_frame(frame, "[F] закрыть  [H] помощь", (panel_x + 8, panel_y + panel_h - 18),
                                        self.font_tiny, text_color=(150, 150, 150), padding=2)

        return frame

    def render_help_overlay(self, frame):
        """Рисует оверлей с горячими клавишами"""
        h, w = frame.shape[:2]
        panel_w = 280  # 🔥 Уменьшили
        panel_h = 180  # 🔥 Уменьшили
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2

        # 🔥 Более прозрачный фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 255, 100), 1)

        frame = self.draw_text_on_frame(frame, "ГОРЯЧИЕ КЛАВИШИ", (panel_x + 10, panel_y + 10),
                                        self.font_tiny, text_color=(100, 255, 100), padding=2)

        keys = [
            ("F", "Показать/скрыть формулы"),
            ("H", "Показать/скрыть помощь"),
            ("Q", "Выход из программы"),
        ]

        y = panel_y + 35
        for key, desc in keys:
            frame = self.draw_text_on_frame(frame, f"[{key}]", (panel_x + 15, y),
                                            self.font_tiny, text_color=(255, 200, 100), padding=2)
            frame = self.draw_text_on_frame(frame, desc, (panel_x + 50, y),
                                            self.font_tiny, text_color=(220, 220, 220), padding=2)
            y += 22

        frame = self.draw_text_on_frame(frame, "[H] закрыть", (panel_x + 10, panel_y + panel_h - 20),
                                        self.font_tiny, text_color=(150, 150, 150), padding=2)
        return frame