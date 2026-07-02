"""Синтез речи (Silero TTS)"""
import re
import time
import torch
import sounddevice as sd
from config import TTS_SPEAKER, TTS_SAMPLE_RATE, TTS_LANGUAGE, TTS_VERSION


class TextToSpeech:
    def __init__(self):
        self.model = None
        self.sample_rate = TTS_SAMPLE_RATE
        self.speaker = TTS_SPEAKER
        self._load_model()

    def _load_model(self):
        print("[TTS] Загрузка Silero v4...")
        try:
            self.device = torch.device('cpu')
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=TTS_LANGUAGE,
                speaker=TTS_VERSION
            )
            self.model.to(self.device)
            print("[TTS] Silero загружена.")
        except Exception as e:
            print(f"[TTS] Ошибка: {e}")
            self.model = None

    @staticmethod
    def clean_text(text):
        """Подготавливает текст для озвучки"""
        if not text:
            return ""
        text = re.sub(r'[\*\\"\'«»]', '', text)

        # Цифры в слова
        digit_map = {'0': 'ноль ', '1': 'один ', '2': 'два ', '3': 'три ',
                     '4': 'четыре ', '5': 'пять ', '6': 'шесть ', '7': 'семь ',
                     '8': 'восемь ', '9': 'девять '}
        for digit, word in digit_map.items():
            text = text.replace(digit, word)

        # Транслитерация
        translit_map = {
            'a': 'а', 'b': 'б', 'v': 'в', 'g': 'г', 'd': 'д', 'e': 'е', 'z': 'з', 'i': 'и',
            'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'r': 'р', 's': 'с',
            't': 'т', 'u': 'у', 'f': 'ф', 'h': 'х', 'c': 'ц', 'y': 'ы',
            'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'E': 'Е', 'Z': 'З', 'I': 'И',
            'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р', 'S': 'С',
            'T': 'Т', 'U': 'У', 'F': 'Ф', 'H': 'Х', 'C': 'Ц', 'Y': 'Ы'
        }

        def replace_latin(match):
            return ''.join(translit_map.get(c, c) for c in match.group(0))

        text = re.sub(r'\b[A-Za-z]+\b', replace_latin, text)
        text = re.sub(r'[^0-9А-Яа-яЁё\s.,!?;:-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def speak(self, text):
        """Озвучивает текст"""
        if not text or self.model is None:
            print(f"[SPEAK] {text}")
            return False

        text = self.clean_text(text)
        if not text:
            return False

        print(f"\n[SPEAK] {text}")
        try:
            audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
            sd.play(audio, self.sample_rate)
            sd.wait()
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"[SPEAK] Ошибка: {e}")
            return False