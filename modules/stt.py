"""Распознавание речи (Vosk)"""
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from config import VOSK_MODEL_PATH, STT_SAMPLE_RATE, STT_BUFFER_SIZE


class SpeechToText:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.mic = None
        self.stream = None
        self._init_model()

    def _init_model(self):
        print("[STT] Инициализация Vosk...")
        import os
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"[STT] Модель не найдена: {VOSK_MODEL_PATH}")
            return
        try:
            self.model = Model(VOSK_MODEL_PATH)
            self.recognizer = KaldiRecognizer(self.model, STT_SAMPLE_RATE)
            self.mic = pyaudio.PyAudio()
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=STT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=STT_BUFFER_SIZE
            )
            self.stream.start_stream()
            print("[STT] Vosk активен.")
        except Exception as e:
            print(f"[STT] Ошибка: {e}")

    def read_audio(self):
        """Читает порцию аудио с микрофона"""
        if self.stream is None:
            return None
        try:
            return self.stream.read(STT_BUFFER_SIZE, exception_on_overflow=False)
        except:
            return None

    def process_audio(self, data):
        """Обрабатывает аудио и возвращает распознанный текст (или None)"""
        if self.recognizer is None or data is None:
            return None
        if self.recognizer.AcceptWaveform(data):
            try:
                res = json.loads(self.recognizer.Result())
                text = res.get("text", "").strip().lower()
                return text if text else None
            except:
                return None
        return None

    def cleanup(self):
        """Освобождает ресурсы"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()