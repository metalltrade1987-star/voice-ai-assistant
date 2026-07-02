"""Конфигурация проекта"""
import os

# Пути
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_PATH = r"D:\Lerning\PythonProject_web_voice\vosk-model-ru-0.22"
KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, "known_faces")
KNOWLEDGE_DOCS_DIR = os.path.join(PROJECT_ROOT, "knowledge_docs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "assistant_memory.db")
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")  # ← Добавлено!

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_TIMEOUT = 180
OLLAMA_TEMPERATURE = 0.7  # Было 0.3. Подняли для более живых ответов
OLLAMA_TOP_P = 0.95       # Было 0.9. Чуть расширили выбор слов
OLLAMA_NUM_PREDICT = 300  # Было 200, стало 300

# TTS (Silero)
TTS_SPEAKER = "kseniya"
TTS_SAMPLE_RATE = 48000
TTS_LANGUAGE = "ru"
TTS_VERSION = "v4_ru"

# STT (Vosk)
STT_SAMPLE_RATE = 16000
STT_BUFFER_SIZE = 4096

# Face recognition
FACE_MODEL_NAME = "Facenet"
FACE_DETECTOR_BACKEND = "opencv"
FACE_MATCH_THRESHOLD = 0.6
FACE_ANALYSIS_EVERY_N_FRAMES = 15

# Vision
VISION_CROP_RATIO = 0.7  # Верхние 70% кадра
VISION_IMAGE_SIZE = (768, 768)
VISION_IMAGE_QUALITY = 85

# Chat history
MAX_CHAT_HISTORY = 10

# Emotion config
EMOTION_CONFIG = {
    'angry': {'ru': 'Злость', 'color': (0, 0, 255)},
    'disgust': {'ru': 'Отвращение', 'color': (128, 0, 128)},
    'fear': {'ru': 'Страх', 'color': (0, 165, 255)},
    'happy': {'ru': 'Радость', 'color': (0, 255, 0)},
    'sad': {'ru': 'Грусть', 'color': (255, 0, 0)},
    'surprise': {'ru': 'Удивление', 'color': (255, 255, 0)},
    'neutral': {'ru': 'Нейтрально', 'color': (200, 200, 200)}
}

# Fonts
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf"
]