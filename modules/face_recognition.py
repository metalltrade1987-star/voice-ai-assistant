"""Распознавание лиц и эмоций"""
import os
import numpy as np
import cv2
from deepface import DeepFace
from config import KNOWN_FACES_DIR, FACE_MODEL_NAME, FACE_DETECTOR_BACKEND, FACE_MATCH_THRESHOLD


class FaceRecognizer:
    def __init__(self):
        self.known_faces_db = {}
        self.load_known_faces()
        # Для отображения формул
        self.last_current_vector = None
        self.last_matched_vector = None
        self.last_matched_name = None

    def load_known_faces(self):
        """Загружает базу лиц из папки"""
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        print("[FACE] Загрузка базы лиц...")
        self.known_faces_db.clear()

        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(KNOWN_FACES_DIR, filename)
                name = os.path.splitext(filename)[0]
                try:
                    emb = DeepFace.represent(
                        img_path=path,
                        model_name=FACE_MODEL_NAME,
                        detector_backend=FACE_DETECTOR_BACKEND,
                        enforce_detection=True
                    )
                    if emb and len(emb) > 0:
                        vec = np.array(emb[0]['embedding'])
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        self.known_faces_db[name] = vec
                        print(f"  [FACE] {name}")
                except Exception as e:
                    print(f"  [FACE] Ошибка {filename}: {e}")

        print(f"[FACE] В базе: {len(self.known_faces_db)} человек")

    def identify_face(self, emb):
        """Идентифицирует лицо через КОСИНУСНОЕ расстояние"""
        if not self.known_faces_db:
            return "Неизвестный", 999.0, 0.0

        # Нормализация вектора
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best, min_dist = "Неизвестный", 999.0
        best_vec = None

        for name, db_vec in self.known_faces_db.items():
            # 🔥 КОСИНУСНОЕ РАССТОЯНИЕ
            # cosine_distance = 1 - cosine_similarity
            cosine_sim = np.dot(emb, db_vec)
            cosine_dist = 1 - cosine_sim

            if cosine_dist < min_dist:
                min_dist, best, best_vec = cosine_dist, name, db_vec

        # Сохраняем для отображения формул
        self.last_current_vector = emb.copy()
        self.last_matched_vector = best_vec.copy() if best_vec is not None else None
        self.last_matched_name = best if best != "Неизвестный" else None

        # 🔥 НОВАЯ ФОРМУЛА УВЕРЕННОСТИ (как у коллег)
        # confidence = (0.60 - distance) / 0.60 × 100%
        confidence = ((0.60 - min_dist) / 0.60) * 100.0
        confidence = max(0.0, confidence)

        # Порог 0.6
        if min_dist < 0.6:
            return best, min_dist, confidence
        else:
            return "Неизвестный", min_dist, confidence

    def add_known_face(self, name, frame):
        """Сохраняет фото с транслитерацией имени"""
        import os
        from config import KNOWN_FACES_DIR, FACE_MODEL_NAME, FACE_DETECTOR_BACKEND

        # 🔥 Транслитерация кириллицы в латиницу
        translit = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }

        safe_name = ''.join(translit.get(c, c) for c in name)

        filename = f"{safe_name}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)

        if os.path.exists(filepath):
            print(f"[FACE] ⚠️ Файл {filepath} уже существует, перезаписываем...")

        cv2.imwrite(filepath, frame)
        print(f"[FACE] Сохранено фото: {filepath}")

        # Извлекаем эмбеддинг
        try:
            emb = DeepFace.represent(
                img_path=filepath,
                model_name=FACE_MODEL_NAME,
                detector_backend=FACE_DETECTOR_BACKEND,
                enforce_detection=True
            )
            if emb and len(emb) > 0:
                vec = np.array(emb[0]['embedding'])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                is_new = name not in self.known_faces_db
                self.known_faces_db[name] = vec

                if is_new:
                    print(f"[FACE] ✅ ДОБАВЛЕН: {name}")
                else:
                    print(f"[FACE] ✅ ОБНОВЛЁН: {name}")

                return True
        except Exception as e:
            print(f"[FACE Error] add_known_face: {e}")

        return False

    def analyze_frame(self, frame):
        """Анализирует кадр: находит лица, эмоции, имена"""
        results = []
        try:
            analysis = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend=FACE_DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
            if isinstance(analysis, dict):
                analysis = [analysis]

            for res in analysis:
                r = res.get('region', {})
                x, y, w, h = r.get('x', 0), r.get('y', 0), r.get('w', 0), r.get('h', 0)
                face = frame[max(0, y):y + h, max(0, x):x + w]
                name = "Неизвестный"
                distance = 999.0
                confidence = 0.0

                if face.size > 0:
                    try:
                        emb = DeepFace.represent(
                            img_path=face,
                            model_name=FACE_MODEL_NAME,
                            detector_backend=FACE_DETECTOR_BACKEND,
                            enforce_detection=False
                        )
                        if emb and len(emb) > 0:
                            name, distance, confidence = self.identify_face(np.array(emb[0]['embedding']))
                    except:
                        name, distance, confidence = "Неизвестный", 999.0, 0.0
                else:
                    name, distance, confidence = "Неизвестный", 999.0, 0.0

                em = res.get('emotion', {})
                dom = max(em, key=em.get) if em else 'neutral'
                if em.get(dom, 0) < 40:
                    dom = "neutral"

                results.append({
                    "dominant": dom,
                    "region": r,
                    "name": name,
                    "distance": distance,
                    "confidence": confidence
                })
        except Exception as e:
            print(f"[FACE] Ошибка: {e}")
        return results