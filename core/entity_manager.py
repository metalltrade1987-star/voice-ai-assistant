"""Управление сущностями и их данными"""
import os
import cv2
import numpy as np
from config import MEDIA_DIR


class EntityManager:
    def __init__(self, db):
        self.db = db
        self.media_dir = MEDIA_DIR
        os.makedirs(os.path.join(self.media_dir, "photos"), exist_ok=True)
        os.makedirs(os.path.join(self.media_dir, "voices"), exist_ok=True)

    def create_or_update_entity(self, name, entity_type="person", photo=None, voice=None, facts=None):
        """Создаёт или обновляет сущность с данными"""
        # 🔥 Нормализуем имя
        name = self.db.normalize_name(name)

        # Создаём сущность
        entity_id = self.db.create_entity(name, entity_type)
        if not entity_id:
            return None

        # Сохраняем фото
        if photo is not None:
            self.save_photo(entity_id, name, photo)

        # Сохраняем голос
        if voice is not None:
            self.save_voice(entity_id, name, voice)

        # Сохраняем факты
        if facts:
            for fact in facts:
                self.db.add_entity_data(entity_id, "fact", fact)

        return entity_id

    def add_fact(self, entity_name, fact_text, category=None):
        """Добавляет факт о сущности"""
        # 🔥 Нормализуем имя
        entity_name = self.db.normalize_name(entity_name)

        entity_id = self.db.get_entity_by_name(entity_name)
        if not entity_id:
            entity_id = self.db.create_entity(entity_name, "person")

        metadata = {"category": category} if category else None
        self.db.add_entity_data(entity_id, "fact", fact_text, metadata)
        print(f"[ENTITY] Добавлен факт: {fact_text}")
        return True

    def create_relationship_safe(self, entity1_name, entity2_name, relationship_type):
        """Создаёт связь, только если её ещё нет"""
        # 🔥 Нормализуем имена
        entity1_name = self.db.normalize_name(entity1_name)
        entity2_name = self.db.normalize_name(entity2_name)

        entity1_id = self.db.get_entity_by_name(entity1_name)
        entity2_id = self.db.get_entity_by_name(entity2_name)

        if not entity1_id:
            entity1_id = self.db.create_entity(entity1_name, "person")
        if not entity2_id:
            entity2_id = self.db.create_entity(entity2_name, "person")

        # Проверяем дубликат
        if self.db.relationship_exists(entity1_id, entity2_id, relationship_type):
            print(f"[ENTITY] Связь уже существует: {entity1_name} --{relationship_type}--> {entity2_name}")
            return False

        self.db.create_relationship(entity1_id, entity2_id, relationship_type)
        print(f"[ENTITY] Создана связь: {entity1_name} --{relationship_type}--> {entity2_name}")
        return True

    def save_photo(self, entity_id, name, frame):
        """Сохраняет фото сущности"""
        try:
            filename = f"{name}_{entity_id}.jpg"
            filepath = os.path.join(self.media_dir, "photos", filename)
            cv2.imwrite(filepath, frame)
            self.db.add_entity_data(entity_id, "photo", filepath)
            print(f"[ENTITY] Сохранено фото: {filepath}")
            return filepath
        except Exception as e:
            print(f"[ENTITY Error] save_photo: {e}")
            return None

    def save_voice(self, entity_id, name, audio_data):
        """Сохраняет образец голоса сущности"""
        try:
            filename = f"{name}_{entity_id}.wav"
            filepath = os.path.join(self.media_dir, "voices", filename)
            # Здесь нужна логика записи аудио
            # Пока просто сохраняем путь
            self.db.add_entity_data(entity_id, "voice", filepath)
            print(f"[ENTITY] Сохранён голос: {filepath}")
            return filepath
        except Exception as e:
            print(f"[ENTITY Error] save_voice: {e}")
            return None

    def add_fact(self, entity_name, fact_text, category=None):
        """Добавляет факт о сущности"""
        entity_id = self.db.get_entity_by_name(entity_name)
        if not entity_id:
            entity_id = self.db.create_entity(entity_name, "person")

        metadata = {"category": category} if category else None
        self.db.add_entity_data(entity_id, "fact", fact_text, metadata)
        print(f"[ENTITY] Добавлен факт: {fact_text}")
        return True

    def create_relationship(self, entity1_name, entity2_name, relationship_type):
        """Создаёт связь между сущностями"""
        entity1_id = self.db.get_entity_by_name(entity1_name)
        entity2_id = self.db.get_entity_by_name(entity2_name)

        if not entity1_id:
            entity1_id = self.db.create_entity(entity1_name, "person")
        if not entity2_id:
            entity2_id = self.db.create_entity(entity2_name, "person")

        self.db.create_relationship(entity1_id, entity2_id, relationship_type)
        print(f"[ENTITY] Создана связь: {entity1_name} --{relationship_type}--> {entity2_name}")
        return True

    def get_entity_full_info(self, entity_name):
        """Получает полную информацию о сущности"""
        entity_id = self.db.get_entity_by_name(entity_name)
        if not entity_id:
            return None

        return self.db.get_entity_info(entity_id)

    def update_fact(self, entity_name, new_fact_text, category=None):
        """Обновляет факт о сущности (заменяет последний)"""
        entity_id = self.db.get_entity_by_name(entity_name)
        if not entity_id:
            entity_id = self.db.create_entity(entity_name, "person")

        metadata = {"category": category} if category else None
        self.db.update_entity_data(entity_id, "fact", new_fact_text, metadata)
        print(f"[ENTITY] Обновлён факт: {new_fact_text}")
        return True

    def delete_entity(self, entity_name):
        """Удаляет сущность и все её данные"""
        entity_id = self.db.get_entity_by_name(entity_name)
        if not entity_id:
            return False

        self.db.delete_entity_data(entity_id)
        self.db.cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self.db.cursor.execute("DELETE FROM relationships WHERE entity1_id = ? OR entity2_id = ?",
                               (entity_id, entity_id))
        self.db.conn.commit()
        print(f"[ENTITY] Удалена сущность: {entity_name}")
        return True

    def create_relationship_safe(self, entity1_name, entity2_name, relationship_type):
        """Создаёт связь, только если её ещё нет"""
        entity1_id = self.db.get_entity_by_name(entity1_name)
        entity2_id = self.db.get_entity_by_name(entity2_name)

        if not entity1_id:
            entity1_id = self.db.create_entity(entity1_name, "person")
        if not entity2_id:
            entity2_id = self.db.create_entity(entity2_name, "person")

        # Проверяем дубликат
        if self.db.relationship_exists(entity1_id, entity2_id, relationship_type):
            print(f"[ENTITY] Связь уже существует: {entity1_name} --{relationship_type}--> {entity2_name}")
            return False

        self.db.create_relationship(entity1_id, entity2_id, relationship_type)
        print(f"[ENTITY] Создана связь: {entity1_name} --{relationship_type}--> {entity2_name}")
        return True