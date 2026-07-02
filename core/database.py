"""Управление SQLite базой данных с графом сущностей"""
import sqlite3
import json
from config import SQLITE_DB_PATH


class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    @staticmethod
    def normalize_name(name):
        """Нормализует имя: приводит к единому формату (кириллица)"""
        if not name:
            return name

        translit_map = {
            'a': 'а', 'b': 'б', 'v': 'в', 'g': 'г', 'd': 'д', 'e': 'е', 'z': 'з', 'i': 'и',
            'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'r': 'р', 's': 'с',
            't': 'т', 'u': 'у', 'f': 'ф', 'h': 'х', 'c': 'ц', 'y': 'ы',
            'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'E': 'Е', 'Z': 'З', 'I': 'И',
            'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р', 'S': 'С',
            'T': 'Т', 'U': 'У', 'F': 'Ф', 'H': 'Х', 'C': 'Ц', 'Y': 'Ы'
        }

        # Если имя полностью на латинице — транслитерируем
        if all(c.isascii() and c.isalpha() for c in name):
            return ''.join(translit_map.get(c, c) for c in name)

        # Иначе возвращаем с заглавной буквы
        return name.capitalize()

    def _create_tables(self):
        # Сущности
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, type)
            )
        ''')

        # Данные о сущностях
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                data_value TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            )
        ''')

        # Связи между сущностями
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1_id INTEGER NOT NULL,
                entity2_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity1_id) REFERENCES entities(id),
                FOREIGN KEY (entity2_id) REFERENCES entities(id)
            )
        ''')

        # Диалоги
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                emotion TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    # ========== УПРАВЛЕНИЕ СУЩНОСТЯМИ ==========

    def create_entity(self, name, entity_type="person"):
        """Создаёт новую сущность"""
        try:
            self.cursor.execute('''
                INSERT INTO entities (name, type) VALUES (?, ?)
            ''', (name, entity_type))
            self.conn.commit()
            entity_id = self.cursor.lastrowid
            print(f"[DB] Создана сущность: {name} ({entity_type})")
            return entity_id
        except sqlite3.IntegrityError:
            # Сущность уже существует — получаем её ID
            return self.get_entity_by_name(name)
        except Exception as e:
            print(f"[DB Error] create_entity: {e}")
            return None

    def get_entity_by_name(self, name):
        """Получает ID сущности по имени"""
        try:
            self.cursor.execute('''
                SELECT id FROM entities WHERE name = ?
            ''', (name,))
            row = self.cursor.fetchone()
            return row[0] if row else None
        except:
            return None

    def get_entity_info(self, entity_id):
        """Получает всю информацию о сущности"""
        try:
            self.cursor.execute('''
                SELECT name, type FROM entities WHERE id = ?
            ''', (entity_id,))
            entity = self.cursor.fetchone()
            if not entity:
                return None

            name, entity_type = entity

            # Данные (фото, голос, факты)
            self.cursor.execute('''
                SELECT data_type, data_value, metadata FROM entity_data 
                WHERE entity_id = ? ORDER BY created_at DESC
            ''', (entity_id,))
            data = self.cursor.fetchall()

            # Связи
            self.cursor.execute('''
                SELECT e.name, r.relationship_type 
                FROM relationships r
                JOIN entities e ON (r.entity2_id = e.id AND r.entity1_id = ?)
                    OR (r.entity1_id = e.id AND r.entity2_id = ?)
            ''', (entity_id, entity_id))
            relationships = self.cursor.fetchall()

            return {
                "name": name,
                "type": entity_type,
                "data": data,
                "relationships": relationships
            }
        except Exception as e:
            print(f"[DB Error] get_entity_info: {e}")
            return None

    # ========== УПРАВЛЕНИЕ ДАННЫМИ ==========

    def add_entity_data(self, entity_id, data_type, data_value, metadata=None):
        """Добавляет данные о сущности (фото, голос, факт)"""
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            self.cursor.execute('''
                INSERT INTO entity_data (entity_id, data_type, data_value, metadata)
                VALUES (?, ?, ?, ?)
            ''', (entity_id, data_type, data_value, metadata_json))
            self.conn.commit()
            print(f"[DB] Добавлены данные: {data_type} для entity_id={entity_id}")
            return True
        except Exception as e:
            print(f"[DB Error] add_entity_data: {e}")
            return False

    def update_entity_data(self, entity_id, data_type, new_value, new_metadata=None):
        """Обновляет существующие данные сущности (заменяет последний факт по типу)"""
        try:
            self.cursor.execute('''
                SELECT id FROM entity_data 
                WHERE entity_id = ? AND data_type = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (entity_id, data_type))
            row = self.cursor.fetchone()

            if row:
                metadata_json = json.dumps(new_metadata) if new_metadata else None
                self.cursor.execute('''
                    UPDATE entity_data 
                    SET data_value = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_value, metadata_json, row[0]))
                self.conn.commit()
                print(f"[DB] Обновлено: {data_type} = {new_value}")
                return True
            else:
                return self.add_entity_data(entity_id, data_type, new_value, new_metadata)
        except Exception as e:
            print(f"[DB Error] update_entity_data: {e}")
            return False

    def delete_entity_data(self, entity_id, data_type=None):
        """Удаляет данные сущности (все или по типу)"""
        try:
            if data_type:
                self.cursor.execute('''
                    DELETE FROM entity_data WHERE entity_id = ? AND data_type = ?
                ''', (entity_id, data_type))
            else:
                self.cursor.execute('''
                    DELETE FROM entity_data WHERE entity_id = ?
                ''', (entity_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error] delete_entity_data: {e}")
            return False

    def get_entity_data(self, entity_id, data_type=None):
        """Получает данные сущности (опционально по типу)"""
        try:
            if data_type:
                self.cursor.execute('''
                    SELECT data_value, metadata FROM entity_data 
                    WHERE entity_id = ? AND data_type = ?
                    ORDER BY created_at DESC
                ''', (entity_id, data_type))
            else:
                self.cursor.execute('''
                    SELECT data_type, data_value, metadata FROM entity_data 
                    WHERE entity_id = ? ORDER BY created_at DESC
                ''', (entity_id,))
            return self.cursor.fetchall()
        except:
            return []

    # ========== УПРАВЛЕНИЕ СВЯЗЯМИ ==========

    def create_relationship(self, entity1_id, entity2_id, relationship_type):
        """Создаёт связь между сущностями"""
        try:
            self.cursor.execute('''
                INSERT INTO relationships (entity1_id, entity2_id, relationship_type)
                VALUES (?, ?, ?)
            ''', (entity1_id, entity2_id, relationship_type))
            self.conn.commit()
            print(f"[DB] Создана связь: {relationship_type}")
            return True
        except Exception as e:
            print(f"[DB Error] create_relationship: {e}")
            return False

    def relationship_exists(self, entity1_id, entity2_id, relationship_type):
        """Проверяет, существует ли уже такая связь"""
        try:
            self.cursor.execute('''
                SELECT id FROM relationships 
                WHERE ((entity1_id = ? AND entity2_id = ?) 
                    OR (entity1_id = ? AND entity2_id = ?))
                    AND relationship_type = ?
            ''', (entity1_id, entity2_id, entity2_id, entity1_id, relationship_type))
            return self.cursor.fetchone() is not None
        except:
            return False

    def get_relationships(self, entity_id):
        """Получает все связи сущности"""
        try:
            self.cursor.execute('''
                SELECT e.name, r.relationship_type, r.entity1_id, r.entity2_id
                FROM relationships r
                JOIN entities e ON (r.entity2_id = e.id AND r.entity1_id = ?)
                    OR (r.entity1_id = e.id AND r.entity2_id = ?)
            ''', (entity_id, entity_id))
            return self.cursor.fetchall()
        except:
            return []

    # ========== ДИАЛОГИ ==========

    def save_conversation(self, user_name, user_msg, assistant_msg, emotion):
        try:
            self.cursor.execute('''
                INSERT INTO conversations (user_name, user_message, assistant_response, emotion)
                VALUES (?, ?, ?, ?)
            ''', (user_name, user_msg, assistant_msg, emotion))
            self.conn.commit()
        except Exception as e:
            print(f"[DB Error] save_conversation: {e}")

    def close(self):
        try:
            self.conn.close()
        except:
            pass