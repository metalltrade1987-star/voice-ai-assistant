"""Векторная база знаний (RAG)"""
import os
import time
import chromadb
from config import KNOWLEDGE_DOCS_DIR, CHROMA_DB_DIR


class KnowledgeBase:
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory or CHROMA_DB_DIR
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[DB] База знаний загружена. Документов: {self.collection.count()}")

    def add_document(self, text, metadata=None, doc_id=None):
        try:
            if doc_id is None:
                doc_id = f"doc_{int(time.time())}_{hash(text) % 10000}"
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {"source": "manual"}],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"[DB Error] add_document: {e}")

    def search(self, query, n_results=3):
        try:
            if self.collection.count() == 0:
                return []
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results['documents'][0] if results['documents'] else []
        except:
            return []

    def load_documents_from_folder(self, folder_path=None):
        folder_path = folder_path or KNOWLEDGE_DOCS_DIR
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return

        # 🔥 Очищаем коллекцию перед загрузкой (чтобы не было дубликатов)
        try:
            self.collection.delete(
                where={"source": {"$in": [f"file_{f}" for f in os.listdir(folder_path) if f.endswith('.txt')]}})
        except:
            pass

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():
                        self.add_document(text=content, metadata={"source": filename}, doc_id=f"file_{filename}")
                        print(f"[DB] Документ добавлен: file_{filename}")
                except Exception as e:
                    print(f"[DB Error] load {filename}: {e}")