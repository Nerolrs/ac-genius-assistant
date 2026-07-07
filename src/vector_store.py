import numpy as np
from typing import List, Dict, Optional
import json
import os
import hashlib


class VectorStore:
    """轻量级向量存储管理器 - 纯Python实现"""

    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.persist_directory = persist_directory
        self.documents = []
        self.vectors = []
        self.ids = []
        os.makedirs(persist_directory, exist_ok=True)
        self._load_from_disk()
        print("[INFO] 向量存储已初始化，当前文档数: {}".format(len(self.documents)))

    def _text_to_vector(self, text: str) -> np.ndarray:
        hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        dim = 384
        vector = np.zeros(dim)
        for i in range(dim):
            vector[i] = (hash_val >> (i % 4 * 8)) % 256 / 255.0

        words = text.lower().split()
        for i, word in enumerate(words[:20]):
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            pos = i % dim
            vector[pos] = (vector[pos] + (word_hash % 256 / 255.0)) / 2

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _save_to_disk(self):
        data = {
            "documents": self.documents,
            "ids": self.ids
        }
        with open(os.path.join(self.persist_directory, "vectors.npy"), 'wb') as f:
            if len(self.vectors) > 0:
                np.save(f, np.array(self.vectors))

        with open(os.path.join(self.persist_directory, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self):
        vec_path = os.path.join(self.persist_directory, "vectors.npy")
        meta_path = os.path.join(self.persist_directory, "metadata.json")

        if os.path.exists(vec_path) and os.path.exists(meta_path):
            try:
                self.vectors = np.load(vec_path).tolist()
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.ids = data.get("ids", [])
                print("[INFO] 从磁盘加载了 {} 条文档".format(len(self.documents)))
            except Exception as e:
                print("[WARN] 加载失败: {}, 重新初始化".format(str(e)))
                self.documents = []
                self.vectors = []
                self.ids = []
        else:
            self.documents = []
            self.vectors = []
            self.ids = []

    def add_documents(self, documents: List[Dict], ids: Optional[List[str]] = None):
        """添加文档到向量库"""
        for idx, doc in enumerate(documents):
            doc_id = ids[idx] if ids else "doc_{}".format(idx)

            if doc_id in self.ids:
                print("[WARN] 文档ID已存在: {}".format(doc_id))
                continue

            text = "问题: {}\n答案: {}".format(doc['question'], doc['answer'])
            vector = self._text_to_vector(text).tolist()

            self.documents.append(doc)
            self.vectors.append(vector)
            self.ids.append(doc_id)

        self._save_to_disk()
        print("[OK] 已添加 {} 条知识到向量库".format(len(documents)))

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """语义搜索 - 返回相似度匹配的文档"""
        if len(self.vectors) == 0:
            return []

        query_vector = self._text_to_vector(query)
        vectors_np = np.array(self.vectors)

        similarities = np.dot(vectors_np, query_vector) / (
            np.linalg.norm(vectors_np, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_docs = []
        for idx in top_indices:
            doc = self.documents[idx]
            matched_docs.append({
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "score": round(float(similarities[idx]), 4)
            })

        return matched_docs

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义搜索 - 返回带详细分数的结果"""
        if len(self.vectors) == 0:
            return []

        query_vector = self._text_to_vector(query)
        vectors_np = np.array(self.vectors)

        similarities = np.dot(vectors_np, query_vector) / (
            np.linalg.norm(vectors_np, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_docs = []
        for idx in top_indices:
            doc = self.documents[idx]
            matched_docs.append({
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "score": round(float(similarities[idx]), 4),
                "distance": round(1 - float(similarities[idx]), 4),
                "document": "问题: {}\n答案: {}".format(doc["question"], doc["answer"])
            })

        return matched_docs

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """根据ID获取文档"""
        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            doc = self.documents[idx]
            return {
                "id": doc_id,
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "keywords": doc.get("keywords", []),
                "document": "问题: {}\n答案: {}".format(doc["question"], doc["answer"])
            }
        return None

    def update_document(self, doc_id: str, document: Dict):
        """更新指定ID的文档"""
        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            self.documents[idx] = document

            text = "问题: {}\n答案: {}".format(document['question'], document['answer'])
            self.vectors[idx] = self._text_to_vector(text).tolist()

            self._save_to_disk()
            print("[OK] 已更新文档: {}".format(doc_id))
        else:
            print("[WARN] 文档不存在: {}".format(doc_id))

    def delete_document(self, doc_id: str):
        """删除指定ID的文档"""
        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            del self.documents[idx]
            del self.vectors[idx]
            del self.ids[idx]

            self._save_to_disk()
            print("[OK] 已删除文档: {}".format(doc_id))
        else:
            print("[WARN] 文档不存在: {}".format(doc_id))

    def delete_by_category(self, category: str):
        """按类别删除文档"""
        to_delete = []
        for idx, doc in enumerate(self.documents):
            if doc.get("category") == category:
                to_delete.append(idx)

        for idx in sorted(to_delete, reverse=True):
            del self.documents[idx]
            del self.vectors[idx]
            del self.ids[idx]

        self._save_to_disk()
        print("[OK] 已删除 {} 条 {} 类文档".format(len(to_delete), category))

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        docs = []
        for idx, doc in enumerate(self.documents):
            docs.append({
                "id": self.ids[idx],
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "keywords": doc.get("keywords", []),
                "document": "问题: {}\n答案: {}".format(doc["question"], doc["answer"])
            })
        return docs

    def get_categories(self) -> List[str]:
        """获取所有类别"""
        categories = set()
        for doc in self.documents:
            categories.add(doc.get("category", "未分类"))
        return sorted(list(categories))

    def count(self) -> int:
        """获取文档总数"""
        return len(self.documents)

    def get_statistics(self) -> Dict:
        """获取向量库统计信息"""
        categories = self.get_categories()
        count = self.count()

        category_counts = {}
        for cat in categories:
            cnt = sum(1 for doc in self.documents if doc.get("category") == cat)
            category_counts[cat] = cnt

        return {
            "total_documents": count,
            "total_categories": len(categories),
            "categories": categories,
            "category_distribution": category_counts,
            "embedding_type": "simple_hash"
        }

    def clear(self):
        """清空向量库"""
        self.documents = []
        self.vectors = []
        self.ids = []
        self._save_to_disk()
        print("[OK] 向量库已清空")

    def rebuild_index(self):
        """重建索引"""
        if len(self.documents) > 0:
            docs = self.get_all_documents()
            self.clear()
            ids = [doc["id"] for doc in docs]
            documents = [
                {
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "category": doc["category"],
                    "keywords": doc["keywords"]
                }
                for doc in docs
            ]
            self.add_documents(documents, ids)
            print("[OK] 索引已重建")
        else:
            print("[WARN] 向量库为空，无需重建")