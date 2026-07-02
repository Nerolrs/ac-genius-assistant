import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Tuple
import json


class VectorStore:
    """向量存储管理器 - 使用 ChromaDB"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name="thermal_knowledge",
            metadata={"description": "空调热管理知识库"}
        )

    def add_documents(self, documents: List[Dict]):
        """添加文档到向量库"""
        ids = []
        texts = []
        metadatas = []

        for idx, doc in enumerate(documents):
            doc_id = f"doc_{idx}"
            # 组合问题和答案作为检索文本
            text = f"问题: {doc['question']}\n答案: {doc['answer']}"
            metadata = {
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "keywords": json.dumps(doc["keywords"], ensure_ascii=False)
            }

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ 已添加 {len(documents)} 条知识到向量库")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """语义搜索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        matched_docs = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for idx, metadata in enumerate(results['metadatas'][0]):
                doc = {
                    "question": metadata["question"],
                    "answer": metadata["answer"],
                    "category": metadata["category"],
                    "score": 1 - results['distances'][0][idx]  # 转换为相似度分数
                }
                matched_docs.append(doc)

        return matched_docs

    def clear(self):
        """清空向量库"""
        self.client.delete_collection("thermal_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="thermal_knowledge"
        )
