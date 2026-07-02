from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.assistant import ACGeniusAssistant
import uvicorn

app = FastAPI(
    title="智能空调精灵 API",
    description="基于 SenseVoice 和向量检索的智能空调助手",
    version="1.0.0"
)

# 全局助手实例
assistant = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class BatchQueryRequest(BaseModel):
    queries: List[str]


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global assistant
    print("🚀 正在启动智能空调精灵 API...")
    assistant = ACGeniusAssistant()
    print("✅ API 服务已就绪")


@app.get("/")
async def root():
    return {
        "service": "AC Genius Assistant API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/query")
async def query_knowledge(request: QueryRequest):
    """文本查询接口"""
    if not assistant:
        raise HTTPException(status_code=503, detail="助手未初始化")

    result = assistant.process_text_query(request.query)
    return result


@app.post("/batch_query")
async def batch_query_knowledge(request: BatchQueryRequest):
    """批量查询接口"""
    if not assistant:
        raise HTTPException(status_code=503, detail="助手未初始化")

    results = assistant.batch_query(request.queries)
    return {"results": results}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "assistant_ready": assistant is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
