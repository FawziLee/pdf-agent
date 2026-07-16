from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.db.session import init_db
from app.db.milvus import init_milvus
from app.api.v1.document import document_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 SQLite / Milvus（各创建一次）
    init_db()
    init_milvus()
    yield


app = FastAPI(
    title="pdf-agent",
    description="pdf-agent demo",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.include_router(document_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "project_name": "pdf-agent",
        "version": "0.1.0",
    }

# 项目启动入口
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 使用FastAPI应用
        host="0.0.0.0",
        port=5557,
        reload=True,
        reload_dirs=["."],   # 监听整个项目
    )
