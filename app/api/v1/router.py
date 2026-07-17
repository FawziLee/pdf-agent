from fastapi import APIRouter
from app.api.v1.document import document_router
from app.api.v1.agent import agent_router

api_v1_router = APIRouter()
api_v1_router.include_router(document_router, tags=["文档管理"])
api_v1_router.include_router(agent_router, tags=["智能体服务"])


