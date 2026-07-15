from fastapi import APIRouter
from app.api.v1.document import document_router

api_v1_router = APIRouter()
api_v1_router.include_router(document_router, prefix="/document", tags=["文档管理"])