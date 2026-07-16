from typing import Annotated
import json

from fastapi import APIRouter, UploadFile, File, Query, Depends
from sqlalchemy.orm import Session

from app.schemas.common import ResponseModel, PageQuery
from app.service.document_service import document_process_service
from app.db.session import get_db
from app.agent import PdfAgent

document_router = APIRouter(prefix="/documents", tags=["documents"])
DBSession = Annotated[Session, Depends(get_db)]
pdf_agent = PdfAgent()


def _parse_section_summaries(raw: str | None) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


@document_router.post("/upload", summary="上传文档")
async def upload_document(
    db: DBSession,
    file: UploadFile = File(..., description="上传的文档文件"),
    user_id: str = Query("demo-user", description="上传用户ID"),
    tenant_id: str = Query("demo-tenant", description="上传租户ID"),
):
    document = await document_process_service.save_upload_file(
        file=file,
        tenant_id=tenant_id,
        user_id=user_id,
        db=db,
    )
    ocr_result = await pdf_agent.load_pdf(document.file_path, document.document_id, document.file_name)
    summary_result = await pdf_agent.generate_summary(ocr_result)
    document.summary = summary_result.overall
    document.section_summaries = summary_result.sections_to_json()
    db.commit()
    db.refresh(document)

    return ResponseModel.success(
        data={
            "document_id": document.document_id,
            "file_name": document.file_name,
            "status": document.status,
            "file_path": document.file_path,
            "summary": document.summary,
            "section_summaries": _parse_section_summaries(document.section_summaries),
        },
        message="文档上传成功（元数据已写入 SQLite）",
    )


@document_router.get("/list", summary="获取文档列表")
async def get_document_list(
    db: DBSession,
    page_query: PageQuery = Depends(),
    document_name: str | None = Query(None, description="文档名称，模糊查询"),
    tenant_id: str = Query("demo-tenant", description="租户ID"),
):
    result = document_process_service.get_document_list(
        tenant_id=tenant_id,
        page=page_query.page,
        page_size=page_query.page_size,
        db=db,
        document_name=document_name,
    )
    return ResponseModel.success(data=result)


@document_router.get("/{document_id}", summary="获取文档详情")
async def get_document_detail(
    document_id: str,
    db: DBSession,
    tenant_id: str = Query("demo-tenant", description="租户ID"),
):
    document = document_process_service.get_document_by_id(document_id, tenant_id, db)
    return ResponseModel.success(
        data={
            "document_id": document.document_id,
            "file_name": document.file_name,
            "file_ext": document.file_ext,
            "file_size": document.file_size,
            "status": document.status,
            "summary": document.summary,
            "section_summaries": _parse_section_summaries(document.section_summaries),
            "created_by": document.created_by,
            "tenant_id": document.tenant_id,
        }
    )


@document_router.delete("/{document_id}", summary="删除文档")
async def delete_document(
    document_id: str,
    db: DBSession,
    tenant_id: str = Query("demo-tenant", description="租户ID"),
):
    document_process_service.delete_document(document_id, tenant_id, db)
    return ResponseModel.success(message="文档删除成功")
