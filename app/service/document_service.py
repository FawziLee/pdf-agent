import uuid
import os
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from loguru import logger
from app.db import session
from app.models.document import Document
# from app.config.settings import settings
# from app.utils.exception import BusinessException

ALLOW_FILE_EXTENSIONS = ["pdf"]
MAX_UPLOAD_FILE_SIZE = 200 * 1024 * 1024


class DocumentProcessService:

    def __init__(self):
        self.upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/upload")

    async def save_upload_file(
        self,
        file: UploadFile,
        user_id: str,
        db: Session,
        tenant_id: str = "demo-tenant",
    ) -> Document:
        """保存上传的文件，创建文档元数据"""
        os.makedirs(self.upload_dir, exist_ok=True)
        # 1. 校验文件格式
        file_ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
        if file_ext not in ALLOW_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，支持的格式：{','.join(ALLOW_FILE_EXTENSIONS)}"
            )
        # 2. 校验文件大小
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > MAX_UPLOAD_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制，最大支持{MAX_UPLOAD_FILE_SIZE/1024/1024}MB"
            )
        # 3. 重置文件指针
        await file.seek(0)
        # 4. 生成唯一文档ID
        document_id = uuid.uuid4().hex
        # 5. 生成文件存储路径
        file_name = file.filename
        save_file_name = f"{document_id}_{file_name}"
        save_path = os.path.join(self.upload_dir, save_file_name)
        # 6. 保存文件到本地
        with open(save_path, "wb") as f:
            f.write(file_content)
        # 7. 创建文档元数据
        document = Document(
            document_id=document_id,
            file_name=file_name,
            file_ext=file_ext,
            file_size=file_size,
            file_path=save_path,
            tenant_id=tenant_id,
            created_by=user_id,
            status=0  # 处理中
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"文件保存成功，文档ID：{document_id}，文件名：{file_name}")
        return document

    def get_document_list(self, tenant_id: int, page: int, page_size: int, db: Session, document_name: str = None):
        """获取租户下的文档列表，分页查询"""
        query = db.query(Document).filter(Document.tenant_id == tenant_id, Document.is_deleted == False)
        # 按文件名模糊查询
        if document_name:
            query = query.filter(Document.file_name.like(f"%{document_name}%"))
        # 总条数
        total = query.count()
        # 分页查询
        offset = (page - 1) * page_size
        list = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size).all()
        # 总页数
        total_page = (total + page_size - 1) // page_size
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_page": total_page,
            "list": [
                {
                    "document_id": d.document_id,
                    "file_name": d.file_name,
                    "file_ext": d.file_ext,
                    "file_size": d.file_size,
                    "status": d.status,
                    "chunk_count": d.chunk_count,
                    "summary": d.summary,
                    "created_by": d.created_by,
                    "tenant_id": d.tenant_id,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in list
            ],
        }

    def get_document_by_id(self, document_id: str, tenant_id: int, db: Session) -> Document:
        """根据文档ID获取文档信息，校验租户权限"""
        document = db.query(Document).filter(Document.document_id == document_id, Document.tenant_id == tenant_id).first()
        if not document:
            raise HTTPException(
                status_code=404,
                detail="文档不存在"
            )
        return document

    def delete_document(self, document_id: str, tenant_id: int, db: Session) -> bool:
        """删除文档，同时删除向量库、关键词索引和本地文件"""
        document = self.get_document_by_id(document_id, tenant_id, db)
        try:
            # 1. 删除向量库中的数据
            self.vector_db.delete_by_document_id(document_id, str(tenant_id))
            # 2. 删除关键词索引中的数据
            self.keyword_retriever.delete_by_document_id(document_id, str(tenant_id))
            # 3. 删除本地文件
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            # 4. 软删除数据库中的元数据
            document.is_deleted = True
            db.commit()
            logger.info(f"文档删除成功，文档ID：{document_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"文档删除失败，文档ID：{document_id}，错误信息：{str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="文档删除失败"
            )

# 全局单例
document_process_service = DocumentProcessService()
