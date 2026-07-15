import uuid
from abc import ABC, abstractmethod

from loguru import logger

from app.core.ocr_tools import PaddleOcrTool
from app.schemas.document import OcrParseResult


class BaseDocumentLoader(ABC):
    def __init__(self, file_path: str, document_id: str = None, document_name: str = None):
        self.file_path = file_path
        self.document_id = document_id or uuid.uuid4().hex
        self.document_name = document_name

    @abstractmethod
    async def load(self) -> OcrParseResult:
        """文档加载方法，子类实现"""
        pass


class PDFDocumentLoader(BaseDocumentLoader):
    """PDF文档加载器"""

    async def load(self) -> OcrParseResult:
        logger.info(f"当前加载文档 {self.file_path}")
        ocr_tool = PaddleOcrTool(
            document_id=self.document_id,
            document_name=self.document_name,
        )
        result: OcrParseResult = await ocr_tool.parse(self.file_path)
        logger.info(
            f"OCR 完成: pages={result.total_pages}, blocks={len(result.extracted_result)}"
        )
        return result
