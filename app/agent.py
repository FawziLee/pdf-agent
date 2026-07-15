from typing import List, Dict, Any
from abc import ABC

from app.core.llm_engine import LLMEngine, get_llm_engine
from app.core.document_loader import PDFDocumentLoader
from app.schemas.document import OcrParseResult


llm_engine = get_llm_engine()


class Agent(ABC):
    def __init__(self, name: str, description: str, tools: List[Dict[str, Any]], llm_engine: LLMEngine):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm_engine = llm_engine


class PdfAgent(Agent):

    def __init__(self, pdf_path: str, document_id: str):
        super().__init__(name="PdfAgent", description="PdfAgent", tools=[], llm_engine=llm_engine)

    async def load_pdf(self, pdf_path: str, document_id: str, document_name: str) -> OcrParseResult:
        pdf_loader = PDFDocumentLoader(
            file_path=pdf_path,
            document_id=document_id,
            document_name=document_name,
        )
        return await pdf_loader.load()

    async def generate_summary(self, ocr_parse_result: OcrParseResult) -> str:
        summary = await self.llm_engine.chat(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的摘要生成器，请根据以下内容生成摘要。",
                },
            ],
        )
        return summary

    
