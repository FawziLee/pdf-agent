from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DocumentInfoResponse(BaseModel):
    id: int
    document_id: str
    file_name: str
    file_ext: str
    file_size: int
    chunk_count: int
    status: int
    failed_reason: str
    created_by: int
    created_at: datetime
    updated_at: datetime


@dataclass
class OcrBlock:
    """单个版面文本块（含坐标，供问答溯源）"""
    page_idx: int
    block_label: str
    block_content: str
    block_bbox: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OcrPage:
    """单页 markdown 结果"""
    markdown: str
    images: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentContentTree:
    """按标题层级组织的内容树节点"""
    title: str
    level: int
    page_idx: int | None = None
    bbox: list[int] | None = None
    block: OcrBlock | None = None
    children: list[DocumentContentTree] = field(default_factory=list)
    content: list[OcrBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "page_idx": self.page_idx,
            "bbox": self.bbox,
            "block": self.block.to_dict() if self.block else None,
            "children": [child.to_dict() for child in self.children],
            "content": [item.to_dict() for item in self.content],
        }


@dataclass
class OcrParseResult:
    """PaddleOcrTool.parse 的统一返回
    Args:
        total_pages: 总页数
        pages: 页数列表
        result: 原始合并 JSON
        extracted_result: 提取出的结果
        content_tree: 内容树
    """
    total_pages: int
    pages: list[OcrPage]
    result: dict[str, Any]  # 原始合并 JSON，先保留 dict
    extracted_result: list[OcrBlock]
    content_tree: DocumentContentTree

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "result": self.result,
            "extracted_result": [b.to_dict() for b in self.extracted_result],
            "content_tree": self.content_tree.to_dict(),
        }
