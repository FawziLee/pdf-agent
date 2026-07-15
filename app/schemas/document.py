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
class DocumentSection:
    """保留到某一级标题后，下级合并成的大文本段（用于总结）"""
    title: str
    level: int
    text: str
    page_idxs: list[int] = field(default_factory=list)
    parent_title: str | None = None

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

    def merge_text(self) -> str:
        """把当前节点及其所有下级标题+正文拼成一段文本。"""
        parts: list[str] = []
        if self.title and self.title != "ROOT":
            parts.append(self.title)
        for block in self.content:
            text = (block.block_content or "").strip()
            if text:
                parts.append(text)
        for child in self.children:
            child_text = child.merge_text()
            if child_text:
                parts.append(child_text)
        return "\n".join(parts)

    def to_summary_sections(self, keep_level: int = 2) -> list[DocumentSection]:
        """保留 keep_level 及以上标题，下级全部合并成大文本段。

        例：keep_level=2 → 保留「一、」「四、」等二级标题，
        其下的「（一）」及正文都并进该节 text。
        """
        sections: list[DocumentSection] = []

        def walk(node: DocumentContentTree, parent_title: str | None = None) -> None:
            if node.title == "ROOT":
                for child in node.children:
                    walk(child, None)
                return

            if node.level < keep_level:
                # 更高级标题（如文档名 L1）只当结构容器往下走
                for child in node.children:
                    walk(child, node.title)
                # 若 L1 下直接有正文且无 L2，也单独成段
                if node.content and not node.children:
                    sections.append(
                        DocumentSection(
                            title=node.title,
                            level=node.level,
                            text=node.merge_text(),
                            page_idxs=_collect_pages(node),
                            parent_title=parent_title,
                        )
                    )
                return

            if node.level == keep_level:
                sections.append(
                    DocumentSection(
                        title=node.title,
                        level=node.level,
                        text=node.merge_text(),
                        page_idxs=_collect_pages(node),
                        parent_title=parent_title,
                    )
                )
                return

            # 比 keep_level 更深的节点：由上级合并，这里不应单独成段
            return

        walk(self)
        return sections


def _collect_pages(node: DocumentContentTree) -> list[int]:
    pages: set[int] = set()
    if node.page_idx is not None:
        pages.add(node.page_idx)
    for block in node.content:
        pages.add(block.page_idx)
    for child in node.children:
        pages.update(_collect_pages(child))
    return sorted(pages)


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
