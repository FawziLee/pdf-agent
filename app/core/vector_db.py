"""向量库业务封装：基于 app.db.milvus 的 MilvusClient。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
from loguru import logger

from app.db.milvus import COLLECTION_NAME, ensure_collection, get_milvus_client

OUTPUT_FIELDS = ["document_id", "tenant_id", "page_idx", "text", "bbox"]


class VectorDBManager:
    """文档向量块：插入 / 稠密检索 / BM25 检索 / 按文档删除。"""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = get_milvus_client()
        ensure_collection(self.client)

    @staticmethod
    def _build_filter(
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> str:
        expr = f'tenant_id == "{tenant_id}"'
        if document_ids:
            ids = ", ".join(f'"{d}"' for d in document_ids)
            expr += f" and document_id in [{ids}]"
        if filter_expr:
            expr += f" and ({filter_expr})"
        return expr

    @staticmethod
    def _format_hits(raw: Any, source: str) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        hits = raw[0] if raw else []
        for hit in hits:
            entity = hit.get("entity") or hit
            formatted.append(
                {
                    "score": hit.get("distance", hit.get("score")),
                    "id": hit.get("id"),
                    "text": entity.get("text"),
                    "document_id": entity.get("document_id"),
                    "tenant_id": entity.get("tenant_id"),
                    "page_idx": entity.get("page_idx"),
                    "bbox": entity.get("bbox"),
                    "source": source,
                }
            )
        return formatted

    def insert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
    ) -> int:
        """
        批量插入分块向量。

        chunks 每项建议包含：
            document_id, tenant_id, page_idx, text, bbox(可选 list/str)
        vectors: shape (n, dim)

        注意：sparse 由 collection 上的 BM25 Function 根据 text 自动生成，插入时不用传。
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"分块数量({len(chunks)})与向量数量({len(vectors)})不匹配"
            )
        if len(chunks) == 0:
            return 0

        rows: list[dict[str, Any]] = []
        for chunk, vec in zip(chunks, vectors):
            bbox = chunk.get("bbox", "")
            if isinstance(bbox, (list, tuple)):
                bbox = json.dumps(list(bbox), ensure_ascii=False)
            text = chunk.get("text") or ""
            # VARCHAR 上限保护
            if len(text) > 65000:
                text = text[:65000]

            rows.append(
                {
                    "vector": np.asarray(vec, dtype=np.float32).tolist(),
                    "document_id": str(chunk.get("document_id", "")),
                    "tenant_id": str(chunk.get("tenant_id", "default")),
                    "page_idx": int(chunk.get("page_idx", 0)),
                    "text": text,
                    "bbox": str(bbox) if bbox is not None else "",
                }
            )

        result = self.client.insert(collection_name=self.collection_name, data=rows)
        # MilvusClient 返回可能是 dict，含 insert_count / ids
        insert_count = (
            result.get("insert_count", len(rows))
            if isinstance(result, dict)
            else len(rows)
        )
        logger.info(f"向量插入成功：collection={self.collection_name}, count={insert_count}")
        return int(insert_count)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """稠密向量语义检索（anns_field=vector）。"""
        expr = self._build_filter(tenant_id, document_ids, filter_expr)
        vec = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[vec.tolist()],
            anns_field="vector",
            limit=top_k,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        formatted = self._format_hits(raw, source="dense")
        logger.info(
            f"稠密检索完成：tenant={tenant_id}, top_k={top_k}, hits={len(formatted)}"
        )
        return formatted

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 5,
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 全文检索：直接传原始查询文本，Milvus 自动走 sparse 字段。"""
        query = (query_text or "").strip()
        if not query:
            return []

        expr = self._build_filter(tenant_id, document_ids, filter_expr)
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            filter=expr,
            output_fields=OUTPUT_FIELDS,  # 不要输出 sparse 字段
            search_params={"metric_type": "BM25"},
        )
        formatted = self._format_hits(raw, source="bm25")
        logger.info(
            f"BM25 检索完成：tenant={tenant_id}, top_k={top_k}, hits={len(formatted)}"
        )
        return formatted

    def delete_by_document_id(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> int:
        """按文档删除向量块（更新/删除文档时用）。"""
        expr = f'tenant_id == "{tenant_id}" and document_id == "{document_id}"'
        result = self.client.delete(collection_name=self.collection_name, filter=expr)
        delete_count = (
            result.get("delete_count", 0) if isinstance(result, dict) else 0
        )
        logger.info(
            f"向量删除完成：document_id={document_id}, tenant={tenant_id}, count={delete_count}"
        )
        return int(delete_count)


@lru_cache(maxsize=1)
def get_vector_db_manager() -> VectorDBManager:
    """全局单例。"""
    return VectorDBManager()
