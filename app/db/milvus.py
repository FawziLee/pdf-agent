import os
from functools import lru_cache
from pathlib import Path

from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

# 项目根目录 / data / milvus.db（Lite）；生产 BM25 建议改连 Standalone
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_URI = str(DATA_DIR / "milvus.db")

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "document_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """全局只创建一次 MilvusClient。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uri = os.getenv("MILVUS_URI", DEFAULT_URI)
    token = os.getenv("MILVUS_TOKEN")  # 远程如 root:Milvus；Lite 可不设
    kwargs = {"uri": uri}
    if token:
        kwargs["token"] = token
    client = MilvusClient(**kwargs)
    logger.info(f"MilvusClient 已初始化：{uri}")
    return client


def ensure_collection(client: MilvusClient | None = None) -> None:
    """若 collection 不存在则创建（含稠密向量 + BM25 稀疏字段）；已存在则跳过。

    注意：
    - 改 schema 不会自动迁移，需删旧 collection / 换库文件后重建。
    - BM25 全文检索在 Milvus Standalone/Distributed 上完整支持；
      Lite 可能不支持，届时把 MILVUS_URI 换成 http://localhost:19530。
    """
    client = client or get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Milvus collection 已存在，跳过创建：{COLLECTION_NAME}")
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 稠密向量：语义检索（embedding）
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    # 原文：开启分词，供 BM25 Function 使用
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
    )
    # 稀疏向量：由 BM25 Function 自动从 text 生成，插入时不必传
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 元数据（过滤 / 溯源）
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="page_idx", datatype=DataType.INT64)
    schema.add_field(field_name="bbox", datatype=DataType.VARCHAR, max_length=128)

    # text → sparse（BM25）
    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    # 语义检索索引
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    # BM25 全文检索索引
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(
        f"已创建 collection：{COLLECTION_NAME}, "
        f"dense_dim={EMBEDDING_DIM}, sparse=BM25"
    )


def init_milvus() -> MilvusClient:
    """应用启动：创建 client，并确保 collection 就绪。"""
    client = get_milvus_client()
    ensure_collection(client)
    return client
