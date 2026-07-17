import os
from functools import lru_cache
from pathlib import Path

from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

# 项目根目录 / data / milvus.db（Lite）
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_URI = str(DATA_DIR / "milvus.db")

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "document_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# 中文文档必须用 jieba；Lite 支持 tokenizer=jieba（不支持 type=chinese）
CHINESE_ANALYZER = {"tokenizer": "jieba"}
# 标记当前业务 schema 版本；变更后自动重建 collection
SCHEMA_MARKER = DATA_DIR / ".milvus_schema_v2_jieba"


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """全局只创建一次 MilvusClient。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 避免使用 pymilvus 自身保留的 MILVUS_URI 环境变量。
    # pymilvus 在 import 阶段会把它按远程 http(s) URI 解析，本地文件路径会报 Illegal uri。
    uri = os.getenv("PDF_AGENT_MILVUS_URI", DEFAULT_URI)
    token = os.getenv("MILVUS_TOKEN")  # 远程如 root:Milvus；Lite 可不设
    kwargs = {"uri": uri}
    if token:
        kwargs["token"] = token
    client = MilvusClient(**kwargs)
    logger.info(f"MilvusClient 已初始化：{uri}")
    return client


def _create_collection(client: MilvusClient) -> None:
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params=CHINESE_ANALYZER,
    )
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="page_idx", datatype=DataType.INT64)
    schema.add_field(field_name="bbox", datatype=DataType.VARCHAR, max_length=128)

    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
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
    client.load_collection(COLLECTION_NAME)
    SCHEMA_MARKER.write_text("jieba-bm25-v2\n", encoding="utf-8")
    logger.info(
        f"已创建 collection：{COLLECTION_NAME}, "
        f"dense_dim={EMBEDDING_DIM}, sparse=BM25, analyzer=jieba"
    )


def ensure_collection(client: MilvusClient | None = None) -> None:
    """确保 collection 存在且使用中文分词；旧 schema 会自动重建（需重新上传文档）。"""
    client = client or get_milvus_client()
    force = os.getenv("MILVUS_FORCE_RECREATE", "").lower() in {"1", "true", "yes"}
    schema_ok = SCHEMA_MARKER.exists()

    if client.has_collection(COLLECTION_NAME) and schema_ok and not force:
        logger.info(f"Milvus collection 已存在且为中文分词 schema，跳过创建：{COLLECTION_NAME}")
        client.load_collection(COLLECTION_NAME)
        return

    if client.has_collection(COLLECTION_NAME):
        reason = "force recreate" if force else "旧 schema 使用默认英文分词，中文 BM25 几乎无效"
        logger.warning(
            f"将删除并重建 collection={COLLECTION_NAME}，原因：{reason}。"
            "重建后请重新上传 PDF。"
        )
        client.drop_collection(COLLECTION_NAME)
        if SCHEMA_MARKER.exists():
            SCHEMA_MARKER.unlink()

    _create_collection(client)


def init_milvus() -> MilvusClient:
    """应用启动：创建 client，并确保 collection 就绪。"""
    client = get_milvus_client()
    ensure_collection(client)
    return client
