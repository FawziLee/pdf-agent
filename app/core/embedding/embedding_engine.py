import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

import numpy as np
from loguru import logger
from openai import AsyncOpenAI, OpenAI


class EmbeddingEngine(ABC):
    """Embedding 模型引擎：统一接口，便于切换不同厂商/模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def single_embed(self, text: str) -> np.ndarray:
        """单条文本 → 向量 shape: (dim,)"""

    @abstractmethod
    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """多条文本 → 向量矩阵 shape: (n, dim)"""

    @abstractmethod
    async def async_single_embed(self, text: str) -> np.ndarray:
        """异步：单条文本 → 向量"""

    @abstractmethod
    async def async_batch_embed(self, texts: List[str]) -> np.ndarray:
        """异步：多条文本 → 向量矩阵"""


class QwenEmbeddingEngine(EmbeddingEngine):
    """阿里云百炼 / DashScope OpenAI 兼容 Embedding。"""

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 5,
    ):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        self.base_url = base_url or os.getenv(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.dimensions = dimensions
        # 接口单次条数上限（如 qwen 常见 5～10）
        self.batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", str(batch_size))))
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(
            f"初始化 Embedding 引擎：{model_name}, base_url={self.base_url}, batch_size={self.batch_size}"
        )

    def _build_kwargs(self, inputs: List[str]) -> dict:
        kwargs = {
            "model": self.model_name,
            "input": inputs if len(inputs) > 1 else inputs[0],
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    @staticmethod
    def _to_array(completion) -> np.ndarray:
        vectors = [item.embedding for item in completion.data]
        return np.asarray(vectors, dtype=np.float32)

    def _create(self, inputs: List[str]) -> np.ndarray:
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)
        completion = self.client.embeddings.create(**self._build_kwargs(inputs))
        return self._to_array(completion)

    async def _acreate(self, inputs: List[str]) -> np.ndarray:
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)
        completion = await self.async_client.embeddings.create(**self._build_kwargs(inputs))
        return self._to_array(completion)

    def single_embed(self, text: str) -> np.ndarray:
        return self._create([text])[0]

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """按 batch_size 切分多次请求，再拼成完整矩阵。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            chunks.append(self._create(batch))
            logger.debug(f"embedding 进度：{min(i + self.batch_size, total)}/{total}")
        return np.vstack(chunks)

    async def async_single_embed(self, text: str) -> np.ndarray:
        arr = await self._acreate([text])
        return arr[0]

    async def async_batch_embed(self, texts: List[str]) -> np.ndarray:
        """按 batch_size 切分多次请求，再拼成完整矩阵（保证全部向量化）。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            chunks.append(await self._acreate(batch))
            logger.info(f"embedding 进度：{min(i + self.batch_size, total)}/{total}")
        return np.vstack(chunks)


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    """全局单例，进程内只创建一次。"""
    return QwenEmbeddingEngine(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "qwen3.7-text-embedding"),
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        engine = get_embedding_engine()
        vec = await engine.async_single_embed("衣服的质量杠杠的")
        print("dim:", vec.shape, "sample:", vec[:8])
        mat = await engine.async_batch_embed(["你好", "向量检索"])
        print("batch:", mat.shape)

    asyncio.run(main())
