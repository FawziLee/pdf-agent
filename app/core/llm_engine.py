import os
import time
from typing import List, Dict, Any, AsyncGenerator, Generator
from abc import ABC, abstractmethod
from loguru import logger
from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError


class LLMEngine(ABC):
    """LLM引擎基类, 定义统一接口"""

    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = 0.3
        self.top_p = 0.1
        self.max_tokens = 32000
        self.timeout = 60
        self.max_retries = 3
        logger.info(f"初始化LLM引擎：{model_name}")

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """同步聊天接口"""
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式聊天接口"""
        pass

    @abstractmethod
    async def async_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """异步聊天接口"""
        pass

    @abstractmethod
    async def async_stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天接口"""
        pass


class OpenAIChatEngine(LLMEngine):
    """OpenAI Chat API 接口引擎"""

    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        super().__init__(model_name, api_key, base_url)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        logger.info(f"OpenAI Chat API 接口引擎初始化完成，Base URL：{base_url}")

    def _build_params(self, **kwargs) -> Dict[str, Any]:
        """构建 Responses API 请求参数"""
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("stream"):
            params["stream"] = True
        return params

    def _split_messages(self, messages: List[Dict[str, str]]) -> tuple[str | None, str | List[Dict[str, str]]]:
        """将 messages 转为 Responses API 的 instructions + input"""
        instructions = None
        conversation: List[Dict[str, str]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                instructions = content if instructions is None else f"{instructions}\n{content}"
            else:
                conversation.append({"role": role, "content": content})

        if len(conversation) == 1 and conversation[0]["role"] == "user":
            input_data: str | List[Dict[str, str]] = conversation[0]["content"]
        else:
            input_data = conversation

        return instructions, input_data

    def _build_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        instructions, input_data = self._split_messages(messages)
        request = {
            "input": input_data,
            **self._build_params(**kwargs),
        }
        if instructions:
            request["instructions"] = instructions
        return request

    def _format_response(self, response: Any, latency: float) -> Dict[str, Any]:
        usage = response.usage
        return {
            "content": response.output_text,
            "finish_reason": response.status,
            "usage": {
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "latency": round(latency, 3),
        }

    def _iter_stream_text(self, stream: Any) -> Generator[str, None, None]:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta

    async def _aiter_stream_text(self, stream: Any) -> AsyncGenerator[str, None]:
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """同步非流式聊天"""
        try:
            request = self._build_request(messages, **kwargs, stream=False)
            start_time = time.time()
            response = self.client.responses.create(**request)
            result = self._format_response(response, time.time() - start_time)
            logger.info(
                f"LLM调用完成，耗时：{result['latency']}s，总token：{result['usage']['total_tokens']}"
            )
            return result

        except (APIError, APIConnectionError, RateLimitError, APITimeoutError) as e:
            logger.error(f"LLM调用失败：{type(e).__name__}，错误信息：{str(e)}")
            raise Exception(f"大模型调用失败：{str(e)}")
        except Exception as e:
            logger.error(f"LLM调用未知错误：{str(e)}")
            raise Exception(f"大模型调用异常：{str(e)}")

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """同步流式聊天"""
        try:
            request = self._build_request(messages, **kwargs, stream=True)
            logger.info("开始流式LLM调用")
            stream = self.client.responses.create(**request)
            yield from self._iter_stream_text(stream)

        except Exception as e:
            logger.error(f"流式LLM调用失败：{str(e)}")
            yield f"[ERROR] 大模型调用失败：{str(e)}"

    async def async_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """异步非流式聊天"""
        try:
            request = self._build_request(messages, **kwargs, stream=False)
            start_time = time.time()
            response = await self.async_client.responses.create(**request)
            result = self._format_response(response, time.time() - start_time)
            logger.info(
                f"异步LLM调用完成，耗时：{result['latency']}s，总token：{result['usage']['total_tokens']}"
            )
            return result

        except Exception as e:
            logger.error(f"异步LLM调用失败：{str(e)}")
            raise Exception(f"大模型调用失败：{str(e)}")

    async def async_stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天"""
        try:
            request = self._build_request(messages, **kwargs, stream=True)
            logger.info("开始异步流式LLM调用")
            stream = await self.async_client.responses.create(**request)
            async for chunk in self._aiter_stream_text(stream):
                yield chunk

        except Exception as e:
            logger.error(f"异步流式LLM调用失败：{str(e)}")
            yield f"[ERROR] 大模型调用失败：{str(e)}"


def get_llm_engine() -> LLMEngine:
    """
    获取LLM引擎实例
    """
    return OpenAIChatEngine(
        model_name=os.getenv("LLM_MODEL_NAME", "qwen-plus"),
        api_key=os.getenv("QWEN_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )


if __name__ == "__main__":
    engine = get_llm_engine()
    test_messages = [
        {"role": "user", "content": "你好，介绍一下RAG系统"},
    ]
    result = engine.chat(test_messages)
    print("同步调用结果：", result["content"])
    print("Token消耗：", result["usage"])

    print("\n流式调用结果：")
    for content in engine.stream_chat(test_messages):
        print(content, end="", flush=True)
