import json
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.agent import PdfAgent, get_pdf_agent

agent_router = APIRouter(prefix="/agents")


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    document_id: str | None = Field(None, description="限定单个文档；不传则租户下检索")
    top_k: int = Field(5, ge=1, le=20)
    use_bm25: bool = Field(True, description="是否混合 BM25")


@agent_router.post("/chat", summary="PDF Agent 问答（需先上传入库）")
async def chat_with_agent(
    body: Annotated[ChatRequest, Body(...)],
    tenant_id: str = Query("demo-tenant", description="租户ID"),
    pdf_agent: Annotated[PdfAgent, Depends(get_pdf_agent)] = None,
):
    document_ids = [body.document_id] if body.document_id else None

    async def event_stream():
        try:
            async for item in pdf_agent.stream_ask(
                question=body.question,
                tenant_id=tenant_id,
                document_ids=document_ids,
                top_k=body.top_k,
                use_bm25=body.use_bm25,
            ):
                event = item["event"]
                data = json.dumps(
                    item["data"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                yield f"event: {event}\ndata: {data}\n\n"
        except Exception as exc:
            data = json.dumps(
                {"message": str(exc)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
