import asyncio
import aiohttp
import json
import os
import re
import sys
from typing import Optional
from pathlib import Path
from loguru import logger

# 支持直接运行本文件：python app/core/ocr_tools.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.schemas.document import (
    OcrBlock,
    OcrPage,
    OcrParseResult,
    DocumentContentTree,
)

TOKEN = os.getenv("PADDLE_TOKEN")
OCR_RESULT_PATH = Path(__file__).resolve().parent.parent.parent / "result"
OCR_RESULT_PATH.mkdir(parents=True, exist_ok=True)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 中文大纲：一、二、 → 二级；(一)(二) → 三级
CN_MAJOR_RE = re.compile(r"^[一二三四五六七八九十百千]+、|^附件|^参考文献")
CN_MINOR_RE = re.compile(r"^[（(][一二三四五六七八九十百千]+[）)]")

"""
    目前的解析结果，完全依据paddleocr提取出的结果，后续需要添加强校验，针对识别错误的结果进行修正
    针对不同文档需要针对处理
"""


class PaddleOcrTool:
    """
    百度 AI Studio PaddleOCR 远程服务,单次调用最好不要超过100页，避免出现超时的情况
    """
    
    def __init__(
        self,
        job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        token: str = TOKEN,
        model: str = "PaddleOCR-VL-1.6",
        document_id: str = None,
        document_name: str = None,
        poll_interval: int = 5,
        timeout: int = 300
    ):
        self.job_url = job_url
        self.token = token
        self.model = model
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.headers = {"Authorization": f"bearer {self.token}"}
        self.ocr_result_path = OCR_RESULT_PATH
        self.document_id = document_id
        self.document_name = document_name

    async def parse(self, file_path: str, options: Optional[dict] = None) -> OcrParseResult:
        """
        主入口：解析文件，返回结构化结果
        """
        job_id = await self._submit_job(file_path, options)
        result_url = await self._poll_job(job_id)
        return await self._download_result(result_url)

    async def _submit_job(self, file_path: str, options: Optional[dict]) -> str:
        """提交 OCR 任务"""
        payload_options = options or {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
            "formatBlockContent": True,
        }

        async with aiohttp.ClientSession() as session:
            if file_path.startswith("http"):
                headers = {**self.headers, "Content-Type": "application/json"}
                payload = {
                    "fileUrl": file_path,
                    "model": self.model,
                    "optionalPayload": payload_options
                }
                async with session.post(self.job_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            else:
                data_payload = {
                    "model": self.model,
                    "optionalPayload": json.dumps(payload_options)
                }
                with open(file_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("file", f, filename=os.path.basename(file_path))
                    for key, val in data_payload.items():
                        form.add_field(key, val)

                    async with session.post(self.job_url, headers=self.headers, data=form) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

        return data["data"]["jobId"]

    async def _poll_job(self, job_id: str) -> str:
        """轮询任务状态，直到完成"""
        start_time = asyncio.get_event_loop().time()
        poll_url = f"{self.job_url}/{job_id}"

        async with aiohttp.ClientSession() as session:
            while True:
                if asyncio.get_event_loop().time() - start_time > self.timeout:
                    raise TimeoutError(f"OCR job {job_id} timeout")

                async with session.get(poll_url, headers=self.headers) as resp:
                    resp.raise_for_status()
                    job_data = (await resp.json())["data"]
                    state = job_data["state"]

                    if state == "done":
                        return job_data["resultUrl"]["jsonUrl"]
                    elif state == "failed":
                        raise Exception(f"OCR failed: {job_data.get('errorMsg', 'unknown')}")

                await asyncio.sleep(self.poll_interval)

    async def _download_result(self, json_url: str) -> OcrParseResult:
        """下载并解析结果"""
        async with aiohttp.ClientSession() as session:
            async with session.get(json_url) as resp:
                text = await resp.text()

        # 接口是 JSONL（多行），不能 json.loads(整段 text)，要按行解析后合并
        results = [json.loads(line)["result"] for line in text.splitlines() if line.strip()]
        pages = [
            OcrPage(
                markdown=layout["markdown"]["text"],
                images=layout["markdown"].get("images", {}),
            )
            for result in results
            for layout in result.get("layoutParsingResults", [])
        ]
        merged = {
            "layoutParsingResults": [
                layout for result in results for layout in result.get("layoutParsingResults", [])
            ],
            "dataInfo": next(
                (r.get("dataInfo") for r in reversed(results) if r.get("dataInfo") is not None),
                None,
            ),
            "preprocessedImages": [
                img for result in results for img in (result.get("preprocessedImages") or [])
            ],
        }

        out_path = self.ocr_result_path / f"{self.document_name}.json"
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(merged, file, ensure_ascii=False, indent=2)
        logger.info(f"OCR 结果已保存: {out_path}, pages={len(merged['layoutParsingResults'])}")

        extracted_result = self._extract_result(merged)
        content_tree = self._blocks_to_tree(extracted_result)

        tree_path = self.ocr_result_path / f"{self.document_name}_content_tree.json"
        with open(tree_path, "w", encoding="utf-8") as file:
            json.dump(content_tree.to_dict(), file, ensure_ascii=False, indent=2)
        logger.info(f"内容树已保存: {tree_path}")

        return OcrParseResult(
            total_pages=len(pages),
            pages=pages,
            result=merged,
            extracted_result=extracted_result,
            content_tree=content_tree,
        )

    def _extract_result(self, result: dict) -> list[OcrBlock]:
        extracted_result: list[OcrBlock] = []
        IGNORE = {"number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"}

        for page_idx, page in enumerate(result.get("layoutParsingResults", [])):
            page_meta = page["prunedResult"]
            for block in page_meta["parsing_res_list"]:
                if block["block_label"] in IGNORE:
                    continue
                extracted_result.append(
                    OcrBlock(
                        page_idx=page_idx,
                        block_label=block["block_label"],
                        block_content=block["block_content"],
                        block_bbox=block["block_bbox"],
                    )
                )
        return extracted_result

    def _heading_level(self, block: OcrBlock) -> int | None:
        """解析标题层级。

        优先看 markdown # 个数，再按中文大纲修正：
        - 一、/二、/附件 → level 2
        - （一）/（二） → level 3（挂到上一节「四、xxx」下）
        """
        text = (block.block_content or "").strip()
        m = HEADING_RE.match(text)
        title = m.group(2).strip() if m else text

        # 中文编号优先于单纯的 ##（因为 （一） 也被标成 ##）
        if CN_MINOR_RE.match(title):
            return 3
        if CN_MAJOR_RE.match(title):
            return 2

        if m:
            return len(m.group(1))
        if block.block_label == "doc_title":
            return 1
        if block.block_label == "paragraph_title":
            return 2
        return None

    def _blocks_to_tree(self, blocks: list[OcrBlock]) -> DocumentContentTree:
        """按 markdown 标题层级将文本块构造成树。"""
        root = DocumentContentTree(title="ROOT", level=0)
        stack = [root]

        for block in blocks:
            level = self._heading_level(block)
            if level is None:
                stack[-1].content.append(block)
                continue

            text = (block.block_content or "").strip()
            m = HEADING_RE.match(text)
            title = m.group(2).strip() if m else text
            node = DocumentContentTree(
                title=title,
                level=level,
                page_idx=block.page_idx,
                bbox=block.block_bbox,
                block=block,
            )
            while stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)

        return root

if __name__ == "__main__":
    async def main():
        ocr_service = PaddleOcrTool(document_name="隐睾症")
        # 用已有 json 验证（无需重新跑 OCR）
        merged = json.loads((OCR_RESULT_PATH / "隐睾症.json").read_text(encoding="utf-8"))
        extracted = ocr_service._extract_result(merged)
        tree = ocr_service._blocks_to_tree(extracted)

        # 1) 完整树
        tree_path = OCR_RESULT_PATH / "隐睾症_content_tree.json"
        tree_path.write_text(
            json.dumps(tree.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2) 仅保留二级标题，下级合并成大段（供总结）
        sections = tree.to_summary_sections(keep_level=2)
        sections_path = OCR_RESULT_PATH / "隐睾症_summary_sections.json"
        sections_path.write_text(
            json.dumps([s.to_dict() for s in sections], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for s in sections:
            print(f"[L{s.level}] {s.title} | pages={s.page_idxs} | chars={len(s.text)}")
        print(f"content_tree: {tree_path}")
        print(f"summary_sections: {sections_path}")

    asyncio.run(main())
