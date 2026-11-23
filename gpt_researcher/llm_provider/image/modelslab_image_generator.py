"""GPT Researcher 的 ModelsLab 图片生成提供者。

此模块通过 ModelsLab 的 API 提供图片生成，支持
Flux、SDXL、Stable Diffusion 以及 5 万多个社区模型。

API 文档：https://docs.modelslab.com
"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TEXT2IMG_URL = "https://modelslab.com/api/v6/images/text2img"
FETCH_BASE_URL = "https://modelslab.com/api/v6/images/fetch"
MAX_POLL_ATTEMPTS = 12
POLL_INTERVAL_SECONDS = 5


class ModelsLabImageGeneratorProvider:
    """使用 ModelsLab 的 API 生成图片的提供者。

    支持 Flux、SDXL、Stable Diffusion 和数千个社区模型。
    认证使用请求体中的 API 密钥（而非 Bearer 令牌）。

    属性：
        model_id: 要使用的 ModelsLab 模型（默认："flux"）。
        api_key: 来自 modelslab.com/account/api-key 的 ModelsLab API 密钥。
        output_dir: 保存生成图片的目录。
    """

    DEFAULT_MODEL = "flux"

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "outputs",
    ):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("MODELSLAB_API_KEY")
        self.output_dir = Path(output_dir)

        if not self.api_key:
            logger.warning(
                "No ModelsLab API key found. Set MODELSLAB_API_KEY "
                "environment variable to enable image generation."
            )

    def _ensure_output_dir(self, research_id: str = "") -> Path:
        path = (
            self.output_dir / "images" / research_id
            if research_id
            else self.output_dir / "images"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _generate_filename(self, prompt: str, index: int = 0) -> str:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"img_{prompt_hash}_{index}.png"

    async def _download_image(self, url: str) -> bytes:
        """异步从 URL 下载图片。"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except ImportError:
            # aiohttp 不可用时在线程中回退到 requests
            import requests

            response = await asyncio.to_thread(requests.get, url, timeout=30)
            response.raise_for_status()
            return response.content

    async def _poll_for_result(self, request_id: str) -> List[str]:
        """轮询 fetch 端点直到生成完成。"""
        try:
            import aiohttp

            for _ in range(MAX_POLL_ATTEMPTS):
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{FETCH_BASE_URL}/{request_id}",
                        json={"key": self.api_key},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        body = await resp.json()
                        if body.get("status") == "success" and body.get("output"):
                            return body["output"]
                        if body.get("status") == "error":
                            raise RuntimeError(
                                body.get("message") or body.get("messege") or "ModelsLab generation error"
                            )
        except ImportError:
            import requests

            for _ in range(MAX_POLL_ATTEMPTS):
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                resp = await asyncio.to_thread(
                    requests.post,
                    f"{FETCH_BASE_URL}/{request_id}",
                    json={"key": self.api_key},
                    timeout=15,
                )
                body = resp.json()
                if body.get("status") == "success" and body.get("output"):
                    return body["output"]
                if body.get("status") == "error":
                    raise RuntimeError(body.get("message") or body.get("messege") or "ModelsLab generation error")

        raise TimeoutError("ModelsLab image generation timed out after polling.")

    async def generate_image(
        self,
        prompt: str,
        context: str = "",
        research_id: str = "",
        aspect_ratio: str = "1:1",
        num_images: int = 1,
        style: str = "dark",
    ) -> List[Dict[str, Any]]:
        """使用 ModelsLab 的文生图 API 生成图片。

        参数：
            prompt: 图片生成提示词。
            context: 额外上下文（追加到提示词）。
            research_id: 用于组织输出目录的研究 ID。
            aspect_ratio: 忽略（ModelsLab 使用宽/高）。
            num_images: 要生成的图片数量（1–4）。
            style: 忽略（提示词直接控制风格）。

        返回：
            含 path、url、prompt 和 alt_text 键的字典列表。
        """
        if not self.api_key:
            logger.warning("No ModelsLab API key set; skipping image generation.")
            return []

        output_path = self._ensure_output_dir(research_id)
        full_prompt = f"{prompt}. {context}" if context else prompt
        samples = min(max(1, num_images), 4)

        payload = {
            "key": self.api_key,
            "model_id": self.model_id,
            "prompt": full_prompt,
            "negative_prompt": "low quality, blurry, watermark, text, nsfw",
            "width": 768,
            "height": 512,
            "samples": samples,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": -1,
            "safety_checker": "yes",
        }

        try:
            image_urls = await self._request_images(payload)
        except Exception as exc:
            logger.error(f"ModelsLab image generation failed: {exc}", exc_info=True)
            return []

        results = []
        for i, url in enumerate(image_urls[:num_images]):
            try:
                image_bytes = await self._download_image(url)
                filename = self._generate_filename(prompt, i)
                filepath = output_path / filename
                with open(filepath, "wb") as fh:
                    fh.write(image_bytes)

                absolute_path = filepath.resolve()
                web_url = (
                    f"/outputs/images/{research_id}/{filename}"
                    if research_id
                    else f"/outputs/images/{filename}"
                )
                results.append(
                    {
                        "path": str(absolute_path),
                        "url": web_url,
                        "absolute_url": str(absolute_path),
                        "prompt": prompt,
                        "alt_text": f"Illustration: {prompt[:120]}",
                    }
                )
                logger.info(f"ModelsLab image saved to: {filepath}")
            except Exception as exc:
                logger.error(f"Failed to download image {i}: {exc}")

        return results

    async def _request_images(self, payload: Dict[str, Any]) -> List[str]:
        """POST 到 text2img 端点并处理异步处理。"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    TEXT2IMG_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.json()
        except ImportError:
            import requests

            body = await asyncio.to_thread(
                lambda: requests.post(TEXT2IMG_URL, json=payload, timeout=30).json()
            )

        if body.get("status") == "error":
            raise RuntimeError(body.get("message") or body.get("messege") or "ModelsLab API error")

        if body.get("status") == "processing" and body.get("id"):
            return await self._poll_for_result(body["id"])

        return body.get("output", [])

    def is_available(self) -> bool:
        """如果已配置 API 密钥则返回 True。"""
        return bool(self.api_key)

    @classmethod
    def from_config(cls, config) -> Optional["ModelsLabImageGeneratorProvider"]:
        """从 Config 对象创建 ModelsLabImageGeneratorProvider。"""
        enabled = getattr(config, "image_generation_enabled", False)
        provider = getattr(config, "image_generation_provider", "google")
        if not enabled or provider != "modelslab":
            return None
        model = getattr(config, "image_generation_model", None)
        return cls(model_id=model or cls.DEFAULT_MODEL)
