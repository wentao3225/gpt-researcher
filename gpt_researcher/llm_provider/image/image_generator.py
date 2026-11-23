"""GPT Researcher 的图片生成提供者。

此模块通过 google.genai SDK 使用 Google 的 Gemini/Imagen 模型
提供图片生成能力。

支持的模型：
- Gemini 图片模型（免费层）：models/gemini-2.5-flash-image
- Imagen 模型（需要计费）：imagen-4.0-generate-001
"""

import asyncio
import base64
import hashlib
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ImageGeneratorProvider:
    """使用 Google 的 Gemini/Imagen 模型生成图片的提供者。
    
    属性：
        model_name: 用于图片生成的模型。
        api_key: 用于认证的 Google API 密钥。
        output_dir: 保存生成图片的目录。
    """
    
    # Gemini 模型使用 generate_content 和 inline_data 响应
    GEMINI_IMAGE_MODELS = [
        "models/gemini-2.5-flash-image",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-exp-image-generation",
        "gemini-3-pro-image-preview",
    ]
    
    # Imagen 模型使用 generate_images（需要计费）
    IMAGEN_MODELS = [
        "imagen-4.0-generate-001",
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-ultra-generate-001",
    ]
    
    DEFAULT_MODEL = "models/gemini-2.5-flash-image"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "outputs",
    ):
        """初始化 ImageGeneratorProvider。
        
        参数：
            model_name: 要使用的模型。默认为 models/gemini-2.5-flash-image。
            api_key: Google API 密钥。未提供时从 GOOGLE_API_KEY 环境变量读取。
            output_dir: 输出的基础目录（图片将位于 output_dir/images/）。
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.output_dir = Path(output_dir)
        self._client = None
        
        # Determine model type
        self._is_imagen = any(m in self.model_name.lower() for m in ['imagen'])
        
        if not self.api_key:
            logger.warning(
                "No Google API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable to enable image generation."
            )
    
    def _ensure_client(self):
        """确保 Google GenAI 客户端已初始化。"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized image generation with model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "google-genai package is required for image generation. "
                    "Install with: pip install google-genai"
                )
            except Exception as e:
                logger.error(f"Failed to initialize image generation client: {e}")
                raise
    
    def _ensure_output_dir(self, research_id: str = "") -> Path:
        """确保输出目录存在并返回路径。"""
        # 使用与 PDF/DOCX 相同的结构：outputs/images/{research_id}/
        if research_id:
            output_path = self.output_dir / "images" / research_id
        else:
            output_path = self.output_dir / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _generate_image_filename(self, prompt: str, index: int = 0) -> str:
        """基于提示词哈希为图片生成唯一文件名。"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"img_{prompt_hash}_{index}.png"
    
    def _crop_to_landscape(self, image_bytes: bytes, target_ratio: float = 16/9) -> bytes:
        """将方形图片裁剪为横版格式（默认 16:9）。
        
        这确保图片能很好地适配文章/报告版式。
        
        参数：
            image_bytes: 原始图片字节。
            target_ratio: 目标宽高比（默认 16:9 ≈ 1.78）。
            
        返回：
            PNG 格式的裁剪后图片字节。
        """
        try:
            from PIL import Image
            import io
            
            # 打开图片
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            # 如果已是横版或更宽，原样返回
            if width / height >= target_ratio:
                return image_bytes
            
            # 计算横版裁剪的新尺寸
            # 保持完整宽度，减小高度
            new_height = int(width / target_ratio)
            
            # 垂直居中裁剪
            top = (height - new_height) // 2
            bottom = top + new_height
            
            # 裁剪图片
            cropped = img.crop((0, top, width, bottom))
            
            # 保存为字节
            output = io.BytesIO()
            cropped.save(output, format='PNG', optimize=True)
            output.seek(0)
            
            logger.info(f"Cropped image from {width}x{height} to {width}x{new_height} (landscape)")
            return output.getvalue()
            
        except ImportError:
            logger.warning("PIL not available for image cropping, returning original")
            return image_bytes
        except Exception as e:
            logger.warning(f"Failed to crop image to landscape: {e}")
            return image_bytes
    
    def _build_enhanced_prompt(self, prompt: str, context: str = "", style: str = "dark") -> str:
        """构建带明确样式指令的增强提示词。
        
        参数：
            prompt: 基础图片提示词。
            context: 来自研究的额外上下文。
            style: 图片风格 - "dark"（匹配应用主题）、"light" 或 "auto"。
            
        返回：
            带样式指令的增强提示词字符串。
        """
        # 特定风格的调色板
        if style == "dark":
            # Dark mode matching the GPT Researcher app theme
            style_instructions = """
STYLE REQUIREMENTS - DARK MODE THEME:
- Dark background (#0d1117 or similar deep charcoal/navy)
- Primary accent color: Teal/Cyan (#14b8a6, #0d9488)
- Secondary colors: Slate grays (#374151, #4b5563), subtle purple accents
- Glowing, neon-like effects for highlights and important elements
- Modern, tech-forward, futuristic aesthetic
- Clean lines with subtle gradients
- High contrast elements that pop against dark background
- Sleek, minimalist design with visual depth
- Icons and diagrams with luminous teal outlines
- Professional infographic style suitable for tech/research context"""
        elif style == "light":
            style_instructions = """
STYLE REQUIREMENTS - LIGHT MODE:
- Clean white or light gray background
- Primary colors: Deep blue (#1e40af), teal (#0d9488)
- Professional, corporate aesthetic
- Subtle shadows for depth
- High readability with dark text elements
- Modern flat design with occasional gradients"""
        else:
            style_instructions = """
STYLE REQUIREMENTS - PROFESSIONAL:
- Sophisticated color palette (teals, blues, grays)
- Clean, modern design
- High contrast for readability
- Professional infographic style"""

        styled_prompt = f"""Create a professional, high-quality illustration for a research report.

SUBJECT: {prompt}

{style_instructions}

TECHNICAL REQUIREMENTS:
- No text, labels, or watermarks in the image
- Clear visual hierarchy
- Well-balanced composition
- Suitable for both digital viewing and printing
- Vector-style or clean photorealistic rendering
- Resolution and detail appropriate for report embedding

AVOID:
- Cartoonish or childish styles
- Cluttered or busy designs  
- Bright white backgrounds (for dark mode)
- Low quality or pixelated elements
- Generic stock photo aesthetics"""

        if context:
            styled_prompt += f"\n\nRESEARCH CONTEXT: {context[:300]}"
        
        return styled_prompt
    
    async def generate_image(
        self,
        prompt: str,
        context: str = "",
        research_id: str = "",
        aspect_ratio: str = "1:1",
        num_images: int = 1,
        style: str = "dark",
    ) -> List[Dict[str, Any]]:
        """基于提示词和可选上下文生成图片。
        
        参数：
            prompt: 图片生成提示词。
            context: 用于提高图片相关性的额外上下文。
            research_id: 用于组织输出的研究 ID。
            aspect_ratio: 图片的宽高比（仅 Imagen）。
            num_images: 要生成的图片数量。
            style: 图片风格 - "dark"、"light" 或 "auto"。
            
        返回：
            包含图片信息（含绝对路径）的字典列表。
        """
        if not self.api_key:
            logger.warning("No API key configured for image generation")
            return []
        
        self._ensure_client()
        output_path = self._ensure_output_dir(research_id)
        
        # 构建带样式的增强提示词
        logger.info(f"Building image prompt with style: {style}")
        full_prompt = self._build_enhanced_prompt(prompt, context, style)
        logger.debug(f"Full prompt (first 500 chars): {full_prompt[:500]}")
        
        try:
            if self._is_imagen:
                return await self._generate_with_imagen(full_prompt, output_path, num_images, aspect_ratio, research_id)
            else:
                return await self._generate_with_gemini(full_prompt, output_path, num_images, research_id, prompt)
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return []
    
    async def _generate_with_gemini(
        self,
        full_prompt: str,
        output_path: Path,
        num_images: int,
        research_id: str,
        original_prompt: str,
    ) -> List[Dict[str, Any]]:
        """使用 Gemini 模型通过 generate_content 生成图片。"""
        generated_images = []
        
        for i in range(num_images):
            try:
                # Gemini 图片模型使用 generate_content
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model_name,
                    contents=full_prompt,
                )
                
                # 调试：记录响应结构
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        logger.debug(f"Response has {len(candidate.content.parts)} parts")
                        for idx, part in enumerate(candidate.content.parts):
                            has_inline = hasattr(part, 'inline_data') and part.inline_data
                            has_text = hasattr(part, 'text') and part.text
                            logger.debug(f"Part {idx}: inline_data={has_inline}, text={has_text}")
                
                # 从响应部分提取图片
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # 找到图片数据
                            image_data = part.inline_data.data
                            mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                            
                            # 确定文件扩展名
                            ext = 'png' if 'png' in mime_type else 'jpg'
                            filename = self._generate_image_filename(original_prompt, i)
                            filepath = output_path / filename
                            
                            # 写入图片数据（可能是 base64 编码）
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data
                            
                            # 注意：保留 Gemini 的原始方形格式
                            # 要启用横版裁剪，取消注释：
                            # image_bytes = self._crop_to_landscape(image_bytes)
                            
                            with open(filepath, 'wb') as f:
                                f.write(image_bytes)
                            
                            # 同时使用绝对路径（供 PDF）和 Web URL（供前端）
                            absolute_path = filepath.resolve()
                            web_url = f"/outputs/images/{research_id}/{filename}" if research_id else f"/outputs/images/{filename}"
                            
                            generated_images.append({
                                "path": str(absolute_path),  # 供 PDF 生成的绝对路径
                                "url": web_url,  # 供前端显示的 Web URL
                                "absolute_url": str(absolute_path),  # 供 PDF 兼容
                                "prompt": original_prompt,
                                "alt_text": self._generate_alt_text(original_prompt),
                            })
                            
                            logger.info(f"Generated image saved to: {filepath}")
                            break  # 每次迭代只取第一张图片
                    else:
                        # 未找到 inline_data - 检查是否有文本（模型拒绝）
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                logger.warning(f"Model returned text instead of image: {part.text[:200]}")
                                break
                            
            except Exception as e:
                logger.error(f"Error generating image {i}: {e}", exc_info=True)
                continue
        
        return generated_images
    
    async def _generate_with_imagen(
        self,
        full_prompt: str,
        output_path: Path,
        num_images: int,
        aspect_ratio: str,
        research_id: str,
    ) -> List[Dict[str, Any]]:
        """使用 Imagen 模型通过 generate_images 生成图片。"""
        from google.genai import types
        
        generated_images = []
        
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_images,
                model=self.model_name,
                prompt=full_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=num_images,
                    aspect_ratio=aspect_ratio,
                ),
            )
            
            if response and response.generated_images:
                for i, gen_image in enumerate(response.generated_images):
                    filename = self._generate_image_filename(full_prompt, i)
                    filepath = output_path / filename
                    
                    # 提取图片字节
                    if hasattr(gen_image, 'image') and hasattr(gen_image.image, 'image_bytes'):
                        image_bytes = gen_image.image.image_bytes
                    elif hasattr(gen_image, 'image_bytes'):
                        image_bytes = gen_image.image_bytes
                    else:
                        logger.warning("Could not extract image bytes")
                        continue
                    
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    
                    # 同时使用绝对路径（供 PDF）和 Web URL（供前端）
                    absolute_path = filepath.resolve()
                    web_url = f"/outputs/images/{research_id}/{filename}" if research_id else f"/outputs/images/{filename}"
                    
                    generated_images.append({
                        "path": str(absolute_path),
                        "url": web_url,
                        "absolute_url": str(absolute_path),
                        "prompt": full_prompt,
                        "alt_text": self._generate_alt_text(full_prompt),
                    })
                    
                    logger.info(f"Generated image saved to: {filepath}")
                    
        except Exception as e:
            logger.error(f"Imagen generation failed: {e}", exc_info=True)
        
        return generated_images
    
    def _generate_alt_text(self, prompt: str) -> str:
        """从提示词生成无障碍替代文本。"""
        # 清理并截断以用于替代文本
        clean_prompt = prompt.replace('\n', ' ').strip()
        # 仅提取核心主题
        if len(clean_prompt) > 120:
            clean_prompt = clean_prompt[:117] + "..."
        return f"Illustration: {clean_prompt}"
    
    def is_available(self) -> bool:
        """检查图片生成是否可用。"""
        if not self.api_key:
            return False
        try:
            self._ensure_client()
            return True
        except Exception as e:
            logger.warning(f"Image generation not available: {e}")
            return False
    
    @classmethod
    def from_config(cls, config) -> Optional["ImageGeneratorProvider"]:
        """从 Config 对象创建 ImageGeneratorProvider。"""
        model = getattr(config, 'image_generation_model', None)
        enabled = getattr(config, 'image_generation_enabled', False)
        
        if not enabled:
            return None
        
        return cls(model_name=model or cls.DEFAULT_MODEL)
