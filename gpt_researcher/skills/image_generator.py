"""GPT Researcher 的图片生成技能。

此模块提供 ImageGenerator 类，处理使用 AI 图片生成
为研究报告生成上下文相关的图片。
"""

import asyncio
import json
import logging
import re

import json_repair
from typing import Any, Dict, List, Optional, Tuple

from ..actions.utils import stream_output
from ..llm_provider.image import ImageGeneratorProvider, ModelsLabImageGeneratorProvider
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


class ImageGenerator:
    """为研究报告生成上下文相关的图片。
    
    此类分析报告内容，识别能从视觉插图中
    受益的章节，并使用 AI 图片生成模型生成图片。
    
    属性：
        researcher: 上级 GPTResearcher 实例。
        image_provider: 图片生成提供者。
        max_images: 每份报告可生成的最大图片数量。
    """
    
    def __init__(self, researcher):
        """初始化 ImageGenerator。
        
        参数：
            researcher: 拥有此生成器的 GPTResearcher 实例。
        """
        self.researcher = researcher
        self.cfg = researcher.cfg
        self.image_provider = None
        self.max_images = getattr(self.cfg, 'image_generation_max_images', 3)
        self.generated_images: List[Dict[str, Any]] = []
        
        # 如果已配置则初始化图片提供者
        self._init_provider()
    
    def _init_provider(self):
        """从配置初始化图片生成提供者。"""
        try:
            # Config._set_attributes() 以 key.lower() 存储每个键，因此
            # 大写名称永远不会解析，`enabled` 以前总是默认 False
            # —— 功能无法打开。与此类中 max_images / style
            # 已使用的读取方式一致（小写）。
            enabled = getattr(self.cfg, 'image_generation_enabled', False)
            if not enabled:
                return
            provider_name = getattr(self.cfg, 'image_generation_provider', 'google')
            model = getattr(self.cfg, 'image_generation_model', None)
            if provider_name == 'modelslab':
                provider = ModelsLabImageGeneratorProvider(model_id=model)
            else:
                provider = ImageGeneratorProvider(model_name=model)
            if provider.is_available():
                self.image_provider = provider
                logger.info(f"Image generation provider initialized: {provider_name}")
            else:
                logger.warning(f"Image generation provider '{provider_name}' not available (missing API key?)")
        except Exception as e:
            logger.error(f"Failed to initialize image provider: {e}")
            self.image_provider = None

    def is_enabled(self) -> bool:
        """检查图片生成是否已启用且可用。
        
        返回：
            如果可以使用图片生成则返回 True。
        """
        return self.image_provider is not None and self.image_provider.is_available()

    async def plan_and_generate_images(
        self,
        context: str,
        query: str,
        research_id: str = "",
    ) -> List[Dict[str, Any]]:
        """基于研究上下文规划并预生成图片。
        
        此方法分析研究上下文，识别 2-3 个能从视觉插图中
        受益的概念，然后在撰写报告开始之前并行生成它们。
        
        参数：
            context: 累积的研究上下文。
            query: 主要的研究查询。
            research_id: 可选的研究 ID，用于文件组织。
            
        返回：
            包含 URL、可嵌入的生成图片信息字典列表。
        """
        if not self.is_enabled():
            logger.info("Image generation is not enabled, skipping pre-generation")
            return []
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "image_planning",
                "🎨 Analyzing research context for visualization opportunities...",
                self.researcher.websocket,
            )
        
        # 步骤 1：使用 LLM 识别最佳可视化机会
        image_concepts = await self._plan_image_concepts(context, query)
        
        if not image_concepts:
            logger.info("No suitable visualization opportunities identified")
            return []
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "image_concepts_identified",
                f"🖼️ Identified {len(image_concepts)} visualization concepts, generating images...",
                self.researcher.websocket,
            )
        
        # 步骤 2：并行生成所有图片
        image_style = getattr(self.cfg, 'image_generation_style', 'dark')
        
        async def generate_single_image(concept: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
            """从概念生成单张图片。"""
            try:
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "image_generating",
                        f"🖼️ Generating image {index + 1}/{len(image_concepts)}: {concept['title'][:50]}...",
                        self.researcher.websocket,
                    )
                
                images = await self.image_provider.generate_image(
                    prompt=concept['prompt'],
                    context=concept.get('context', ''),
                    research_id=research_id,
                    num_images=1,
                    style=image_style,
                )
                
                if images:
                    image_info = images[0]
                    image_info['title'] = concept['title']
                    image_info['section_hint'] = concept.get('section_hint', '')
                    return image_info
                    
            except Exception as e:
                logger.error(f"Failed to generate image for '{concept['title']}': {e}")
            return None
        
        # 并行生成所有图片
        tasks = [generate_single_image(concept, i) for i, concept in enumerate(image_concepts)]
        results = await asyncio.gather(*tasks)
        
        # 过滤生成失败项
        generated_images = [img for img in results if img is not None]
        self.generated_images = generated_images
        
        if self.researcher.verbose:
            if generated_images:
                await stream_output(
                    "logs",
                    "images_ready",
                    f"✅ {len(generated_images)} images ready for report embedding",
                    self.researcher.websocket,
                )
            else:
                await stream_output(
                    "logs",
                    "images_failed",
                    "⚠️ No images could be generated",
                    self.researcher.websocket,
                )
        
        return generated_images

    async def _plan_image_concepts(
        self,
        context: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """使用 LLM 从研究上下文中识别最佳可视化机会。
        
        参数：
            context: 要分析的研究上下文。
            query: 主要的研究查询。
            
        返回：
            带 title、prompt 和 section_hint 的图片概念字典列表。
        """
        # 上下文过长则截断
        max_context_length = 6000
        truncated_context = context[:max_context_length] if len(context) > max_context_length else context
        
        planning_prompt = f"""Analyze this research context and identify 2-3 concepts that would significantly benefit from professional diagram/infographic illustrations.

RESEARCH QUERY: {query}

RESEARCH CONTEXT:
{truncated_context}

For each visualization opportunity, provide:
1. title: A short descriptive title (e.g., "System Architecture", "Comparison Chart")
2. prompt: A detailed image generation prompt describing exactly what to visualize, including layout and key elements (minimum 30 words)
3. section_hint: Which section of the report this image relates to

Focus on:
- Architecture/system diagrams
- Process flows and workflows
- Comparison charts
- Data visualizations
- Conceptual illustrations

IMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation.

Example output:
[
  {{
    "title": "System Architecture Overview",
    "prompt": "A layered architecture diagram showing the frontend application on top, connecting to an API gateway in the middle, which routes to microservices at the bottom. Use clean boxes with connecting arrows, modern tech aesthetic.",
    "section_hint": "Architecture"
  }}
]

Return 2-3 visualization concepts as a JSON array:"""

        try:
            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[
                    {"role": "system", "content": "You are a visualization expert. Return only valid JSON arrays."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.4,
                llm_provider=self.cfg.fast_llm_provider,
                max_tokens=4000,  # 为推理模型的推理 token 预留余量
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs,
            )
            
            # 解析 JSON 响应（模型经常给 JSON 加围栏/前言；与 deep_research 一致）
            concepts = json_repair.loads(response)
            if not isinstance(concepts, list):
                logger.error("Image planning response was not a JSON array")
                return []

            # 校验并限制到 max_images
            valid_concepts = []
            for concept in concepts[:self.max_images]:
                if isinstance(concept, dict) and 'title' in concept and 'prompt' in concept:
                    valid_concepts.append(concept)
            
            logger.info(f"Planned {len(valid_concepts)} image concepts")
            return valid_concepts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse image planning response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during image planning: {e}")
            return []

    async def analyze_report_for_images(
        self,
        report: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """分析报告，识别能从图片中受益的章节。
        
        使用 LLM 识别报告中 2-3 个将从视觉插图中
        受益的关键概念或章节。
        
        参数：
            report: Markdown 报告文本。
            query: 原始研究查询。
            
        返回：
            带章节信息和建议图片提示词的字典列表。
        """
        if not self.is_enabled():
            return []
        
        # 从报告中提取章节
        sections = self._extract_sections(report)
        
        if not sections:
            logger.warning("No sections found in report for image analysis")
            return []
        
        # 使用 LLM 识别最适合配图的章节
        try:
            analysis_prompt = self._build_analysis_prompt(query, sections)
            
            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying content that would benefit from visual illustrations."},
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.3,
                llm_provider=self.cfg.fast_llm_provider,
                stream=False,
                websocket=None,
                max_tokens=4000,  # 为推理模型的推理 token 预留余量
                llm_kwargs=self.cfg.llm_kwargs,
            )
            
            # 解析响应
            image_suggestions = self._parse_analysis_response(response, sections)
            return image_suggestions[:self.max_images]
            
        except Exception as e:
            logger.error(f"Error analyzing report for images: {e}")
            return []
    
    def _extract_sections(self, report: str) -> List[Dict[str, Any]]:
        """从 Markdown 报告中提取章节。
        
        参数：
            report: Markdown 报告文本。
            
        返回：
            带 header、content 和 position 的章节字典列表。
        """
        sections = []
        lines = report.split('\n')
        current_section = None
        current_content = []
        section_start = 0
        
        for i, line in enumerate(lines):
            # 检查标题（## 或 ###）
            header_match = re.match(r'^(#{2,3})\s+(.+)$', line)
            
            if header_match:
                # 保存前一章节
                if current_section:
                    sections.append({
                        "header": current_section,
                        "content": '\n'.join(current_content).strip(),
                        "start_line": section_start,
                        "end_line": i - 1,
                    })
                
                # 开始新章节
                current_section = header_match.group(2)
                current_content = []
                section_start = i
            elif current_section:
                current_content.append(line)
        
        # 不要忘记最后一节
        if current_section:
            sections.append({
                "header": current_section,
                "content": '\n'.join(current_content).strip(),
                "start_line": section_start,
                "end_line": len(lines) - 1,
            })
        
        return sections
    
    def _build_analysis_prompt(
        self,
        query: str,
        sections: List[Dict[str, Any]],
    ) -> str:
        """构建提示词，让 LLM 分析哪些章节需要图片。
        
        参数：
            query: 研究查询。
            sections: 提取的章节列表。
            
        返回：
            分析提示词字符串。
        """
        sections_text = "\n\n".join([
            f"### Section {i+1}: {s['header']}\n{s['content'][:500]}..."
            for i, s in enumerate(sections)
        ])
        
        return f"""Analyze the following research report sections and identify which {self.max_images} sections would benefit MOST from a visual illustration or diagram.

RESEARCH TOPIC: {query}

REPORT SECTIONS:
{sections_text}

For each recommended section, provide:
1. The section number (1-indexed)
2. A specific, detailed image prompt that would create an informative illustration
3. A brief explanation of why this section benefits from visualization

IMPORTANT:
- Choose sections where visual representation would genuinely aid understanding
- Focus on concepts, processes, comparisons, or data that are inherently visual
- Avoid sections that are purely textual analysis or conclusions
- The image prompt should be specific enough to generate a relevant, professional illustration
- Do NOT suggest images for introduction or conclusion sections

Respond in JSON format:
{{
    "suggestions": [
        {{
            "section_number": 1,
            "section_header": "Section Title",
            "image_prompt": "Detailed prompt for generating an informative illustration...",
            "reason": "Why this section benefits from visualization"
        }}
    ]
}}

Return ONLY the JSON, no additional text."""
    
    def _parse_analysis_response(
        self,
        response: str,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """解析 LLM 的分析响应。
        
        参数：
            response: LLM 响应文本。
            sections: 原始章节列表。
            
        返回：
            图片建议字典列表。
        """
        try:
            data = json_repair.loads(response)
            if not isinstance(data, dict):
                logger.warning("No JSON object found in analysis response")
                return []

            suggestions = data.get("suggestions", []) or []
            if not isinstance(suggestions, list):
                return []

            # 用章节数据充实
            enriched = []
            for s in suggestions:
                if not isinstance(s, dict):
                    continue
                section_num = s.get("section_number", 0) - 1  # 转为 0 索引
                if 0 <= section_num < len(sections):
                    section = sections[section_num]
                    enriched.append({
                        "section_header": section["header"],
                        "section_content": section["content"][:1000],
                        "image_prompt": s.get("image_prompt", ""),
                        "reason": s.get("reason", ""),
                        "insert_after_line": section["start_line"],
                    })

            return enriched

        except Exception as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            return []
    
    async def generate_images_for_report(
        self,
        report: str,
        query: str,
        research_id: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """生成图片并嵌入报告。
        
        这是编排研究报告完整图片生成工作流的
        主要方法。
        
        参数：
            report: Markdown 报告文本。
            query: 原始研究查询。
            research_id: 可选的研究 ID，用于文件组织。
            
        返回：
            （带嵌入图片的修改后报告, 生成的图片列表）元组。
        """
        if not self.is_enabled():
            logger.info("Image generation is not enabled, skipping")
            return report, []
        
        # 通知图片生成开始
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "image_generation_start",
                "🎨 Analyzing report for image generation opportunities...",
                self.researcher.websocket,
            )
        
        # 分析报告以寻找图片机会
        suggestions = await self.analyze_report_for_images(report, query)
        
        if not suggestions:
            logger.info("No sections identified for image generation")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "image_generation_skip",
                    "📝 No sections identified that would benefit from images",
                    self.researcher.websocket,
                )
            return report, []
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "image_generation_analyzing",
                f"🔍 Found {len(suggestions)} sections that would benefit from images",
                self.researcher.websocket,
            )
        
        # 为每条建议生成图片
        generated_images = []
        for i, suggestion in enumerate(suggestions):
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "image_generating",
                    f"🖼️ Generating image {i+1}/{len(suggestions)}: {suggestion['section_header'][:50]}...",
                    self.researcher.websocket,
                )
            
            try:
                images = await self.image_provider.generate_image(
                    prompt=suggestion["image_prompt"],
                    context=suggestion["section_content"],
                    research_id=research_id,
                    num_images=1,
                )
                
                if images:
                    image_info = images[0]
                    image_info["section_header"] = suggestion["section_header"]
                    generated_images.append(image_info)
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "image_generated",
                            f"✅ Image generated for: {suggestion['section_header'][:50]}",
                            self.researcher.websocket,
                        )
            except Exception as e:
                logger.error(f"Failed to generate image for section '{suggestion['section_header']}': {e}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "image_generation_error",
                        f"⚠️ Failed to generate image: {str(e)[:100]}",
                        self.researcher.websocket,
                    )
        
        # 将图片嵌入报告
        if generated_images:
            report = self._embed_images_in_report(report, generated_images, suggestions)
            self.generated_images = generated_images
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "image_generation_complete",
                    f"🎉 Successfully generated and embedded {len(generated_images)} images",
                    self.researcher.websocket,
                )
                
                # 通过 WebSocket 发送生成的图片
                await stream_output(
                    "generated_images",
                    "inline_images",
                    json.dumps([{"url": img["url"], "alt": img["alt_text"]} for img in generated_images]),
                    self.researcher.websocket,
                    True,
                    generated_images,
                )
        
        return report, generated_images
    
    def _embed_images_in_report(
        self,
        report: str,
        images: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]],
    ) -> str:
        """将生成的图片嵌入报告 Markdown。
        
        参数：
            report: 原始报告 Markdown。
            images: 生成的图片信息列表。
            suggestions: 带章节信息的原始建议。
            
        返回：
            带嵌入图片的修改后报告。
        """
        lines = report.split('\n')
        
        # 创建章节标题到图片的映射
        section_to_image = {}
        for img, sug in zip(images, suggestions):
            section_to_image[sug["section_header"]] = img
        
        # 找到章节标题并在其后插入图片
        modified_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            modified_lines.append(line)
            
            # 检查这是否需要图片的标题
            header_match = re.match(r'^(#{2,3})\s+(.+)$', line)
            if header_match:
                header_text = header_match.group(2)
                if header_text in section_to_image:
                    img = section_to_image[header_text]
                    # 在标题后插入图片并留空行
                    image_markdown = f"\n![{img['alt_text']}]({img['url']})\n"
                    modified_lines.append(image_markdown)
            
            i += 1
        
        return '\n'.join(modified_lines)
    
    def get_generated_images(self) -> List[Dict[str, Any]]:
        """获取已生成图片列表。
        
        返回：
            生成的图片信息字典列表。
        """
        return self.generated_images

    async def process_image_placeholders(
        self,
        report: str,
        query: str,
        research_id: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """处理报告中的 [IMAGE: description] 占位符并生成图片。
        
        此方法找到报告中所有图片占位符，为每个
        占位符生成图片，并把占位符替换为实际的
        Markdown 图片。
        
        参数：
            report: 带 [IMAGE: ...] 占位符的 Markdown 报告文本。
            query: 原始研究查询（用于上下文）。
            research_id: 可选的研究 ID，用于文件组织。
            
        返回：
            （带嵌入图片的修改后报告, 生成的图片列表）元组。
        """
        if not self.is_enabled():
            # 图片生成未启用时，直接移除占位符
            report = re.sub(r'\[IMAGE:\s*[^\]]+\]', '', report)
            return report, []
        
        # 找到所有图片占位符
        placeholder_pattern = r'\[IMAGE:\s*([^\]]+)\]'
        placeholders = list(re.finditer(placeholder_pattern, report))
        
        if not placeholders:
            logger.info("No image placeholders found in report")
            return report, []
        
        # 限制到 max_images
        placeholders = placeholders[:self.max_images]
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "image_placeholders_found",
                f"🎨 Found {len(placeholders)} image placeholders to process",
                self.researcher.websocket,
            )
        
        generated_images = []
        replacements = []  # List of (original_text, replacement_text) tuples
        
        for i, match in enumerate(placeholders):
            image_description = match.group(1).strip()
            original_text = match.group(0)
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "image_generating",
                    f"🖼️ Generating image {i+1}/{len(placeholders)}: {image_description[:60]}...",
                    self.researcher.websocket,
                )
            
            try:
                # 从配置获取图片风格（默认为匹配应用主题的 "dark"）
                image_style = getattr(self.cfg, 'image_generation_style', 'dark')
                logger.info(f"Using image style: {image_style}")
                
                # 用暗色模式样式生成图片
                images = await self.image_provider.generate_image(
                    prompt=image_description,
                    context=query,  # 使用查询作为上下文
                    research_id=research_id,
                    num_images=1,
                    style=image_style,
                )
                
                if images:
                    image_info = images[0]
                    image_info["description"] = image_description
                    generated_images.append(image_info)
                    
                    # 创建 Markdown 替换，使用绝对路径以兼容 PDF
                    # 使用绝对 URL 以在 PDF/DOCX 中正确渲染
                    markdown_image = f"\n\n![{image_info['alt_text']}]({image_info['url']})\n\n"
                    replacements.append((original_text, markdown_image))
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "image_generated",
                            f"✅ Generated: {image_description[:40]}...",
                            self.researcher.websocket,
                        )
                else:
                    # 图片生成失败时移除占位符
                    replacements.append((original_text, ""))
                    logger.warning(f"No image generated for: {image_description[:50]}")
                    
            except Exception as e:
                logger.error(f"Failed to generate image for '{image_description[:50]}': {e}")
                # 移除失败的占位符
                replacements.append((original_text, ""))
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "image_generation_error",
                        f"⚠️ Failed to generate: {str(e)[:80]}",
                        self.researcher.websocket,
                    )
        
        # 应用所有替换
        modified_report = report
        for original, replacement in replacements:
            modified_report = modified_report.replace(original, replacement, 1)
        
        self.generated_images = generated_images
        
        if generated_images and self.researcher.verbose:
            await stream_output(
                "logs",
                "image_generation_complete",
                f"🎉 Successfully generated {len(generated_images)} inline images",
                self.researcher.websocket,
            )
            
            # 通过 WebSocket 发送生成的图片
            await stream_output(
                "generated_images",
                "inline_images",
                json.dumps([{"url": img["url"], "alt": img["alt_text"]} for img in generated_images]),
                self.researcher.websocket,
                True,
                generated_images,
            )
        
        return modified_report, generated_images
