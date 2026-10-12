from __future__ import annotations

import operator
import os
import re
from urllib.parse import quote
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
# class ImageSpec(BaseModel):
#     placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
#     filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
#     alt: str
#     caption: str
#     prompt: str = Field(..., description="Prompt to send to the image model.")
#     size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
#     quality: Literal["low", "medium", "high"] = "medium"

class TextElement(BaseModel):
    text: str = Field(
        ...,
        description="Exact text that must appear on the final image. Preserve spelling exactly."
    )
    x: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Horizontal position as normalized coordinate from 0 to 1."
    )
    y: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Vertical position as normalized coordinate from 0 to 1."
    )
    font_size: int = Field(
        32,
        ge=14,
        le=80,
        description="Font size in pixels."
    )
    bold: bool = True


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str

    prompt: str = Field(
        ...,
        description=(
            "Prompt for generating ONLY the visual artwork. "
            "Do not depend on the image model to render exact text."
        )
    )

    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024"
    ] = "1024x1024"

    quality: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"

    text_elements: List[TextElement] = Field(
        default_factory=list,
        description=(
            "Exact labels/headings/numbers to overlay on the generated image "
            "using Python. Text must be spelled exactly."
        )
    )


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def _response_text(content: Any) -> str:
    """Extract text from string or content-block responses returned by chat models."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        text = "".join(text_parts)
    else:
        text = ""

    text = text.strip()
    if not text:
        raise ValueError("Worker model returned no text content.")
    return text


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    )
    section_md = _response_text(response.content)

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


# DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
# Decide if images/diagrams are needed for THIS blog.

# Rules:
# - Max 3 images total.
# - Each image must materially improve understanding (diagram/flow/table-like visual).
# - Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
# - If no images needed: md_with_placeholders must equal input and images=[].
# - Avoid decorative images; prefer technical diagrams with short labels.
# Return strictly GlobalImagePlan.
# """

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor and visual designer.

Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding.
- Prefer technical diagrams, architecture diagrams, workflows,
  process diagrams, comparisons, and explanatory infographics.
- Avoid decorative images.

IMPORTANT TEXT RULE:
The image generation model is NOT reliable at rendering exact text.

Therefore:
- The image generation prompt must NOT ask the image model to render
  important text, labels, headings, numbers, percentages, or terminology.
- The image prompt should describe ONLY the visual structure, shapes,
  icons, arrows, colors, objects, layout, and overall composition.
- Put all important exact text into text_elements.
- text_elements will be rendered later by Python/PIL.
- Preserve spelling exactly in text_elements.
- Do NOT put text inside the image-generation prompt unless absolutely
  necessary for visual context.

For text_elements:
- Use short, meaningful labels.
- Use normalized x/y coordinates from 0 to 1.
- Place labels near the relevant visual element.
- Avoid overlapping labels.
- Use 18–50 px font sizes depending on importance.

Insert placeholders exactly:
[[IMAGE_1]]
[[IMAGE_2]]
[[IMAGE_3]]

If no images are needed:
md_with_placeholders must equal input and images=[].

Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _pollinations_generate_image_bytes(
    prompt: str,
    size: str = "1024x1024",
) -> bytes:
    """
    Generate a visual-only image using Pollinations.

    Important:
    We explicitly tell the image model NOT to render text.
    Exact text is added later using PIL.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    api_key = os.getenv("POLLINATIONS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "POLLINATIONS_API_KEY is not configured in the environment."
        )

    clean_prompt = re.sub(r"\[.*?\]", "", prompt)
    clean_prompt = clean_prompt.replace("\n", " ").strip()

    if not clean_prompt:
        raise ValueError("Prompt became empty after cleaning.")

    # Very important instruction for the image model.
    visual_only_instruction = """
IMPORTANT:
Generate ONLY the visual artwork.

DO NOT render:
- words
- letters
- numbers
- headings
- labels
- captions
- percentages
- logos
- UI text
- typography

Leave appropriate clean space for text that will be added later
programmatically.
"""

    final_prompt = f"{clean_prompt} {visual_only_instruction}"

    encoded_prompt = quote(final_prompt, safe="")

    url = (
        "https://gen.pollinations.ai/image/"
        f"{encoded_prompt}"
        "?model=flux"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=120,
        )

        if response.status_code != 200:
            error_text = response.text[:500]

            raise RuntimeError(
                "Pollinations image generation failed "
                f"(HTTP {response.status_code}): {error_text}"
            )

        content_type = response.headers.get("Content-Type", "")

        if not content_type.startswith("image/"):
            raise RuntimeError(
                "Pollinations returned unexpected content type: "
                f"{content_type}"
            )

        if not response.content:
            raise RuntimeError(
                "Pollinations returned an empty image response."
            )

        return response.content

    except requests.RequestException as e:
        raise RuntimeError(
            f"Pollinations network/API error: {e}"
        ) from e

def _get_font(size: int, bold: bool = True):
    """
    Find a commonly available font across Windows/Linux environments.
    """

    candidates = []

    if bold:
        candidates.extend([
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def _draw_text_with_background(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font,
    max_width: int,
):
    """
    Draw readable text with a subtle background box.
    """

    x, y = position

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Prevent very long labels from going outside image.
    if text_width > max_width:
        words = text.split()
        lines = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip()

            candidate_bbox = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )

            candidate_width = (
                candidate_bbox[2] - candidate_bbox[0]
            )

            if candidate_width <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        text = "\n".join(lines)

        bbox = draw.multiline_textbbox(
            (x, y),
            text,
            font=font,
            spacing=4,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    padding = 8

    draw.rounded_rectangle(
        (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding,
        ),
        radius=8,
        fill=(255, 255, 255, 235),
    )

    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(20, 20, 20),
        spacing=4,
    )


def _overlay_exact_text(
    image_bytes: bytes,
    text_elements: List[dict],
) -> bytes:
    """
    Open the generated image and render exact text using PIL.

    The image model never gets to decide how the text is spelled.
    """

    from io import BytesIO

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")

    width, height = image.size

    # Transparent overlay.
    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    for item in text_elements:
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        x_ratio = float(item.get("x", 0.5))
        y_ratio = float(item.get("y", 0.5))

        font_size = int(item.get("font_size", 32))
        bold = bool(item.get("bold", True))

        x_ratio = max(0.0, min(1.0, x_ratio))
        y_ratio = max(0.0, min(1.0, y_ratio))

        x = int(width * x_ratio)
        y = int(height * y_ratio)

        font = _get_font(
            size=font_size,
            bold=bold,
        )

        # Keep text inside the image.
        x = min(
            max(20, x),
            max(20, width - 250),
        )

        y = min(
            max(20, y),
            max(20, height - 100),
        )

        _draw_text_with_background(
            draw=draw,
            position=(x, y),
            text=text,
            font=font,
            max_width=min(420, width - x - 30),
        )

    final_image = Image.alpha_composite(
        image,
        overlay,
    )

    output = BytesIO()

    final_image.convert("RGB").save(
        output,
        format="PNG",
        optimize=True,
    )

    return output.getvalue()
    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    # parts = getattr(resp, "parts", None)
    # if not parts and getattr(resp, "candidates", None):
    #     try:
    #         parts = resp.candidates[0].content.parts
    #     except Exception:
    #         parts = None

    # if not parts:
    #     raise RuntimeError("No image content returned (safety/quota/SDK change).")

    # for part in parts:
    #     inline = getattr(part, "inline_data", None)
    #     if inline and getattr(inline, "data", None):
    #         return inline.data

    # raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


# def generate_and_place_images(state: State) -> dict:
#     plan = state["plan"]
#     assert plan is not None

#     md = state.get("md_with_placeholders") or state["merged_md"]
#     image_specs = state.get("image_specs", []) or []

#     # If no images requested, just write merged markdown
#     if not image_specs:
#         filename = f"{_safe_slug(plan.blog_title)}.md"
#         Path(filename).write_text(md, encoding="utf-8")
#         return {"final": md}

#     images_dir = Path("images")
#     images_dir.mkdir(exist_ok=True)

#     for spec in image_specs:
#         placeholder = spec["placeholder"]
#         filename = spec["filename"]
#         out_path = images_dir / filename

#         # generate only if needed
#         if not out_path.exists():
#             try:
#                 img_bytes = _pollinations_generate_image_bytes(spec["prompt"])
#                 out_path.write_bytes(img_bytes)
#             except Exception as e:
#                 # graceful fallback: keep doc usable
#                 prompt_block = (
#                     f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
#                     f"> **Alt:** {spec.get('alt','')}\n>\n"
#                     f"> **Prompt:** {spec.get('prompt','')}\n>\n"
#                     f"> **Error:** {e}\n"
#                 )
#                 md = md.replace(placeholder, prompt_block)
#                 continue

#         img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
#         md = md.replace(placeholder, img_md)

#     filename = f"{_safe_slug(plan.blog_title)}.md"
#     Path(filename).write_text(md, encoding="utf-8")
#     return {"final": md}

def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                # --------------------------------------------------
                # STEP 1:
                # Pollinations creates ONLY the visual.
                # --------------------------------------------------
                img_bytes = _pollinations_generate_image_bytes(
                    prompt=spec["prompt"],
                    size=spec.get("size", "1024x1024"),
                )

                # --------------------------------------------------
                # STEP 2:
                # Python adds exact text.
                # --------------------------------------------------
                text_elements = spec.get(
                    "text_elements",
                    [],
                )

                if text_elements:
                    img_bytes = _overlay_exact_text(
                        image_bytes=img_bytes,
                        text_elements=text_elements,
                    )

                # --------------------------------------------------
                # STEP 3:
                # Save final image.
                # --------------------------------------------------
                out_path.write_bytes(img_bytes)

            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** "
                    f"{spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )

                md = md.replace(
                    placeholder,
                    prompt_block,
                )

                continue

        img_md = (
            f"![{spec['alt']}](images/{filename})\n"
            f"*{spec['caption']}*"
        )

        md = md.replace(
            placeholder,
            img_md,
        )

    filename = f"{_safe_slug(plan.blog_title)}.md"

    Path(filename).write_text(
        md,
        encoding="utf-8",
    )

    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
