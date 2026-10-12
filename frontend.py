from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Page configuration must run before any other Streamlit command so the
# sidebar state is applied reliably.
st.set_page_config(
    page_title="BlogBolt AI - Topic In, Blog Out",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    # [data-testid="stHeader"] {
    #     display: none;
    # }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep the existing LangGraph backend. This UI does not rewrite backend.py.
from backend import app


# -------------------------------------------------------------------
# Page setup and styling
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
      --bp-bg: #071124;
      --bp-panel: #0b1830;
      --bp-panel-2: #0d1c36;
      --bp-border: rgba(123, 151, 209, .22);
      --bp-text: #eef3ff;
      --bp-muted: #98a9c8;
      --bp-purple: #7757ff;
      --bp-blue: #3d6bff;
      --bp-green: #39d6ae;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp {
      background:
        radial-gradient(ellipse at 78% 0%, rgba(43, 73, 173, .18), transparent 35%),
        radial-gradient(ellipse at 4% 80%, rgba(104, 55, 220, .09), transparent 30%),
        var(--bp-bg);
      color: var(--bp-text);
    }
    [data-testid="stHeader"] { background: rgba(7,17,36,.72); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0b1730 0%, #091328 100%);
      border-right: 1px solid var(--bp-border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1rem; }
    .block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1800px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; }
    p, li, label, div { overflow-wrap: anywhere; }
    .bp-brand { font: 700 1.28rem 'Space Grotesk', sans-serif; color: #f3f5ff; letter-spacing: -.04em; }
    .bp-brand span { color: #9b7aff; }
    .bp-tagline { color: var(--bp-muted); font-size: .82rem; margin-top: -.15rem; }
    .bp-kicker { color: #a8b7d7; text-transform: uppercase; letter-spacing: .13em; font-size: .68rem; font-weight: 700; }
    .bp-heading { font: 700 1.7rem 'Space Grotesk', sans-serif; color: #f5f7ff; margin-top: .7rem; margin-bottom: .1rem; }
    .bp-subheading { color: var(--bp-muted); font-size: .91rem; margin-bottom: 1rem; }
    .bp-card {
      background: linear-gradient(145deg, rgba(15,31,58,.97), rgba(9,22,43,.97));
      border: 1px solid var(--bp-border);
      border-radius: 16px;
      padding: 1rem 1.05rem;
      box-shadow: 0 14px 40px rgba(0,0,0,.13);
      margin-bottom: .85rem;
    }
    .bp-card-title { font: 600 1rem 'Space Grotesk', sans-serif; color: #f0f4ff; margin-bottom: .65rem; }
    .bp-muted { color: var(--bp-muted); font-size: .82rem; }
    .bp-status {
      display: inline-flex; align-items: center; gap: .45rem;
      border: 1px solid rgba(57,214,174,.22); background: rgba(57,214,174,.08);
      color: #78e8c8; padding: .32rem .65rem; border-radius: 999px; font-size: .76rem;
    }
    .bp-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#39d6ae; box-shadow:0 0 10px rgba(57,214,174,.7); }
    .bp-step { display:flex; align-items:center; gap:.65rem; padding:.58rem .52rem; border-radius:10px; margin:.18rem 0; font-size:.83rem; }
    .bp-step.active { background: linear-gradient(90deg, rgba(119,87,255,.24), rgba(61,107,255,.08)); border:1px solid rgba(119,87,255,.22); }
    .bp-step.done { color:#bcefe2; }
    .bp-step.pending { color:#7f91b2; }
    .bp-step-icon { width:25px; height:25px; border-radius:8px; display:flex; align-items:center; justify-content:center; background:#13284a; border:1px solid var(--bp-border); flex-shrink:0; }
    .bp-step.done .bp-step-icon { color:#48dfb8; }
    .bp-step.active .bp-step-icon { background:#6244ec; color:white; box-shadow:0 0 16px rgba(119,87,255,.28); }
    .bp-divider { border:0; border-top:1px solid var(--bp-border); margin:1rem 0; }
    .bp-visual-label { color:#dce6ff; font-size:.8rem; font-weight:600; margin:.35rem 0 .2rem; }
    .bp-small { color:var(--bp-muted); font-size:.72rem; }
    .bp-footer-status { color:#66e5c2; font-size:.77rem; font-weight:600; }
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
      background: #09172d !important; color:#edf3ff !important;
      border-color: rgba(117,145,204,.28) !important; border-radius:10px !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
      color: #9fb4d8 !important; opacity: 1 !important;
    }
    .stButton > button, .stDownloadButton > button {
      border-radius: 10px; border: 1px solid rgba(128,151,204,.3);
      background: #102140; color: #eaf0ff; font-weight:600;
      transition: all .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      border-color: #8065ff; color:white; transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
      background: linear-gradient(100deg, #3d63ff 0%, #8050ff 100%);
      border: 0; color:white; box-shadow: 0 7px 22px rgba(83,78,255,.2);
    }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--bp-border); }
    .stTabs [data-baseweb="tab"] { color:#9eafd0; padding:.6rem .85rem; }
    .stTabs [aria-selected="true"] { color:#fff !important; }
    [data-testid="stMetric"] { background:#0c1b34; border:1px solid var(--bp-border); padding:.7rem; border-radius:12px; }
    [data-testid="stMetricLabel"] { color:#a3b4d4; }
    [data-testid="stMetricValue"] { color:#f1f5ff; }
    .stAlert { border-radius:12px; }
    @media (max-width: 900px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .bp-heading { font-size:1.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_slug(title: str) -> str:
    value = title.strip().lower()
    value = re.sub(r"[^a-z0-9 _-]+", "", value)
    value = re.sub(r"\s+", "_", value).strip("_")
    return value or "blog"


def extract_title(md: str, fallback: str = "Untitled blog") -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def as_plain_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [as_plain_dict(v) for v in value]
    return value


def get_recent_blogs() -> List[Path]:
    excluded = {"README.md", "requirements.md"}
    files = [p for p in ROOT.glob("*.md") if p.is_file() and p.name not in excluded]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_image(src: str) -> Path:
    clean = src.strip().lstrip("./")
    path = (ROOT / clean).resolve()
    try:
        path.relative_to(ROOT.resolve())
        return path
    except ValueError:
        return ROOT / "__invalid_path__"


def render_markdown_with_images(md: str) -> None:
    """Render Markdown in chunks so local images and captions display correctly."""
    image_re = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
    caption_re = re.compile(r"^\*(?P<caption>.+)\*$")
    last = 0
    matches = list(image_re.finditer(md))

    if not matches:
        st.markdown(md)
        return

    for match in matches:
        before = md[last:match.start()]
        if before.strip():
            st.markdown(before)

        alt = match.group("alt").strip()
        src = match.group("src").strip()
        caption = None
        # Captions are usually the next standalone italic line.
        tail = md[match.end():]
        lines = tail.splitlines()
        if lines and lines[0].strip() == "":
            lines = lines[1:]
        if lines:
            cap_match = caption_re.match(lines[0].strip())
            if cap_match:
                caption = cap_match.group("caption").strip()
                consumed = len(tail.splitlines()[0]) + 1 if tail.splitlines() else 0
                # Remove the caption line from the remaining Markdown.
                remainder = tail[consumed:]
                md = md[:match.end()] + remainder
                matches = list(image_re.finditer(md))
                # The main loop is driven by offsets in the original text, so
                # avoid mutating offsets here; caption is optional for preview.
        if src.startswith(("http://", "https://")):
            st.image(src, caption=caption or alt or None, width="stretch")
        else:
            image_path = resolve_image(src)
            if image_path.exists() and image_path.is_file():
                st.image(str(image_path), caption=caption or alt or None, width="stretch")
            else:
                st.warning(f"Image not found: `{src}`")
        last = match.end()

    remaining = md[last:]
    if remaining.strip():
        st.markdown(remaining)


def make_bundle(md_text: str, md_filename: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(md_filename, md_text.encode("utf-8"))
        if IMAGES_DIR.exists():
            for path in IMAGES_DIR.iterdir():
                if path.is_file():
                    archive.write(path, arcname=f"images/{path.name}")
    return buffer.getvalue()


def get_graph_node(payload: Any) -> Optional[str]:
    if isinstance(payload, dict) and len(payload) == 1:
        key = next(iter(payload))
        if isinstance(payload[key], dict):
            return str(key)
    return None


def workflow_status(node_name: str) -> Dict[str, str]:
    """Map actual graph node names to UI stage labels; unknown stages remain neutral."""
    name = node_name.lower()
    if "router" in name:
        return {"Router": "completed"}
    if "research" in name:
        return {"Router": "completed", "Web Research": "completed"}
    if "orchestrator" in name:
        return {"Router": "completed", "Web Research": "completed", "Outline": "completed"}
    if "worker" in name:
        return {"Router": "completed", "Web Research": "completed", "Outline": "completed", "Writing": "running"}
    if "reducer" in name or "merge_content" in name:
        return {"Router": "completed", "Web Research": "completed", "Outline": "completed", "Writing": "completed", "Content Reduction": "running"}
    if "decide_images" in name:
        return {"Router": "completed", "Web Research": "completed", "Outline": "completed", "Writing": "completed", "Content Reduction": "completed", "Visual Generation": "running"}
    if "generate_and_place_images" in name:
        return {"Router": "completed", "Web Research": "completed", "Outline": "completed", "Writing": "completed", "Content Reduction": "completed", "Visual Generation": "running"}
    return {}


def render_workflow(statuses: Dict[str, str]) -> None:
    stages = [
        ("Router", "↗"),
        ("Web Research", "⌕"),
        ("Outline", "▤"),
        ("Writing", "✎"),
        ("Content Reduction", "⇄"),
        ("Visual Generation", "▧"),
    ]
    for label, icon in stages:
        state = statuses.get(label, "pending")
        if state == "completed":
            symbol, detail, css_state = "✓", "Completed", "done"
        elif state == "running":
            symbol, detail, css_state = "◌", "In progress", "active"
        else:
            symbol, detail, css_state = icon, "Pending", "pending"
        st.markdown(
            f'<div class="bp-step {css_state}"><div class="bp-step-icon">{symbol}</div>'
            f'<div style="flex:1">{label}</div><div class="bp-small">{detail}</div></div>',
            unsafe_allow_html=True,
        )


def current_output() -> Optional[Dict[str, Any]]:
    return st.session_state.get("last_out")


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="bp-brand">⚡BlogBolt <span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-tagline">From ideas to impact</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("＋  New Blog", type="primary", width="stretch"):
        st.session_state["last_out"] = None
        st.session_state["topic"] = ""
        st.rerun()

    st.markdown('<div class="bp-kicker" style="margin:1.2rem 0 .55rem">Recent blogs</div>', unsafe_allow_html=True)
    recent = get_recent_blogs()
    if not recent:
        st.caption("Your generated blogs will appear here.")
    else:
        for path in recent[:6]:
            title = path.stem.replace("_", " ")
            try:
                title = extract_title(path.read_text(encoding="utf-8", errors="replace"), title)
            except OSError:
                pass
            if st.button(f"▤  {title[:30]}", key=f"recent_{path.name}", width="stretch"):
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": path.read_text(encoding="utf-8", errors="replace"),
                }
                st.rerun()

    st.markdown('<hr class="bp-divider">', unsafe_allow_html=True)
    if st.button("▧  Generated Images", width="stretch"):
        st.session_state["active_tab"] = "Images"
    if st.button("⚙  Settings", width="stretch"):
        st.session_state["show_settings"] = not st.session_state.get("show_settings", False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="bp-muted">Powered by LangGraph · Gemini · Tavily · Pollinations</div>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
head_left, head_right = st.columns([5, 1.4], vertical_alignment="center")
with head_left:
    st.markdown('<div class="bp-heading">AI Blog Writer</div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-subheading">Research, write, and create technical visuals in one workspace.</div>', unsafe_allow_html=True)
with head_right:
    st.markdown('<div style="text-align:right"><span class="bp-status"><span class="bp-dot"></span> AI Ready</span></div>', unsafe_allow_html=True)

if st.session_state.get("show_settings"):
    with st.expander("Workspace settings", expanded=True):
        st.info("API keys remain in your backend environment. This panel does not expose secrets.")
        st.caption("The existing LangGraph backend is imported directly from `backend.py`.")


# -------------------------------------------------------------------
# Topic form
# -------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="bp-card-title">What do you want to write about?</div>', unsafe_allow_html=True)
    topic_default = st.session_state.get("topic", "")
    topic = st.text_area(
        "Blog topic",
        value=topic_default,
        height=90,
        placeholder="e.g. How AI Agents Actually Work: From LLM Reasoning to Tool Use and Memory",
        label_visibility="collapsed",
        key="topic_input",
    )
    c1, c2, c3 = st.columns([1, 1, 1.1])
    with c1:
        audience = st.selectbox("Audience", ["Developers", "Beginners", "Data Scientists", "Technical Leaders", "General Readers"], index=0)
    with c2:
        tone = st.selectbox("Tone", ["Technical & Clear", "Conversational", "Professional", "Beginner-friendly", "Research-oriented"], index=0)
    with c3:
        research = st.toggle("Web research", value=True, help="The current backend router still decides whether research is needed based on the topic.")

    generate = st.button("✦  Generate Blog", type="primary", width="stretch")


# -------------------------------------------------------------------
# Generation flow — call the graph once and collect its real updates.
# -------------------------------------------------------------------
if generate:
    if not topic.strip():
        st.warning("Please enter a topic before generating a blog.")
    else:
        st.session_state["topic"] = topic.strip()
        inputs: Dict[str, Any] = {
            "topic": topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": date.today().isoformat(),
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }

        status_box = st.status("Running BlogPilot AI workflow…", expanded=True)
        workflow_col, event_col = st.columns([1.2, 1])
        with workflow_col:
            st.markdown("**Workflow progress**")
            workflow_placeholder = st.empty()
        with event_col:
            st.markdown("**Latest event**")
            event_placeholder = st.empty()

        latest_state: Dict[str, Any] = {}
        logs: List[str] = []
        latest_node = ""
        statuses: Dict[str, str] = {}

        try:
            # Stream updates from the graph exactly once; then use the final
            # output carried by the graph stream. This avoids duplicate LLM runs.
            stream = app.stream(inputs, stream_mode="updates")
            for event in stream:
                node = get_graph_node(event)
                if node:
                    latest_node = node
                    mapped = workflow_status(node)
                    if mapped:
                        statuses.update(mapped)
                    event_placeholder.markdown(f"`{node}` is running / returned an update.")
                if isinstance(event, dict):
                    for value in event.values():
                        if isinstance(value, dict):
                            latest_state.update(value)
                logs.append(json.dumps(event, default=str)[:1500])
                with workflow_placeholder.container():
                    render_workflow(statuses)

            # Stream updates may not expose the complete state on every graph version.
            # Invoke only if the stream did not yield a final field; this is a fallback.
            if "final" not in latest_state:
                status_box.write("Collecting final output…")
                latest_state = app.invoke(inputs)

            latest_state = as_plain_dict(latest_state)
            if not isinstance(latest_state, dict):
                raise RuntimeError("The graph returned an unexpected output format.")

            st.session_state["last_out"] = latest_state
            st.session_state["last_logs"] = logs
            status_box.update(label="Blog generated successfully", state="complete", expanded=False)
            st.success("Blog generated successfully.")
        except Exception as exc:
            status_box.update(label="Blog generation failed", state="error", expanded=True)
            st.error(f"Unable to generate the blog: {exc}")
            st.session_state["last_logs"] = logs


# -------------------------------------------------------------------
# Output workspace
# -------------------------------------------------------------------
out = current_output()

if out:
    final_md = str(out.get("final") or "")
    plan_obj = as_plain_dict(out.get("plan"))
    evidence = as_plain_dict(out.get("evidence") or [])
    image_specs = as_plain_dict(out.get("image_specs") or [])
    title = (plan_obj or {}).get("blog_title") if isinstance(plan_obj, dict) else None
    title = title or extract_title(final_md, "Generated blog")
    md_filename = f"{safe_slug(title)}.md"

    left, right = st.columns([2.25, 1], gap="large")

    with left:
        st.markdown('<div class="bp-card-title">Article workspace</div>', unsafe_allow_html=True)
        article_tabs = st.tabs(["📝  Article preview", "⌘  Markdown source", "🧩  Plan", "🔎  Evidence", "▧  Images", "☷  Logs"])

        with article_tabs[0]:
            if final_md.strip():
                render_markdown_with_images(final_md)
            else:
                st.info("The graph completed but did not return final Markdown.")
            a1, a2, a3 = st.columns([1, 1.25, 1])
            with a1:
                st.download_button("⬇ Export Markdown", data=final_md.encode("utf-8"), file_name=md_filename, mime="text/markdown", width="stretch")
            with a2:
                st.download_button("📦 Download blog + images", data=make_bundle(final_md, md_filename), file_name=f"{safe_slug(title)}_bundle.zip", mime="application/zip", width="stretch")
            with a3:
                if st.button("↻ Clear result", width="stretch"):
                    st.session_state["last_out"] = None
                    st.rerun()

        with article_tabs[1]:
            st.code(final_md or "No Markdown returned.", language="markdown")
            st.download_button("Download source .md", data=final_md.encode("utf-8"), file_name=md_filename, mime="text/markdown")

        with article_tabs[2]:
            if not plan_obj:
                st.info("No outline plan is available for this saved blog.")
            else:
                st.markdown(f"**{plan_obj.get('blog_title', title)}**")
                p1, p2, p3 = st.columns(3)
                p1.metric("Audience", plan_obj.get("audience", "—"))
                p2.metric("Tone", plan_obj.get("tone", "—"))
                p3.metric("Blog type", plan_obj.get("blog_kind", "—"))
                tasks = plan_obj.get("tasks", [])
                if tasks:
                    df = pd.DataFrame([{
                        "ID": task.get("id"),
                        "Section": task.get("title"),
                        "Target words": task.get("target_words"),
                        "Research": task.get("requires_research"),
                        "Citations": task.get("requires_citations"),
                        "Code": task.get("requires_code"),
                    } for task in tasks])
                    st.dataframe(df, width="stretch", hide_index=True)
                    with st.expander("Outline details"):
                        st.json(tasks)

        with article_tabs[3]:
            if not evidence:
                st.info("No evidence returned. This may be an evergreen topic, or the backend may not have returned research results.")
            else:
                rows = []
                for item in evidence:
                    item = as_plain_dict(item)
                    if isinstance(item, dict):
                        rows.append({
                            "Title": item.get("title"),
                            "Date": item.get("published_at"),
                            "Source": item.get("source"),
                            "URL": item.get("url"),
                        })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                for row in rows:
                    if row.get("Title") and row.get("URL"):
                        st.markdown(f"- [{row['Title']}]({row['URL']})")

        with article_tabs[4]:
            image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file()]) if IMAGES_DIR.exists() else []
            if not image_files:
                st.info("No generated images found in the images/ directory.")
            else:
                image_cols = st.columns(2)
                for index, image_path in enumerate(image_files):
                    with image_cols[index % 2]:
                        st.image(str(image_path), caption=image_path.name, width="stretch")
                        st.caption("Generated visual")
            if image_specs:
                with st.expander("Image plan metadata"):
                    st.json(image_specs)

        with article_tabs[5]:
            logs = st.session_state.get("last_logs", [])
            if logs:
                st.text_area("Graph events", value="\n\n".join(logs[-80:]), height=420)
            else:
                st.info("No event log was saved for this result.")

    with right:
        st.markdown('<div class="bp-card"><div class="bp-card-title">AI Generation</div>', unsafe_allow_html=True)
        # The run has completed if a result exists. Display stages conservatively;
        # only claim research when the output indicates it was used.
        statuses = {
            "Router": "completed",
            "Web Research": "completed" if (isinstance(out, dict) and (out.get("needs_research") or evidence)) else "pending",
            "Outline": "completed" if plan_obj else "pending",
            "Writing": "completed" if final_md else "pending",
            "Content Reduction": "completed" if final_md else "pending",
            "Visual Generation": "completed" if image_specs or (IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir())) else "pending",
        }
        render_workflow(statuses)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="bp-card"><div class="bp-card-title">Generated visuals</div>', unsafe_allow_html=True)
        image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file()]) if IMAGES_DIR.exists() else []
        if not image_files:
            st.markdown('<div class="bp-muted">Images will appear here when the blog workflow generates them.</div>', unsafe_allow_html=True)
        else:
            for image_path in image_files[:4]:
                st.image(str(image_path), width="stretch")
                st.markdown(f'<div class="bp-visual-label">{image_path.stem.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                st.markdown('<div class="bp-small">Generated technical visual</div>', unsafe_allow_html=True)
                st.markdown("<hr class='bp-divider'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="bp-card"><div class="bp-card-title">At a glance</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Sections", len((plan_obj or {}).get("tasks", [])) if isinstance(plan_obj, dict) else "—")
        m2.metric("Images", len(image_files))
        st.markdown('<div class="bp-muted">Built with LangGraph, Gemini, Tavily, Pollinations AI and PIL text overlay.</div></div>', unsafe_allow_html=True)

else:
    # Helpful first-run empty state
    st.markdown(
        """
        <div class="bp-card" style="text-align:center;padding:2.5rem 1.5rem 2.2rem">
          <div style="font-size:2.5rem;margin-bottom:.4rem">✦</div>
          <div style="font:700 1.55rem 'Space Grotesk',sans-serif;color:#f4f6ff">Create your next technical blog</div>
          <div class="bp-muted" style="max-width:560px;margin:.55rem auto 1.2rem">
            Give BlogBolt AI a topic and let your agentic workflow plan, write, research when needed, and create supporting visuals.
          </div>
          <div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap">
            <span class="bp-status">⌕ Research</span>
            <span class="bp-status">✎ Structured writing</span>
            <span class="bp-status">▧ AI visuals</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Need an idea? Try one of these topics:**")
    ideas = [
        "How AI Agents Actually Work: From LLM Reasoning to Tool Use and Memory",
        "RAG vs Fine-Tuning: Choosing the Right Approach for LLM Applications",
        "How Transformers Understand Text: Tokens, Embeddings, and Self-Attention Explained",
    ]
    idea_cols = st.columns(3)
    for idx, idea in enumerate(ideas):
        with idea_cols[idx]:
            if st.button(idea, key=f"idea_{idx}", width="stretch"):
                st.session_state["topic"] = idea
                st.rerun()

st.markdown(
    '<div style="text-align:center;color:#7083a7;font-size:.75rem;padding:1.3rem 0 .2rem">'
    'BlogBolt AI · Research → Write → Generate visuals → Export'
    '</div>',
    unsafe_allow_html=True,
)
