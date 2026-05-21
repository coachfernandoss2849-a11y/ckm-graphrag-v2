# modules/science.py
"""
Health education resource module:
- Load asset index
- Recommend resources based on patient risk profile
- Render video player (HTML5 + subtitle track + transcript)
- Render Lottie animation cards
"""
import base64
import json
import pathlib
import re
import streamlit as st
import streamlit.components.v1 as components

_INDEX_PATH = pathlib.Path(__file__).parent.parent / "assets" / "index.json"
_ASSETS     = pathlib.Path(__file__).parent.parent / "assets"


def load_index() -> dict:
    """Load assets/index.json. Returns empty structure if file missing."""
    if _INDEX_PATH.exists():
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    return {"videos": [], "lottie": []}


def recommend_resources(risk_level: str, egfr_slope: float | None,
                        map_slope: float | None, index: dict) -> list:
    seen = set()
    recs = []

    def _add(item, kind):
        key = (kind, item["id"])
        if key not in seen:
            seen.add(key)
            recs.append({**item, "type": kind})

    active_tags = set()
    if egfr_slope is not None and egfr_slope < -3:
        active_tags.update(["kidney", "egfr"])
    if map_slope is not None and map_slope > 2:
        active_tags.add("hypertension")
    if risk_level in ("High", "Very High"):
        active_tags.add("high_risk")
    if not active_tags:
        active_tags.add("ckm")

    for v in index.get("videos", []):
        if active_tags & set(v.get("tags", [])):
            _add(v, "video")
    for l in index.get("lottie", []):
        if active_tags & set(l.get("tags", [])):
            _add(l, "lottie")

    if not recs:
        for v in index.get("videos", []):
            if "overview" in v.get("tags", []) or v["id"] == "ckm_overview":
                _add(v, "video")
                break

    return recs


def _b64(path: pathlib.Path, mime: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _vtt_to_transcript(vtt_text: str) -> str:
    """Strip VTT timestamps and return plain text."""
    lines = []
    for line in vtt_text.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT":
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"[\d:.,]+ --> [\d:.,]+", line):
            continue
        lines.append(line)
    return " ".join(lines)


def render_video_card(video: dict, view_mode: str, key_prefix: str = "") -> None:
    title      = video["title_en"]
    video_path = _ASSETS / "videos" / video["filename"]
    en_vtt     = _ASSETS / video.get("subtitle_en", f"subtitles/{video['id']}_en.vtt")

    if not video_path.exists():
        st.markdown(f"""
        <div style='background:#F8F9FA;border:1px dashed #BDC3C7;border-radius:8px;
                    padding:24px;text-align:center;color:#7F8C8D;'>
          <div style='font-size:32px;margin-bottom:8px;'>🎬</div>
          <div style='font-weight:600;font-size:14px;'>{title}</div>
          <div style='font-size:12px;margin-top:6px;'>
            Video not yet downloaded. Run <code>python download_assets.py</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**{title}**")

    play_key = f"{key_prefix}playing_{video['id']}"
    if not st.session_state.get(play_key, False):
        thumb_path = _ASSETS / "thumbnails" / f"{video['id']}.jpg"
        if thumb_path.exists():
            col_t, col_i = st.columns([1, 2])
            with col_t:
                st.image(str(thumb_path), use_container_width=True)
            with col_i:
                if st.button("▶ Play", key=f"{key_prefix}play_{video['id']}"):
                    st.session_state[play_key] = True
        else:
            if st.button("▶ Play", key=f"{key_prefix}play_{video['id']}"):
                st.session_state[play_key] = True
        return

    # Build HTML5 player with embedded video + subtitle track
    vid_src = _b64(video_path, "video/mp4")

    track_tag = ""
    transcript_text = ""
    if en_vtt.exists():
        vtt_src = _b64(en_vtt, "text/vtt")
        track_tag = (
            f'<track kind="subtitles" src="{vtt_src}" '
            f'srclang="en" label="English" default>'
        )
        transcript_text = _vtt_to_transcript(en_vtt.read_text(encoding="utf-8"))

    components.html(f"""
    <video controls style="width:100%;border-radius:8px;" crossorigin="anonymous">
      <source src="{vid_src}" type="video/mp4">
      {track_tag}
    </video>
    """, height=320)

    if transcript_text:
        with st.expander("📄 English Transcript"):
            st.markdown(f'<p style="font-size:13px;line-height:1.7;">{transcript_text}</p>',
                        unsafe_allow_html=True)
    elif en_vtt.exists() is False:
        st.caption("Subtitles not yet generated. Run `python download_assets.py`.")


def render_lottie_card(lottie_item: dict, view_mode: str, key_prefix: str = "") -> None:
    name = lottie_item["name_en"]
    fname = lottie_item["filename_en"]
    lottie_path = _ASSETS / "lottie" / fname

    if not lottie_path.exists():
        lottie_path = _ASSETS / "lottie" / lottie_item["filename"]

    if lottie_path.exists():
        try:
            from streamlit_lottie import st_lottie
            animation = json.loads(lottie_path.read_text(encoding="utf-8"))
            st.markdown(f"**{name}**")
            st_lottie(animation, height=180, key=f"{key_prefix}lottie_{lottie_item['id']}")
        except Exception:
            _lottie_placeholder(name)
    else:
        _lottie_placeholder(name)


def _lottie_placeholder(name: str) -> None:
    st.markdown(f"""
    <div style='background:#F0F4FF;border:1px dashed #A9C4F5;border-radius:8px;
                padding:20px;text-align:center;color:#5D7EC7;'>
      <div style='font-size:28px;margin-bottom:6px;'>🎞️</div>
      <div style='font-weight:600;font-size:13px;'>{name}</div>
      <div style='font-size:11px;margin-top:4px;'>
        Animation not yet downloaded. Run <code>python download_assets.py</code>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_resource_grid(recs: list, view_mode: str, key_prefix: str = "") -> None:
    """Render recommended resources in a 2-column grid."""
    if not recs:
        st.info("No specific resources matched this patient's profile. "
                "Browse all resources below.")
        return

    cols = st.columns(2)
    for i, item in enumerate(recs):
        with cols[i % 2]:
            with st.container():
                st.markdown("""
                <div style='background:white;border-radius:10px;padding:12px;
                            box-shadow:0 1px 6px rgba(0,0,0,0.08);margin-bottom:12px;'>
                """, unsafe_allow_html=True)
                if item["type"] == "video":
                    render_video_card(item, view_mode, key_prefix=key_prefix)
                else:
                    render_lottie_card(item, view_mode, key_prefix=key_prefix)
                st.markdown("</div>", unsafe_allow_html=True)

