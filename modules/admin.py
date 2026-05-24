# modules/admin.py
"""
Admin panel helpers:
- Password gate
- Video upload + Whisper transcription + index update
- PDF/TXT upload + ChromaDB rebuild
- Video entry management (delete / edit title / tags)
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import streamlit as st

_ASSETS       = pathlib.Path(__file__).parent.parent / "assets"
_INDEX_PATH   = _ASSETS / "index.json"
_KB_PATH      = pathlib.Path(__file__).parent.parent / "knowledge_base"
_CHROMA_PATH  = pathlib.Path(__file__).parent.parent / "chroma_db"
_VIDEOS_DIR   = _ASSETS / "videos"
_THUMBS_DIR   = _ASSETS / "thumbnails"
_SUBS_DIR     = _ASSETS / "subtitles"

ADMIN_PASSWORD = os.environ.get("CKM_ADMIN_PASSWORD", "ckm2026")

# ── Auth ──────────────────────────────────────────────────────────────────────

def check_auth() -> bool:
    if st.session_state.get("admin_authed"):
        return True
    pwd = st.text_input("Admin password", type="password", key="admin_pwd_input")
    if st.button("Login", key="admin_login_btn"):
        if pwd == ADMIN_PASSWORD:
            st.session_state["admin_authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


# ── Index helpers ─────────────────────────────────────────────────────────────

def _load_index() -> dict:
    if _INDEX_PATH.exists():
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    return {"videos": [], "lottie": []}


def _save_index(index: dict):
    _INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Whisper transcription ─────────────────────────────────────────────────────

def _extract_pcm(video_path: pathlib.Path) -> bytes | None:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        result = subprocess.run([
            ffmpeg_exe, "-i", str(video_path),
            "-f", "s16le", "-ac", "1", "-ar", "16000",
            "-loglevel", "error", "-",
        ], check=True, capture_output=True)
        return result.stdout
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        return None


def _transcribe(video_path: pathlib.Path, video_id: str) -> tuple[str, str]:
    """Run Whisper on video, return (zh_vtt_rel, en_vtt_rel)."""
    import whisper

    zh_vtt = _SUBS_DIR / f"{video_id}_zh.vtt"
    en_vtt = _SUBS_DIR / f"{video_id}_en.vtt"
    rel_zh = f"subtitles/{video_id}_zh.vtt"
    rel_en = f"subtitles/{video_id}_en.vtt"

    _SUBS_DIR.mkdir(parents=True, exist_ok=True)

    pcm = _extract_pcm(video_path)
    if not pcm:
        return rel_zh, rel_en

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    model = whisper.load_model("small")

    zh_result = model.transcribe(audio, language="zh", task="transcribe")
    en_result = model.transcribe(audio, language="zh", task="translate")

    def _segs_to_vtt(segs) -> str:
        def fmt(t):
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int((s % 1)*1000):03d}"
        lines = ["WEBVTT", ""]
        for i, seg in enumerate(segs, 1):
            lines += [str(i), f"{fmt(seg['start'])} --> {fmt(seg['end'])}", seg["text"].strip(), ""]
        return "\n".join(lines)

    zh_vtt.write_text(_segs_to_vtt(zh_result["segments"]), encoding="utf-8")
    en_vtt.write_text(_segs_to_vtt(en_result["segments"]), encoding="utf-8")
    return rel_zh, rel_en


# ── Video upload section ──────────────────────────────────────────────────────

def render_video_upload():
    st.markdown("#### Upload New Video")
    uploaded = st.file_uploader("MP4 file", type=["mp4"], key="admin_video_upload")
    if not uploaded:
        return

    col1, col2 = st.columns(2)
    with col1:
        vid_id    = st.text_input("Video ID (no spaces, e.g. sglt2_new)", key="admin_vid_id")
        title_en  = st.text_input("English title", key="admin_title_en")
    with col2:
        title_zh  = st.text_input("Chinese title", key="admin_title_zh")
        tags_raw  = st.text_input("Tags (comma-separated)", value="ckm", key="admin_tags")

    run_whisper = st.checkbox("Generate subtitles with Whisper (slow on CPU)", value=True, key="admin_whisper")

    if st.button("Save & Process", key="admin_save_video"):
        if not vid_id or not title_en:
            st.error("Video ID and English title are required.")
            return

        _VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        _THUMBS_DIR.mkdir(parents=True, exist_ok=True)

        video_path = _VIDEOS_DIR / f"{vid_id}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved {video_path.name}")

        rel_zh = f"subtitles/{vid_id}_zh.vtt"
        rel_en = f"subtitles/{vid_id}_en.vtt"

        if run_whisper:
            with st.spinner("Transcribing & translating (this may take several minutes)..."):
                rel_zh, rel_en = _transcribe(video_path, vid_id)
            st.success("Subtitles generated.")

        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        index = _load_index()

        # Remove existing entry with same id
        index["videos"] = [v for v in index["videos"] if v["id"] != vid_id]
        index["videos"].append({
            "id":          vid_id,
            "title_zh":    title_zh or title_en,
            "title_en":    title_en,
            "filename":    f"{vid_id}.mp4",
            "subtitle_zh": rel_zh,
            "subtitle_en": rel_en,
            "thumbnail":   f"thumbnails/{vid_id}.jpg",
            "tags":        tags,
        })
        _save_index(index)
        st.success("index.json updated. Refresh Tab 4 to see the new video.")


# ── Knowledge base upload section ─────────────────────────────────────────────

def render_kb_upload():
    st.markdown("#### Upload Literature / Guidelines")
    st.caption("Accepts .txt or .pdf files. After upload, ChromaDB will be rebuilt automatically.")

    files = st.file_uploader(
        "Select files", type=["txt", "pdf"],
        accept_multiple_files=True, key="admin_kb_upload"
    )
    if not files:
        return

    if st.button("Save & Rebuild Knowledge Base", key="admin_rebuild_kb"):
        _KB_PATH.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            dest = _KB_PATH / f.name
            if f.name.endswith(".pdf"):
                # Extract text from PDF
                try:
                    import fitz  # PyMuPDF
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    doc = fitz.open(tmp_path)
                    text = "\n".join(page.get_text() for page in doc)
                    doc.close()
                    os.unlink(tmp_path)
                    txt_dest = _KB_PATH / (pathlib.Path(f.name).stem + ".txt")
                    txt_dest.write_text(text, encoding="utf-8")
                    saved.append(txt_dest.name)
                except Exception as e:
                    st.warning(f"PDF extraction failed for {f.name}: {e} — saving raw bytes.")
                    dest.write_bytes(f.read())
                    saved.append(dest.name)
            else:
                dest.write_bytes(f.read())
                saved.append(dest.name)

        st.success(f"Saved: {', '.join(saved)}")

        with st.spinner("Rebuilding ChromaDB..."):
            _rebuild_chroma()
        st.success("Knowledge base rebuilt successfully.")


def _rebuild_chroma():
    import chromadb
    import requests
    import os

    if _CHROMA_PATH.exists():
        shutil.rmtree(str(_CHROMA_PATH))

    # Load all .txt files manually
    docs = []
    if _KB_PATH.exists():
        for txt_file in _KB_PATH.rglob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                docs.append({"text": text, "source": str(txt_file)})
            except Exception:
                pass

    if not docs:
        return

    # Chunk documents
    def _chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # Build embeddings via ZhipuAI API
    api_key = os.environ.get("ZHIPUAI_API_KEY", "")

    def _embed(text):
        try:
            resp = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "embedding-3", "input": text[:2048]},
                timeout=30,
            )
            return resp.json()["data"][0]["embedding"]
        except Exception:
            return [0.0] * 2048

    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = client.get_or_create_collection("langchain")

    batch_texts, batch_metas, batch_ids = [], [], []
    idx = 0
    for doc in docs:
        for chunk in _chunk_text(doc["text"]):
            batch_texts.append(chunk)
            batch_metas.append({"source": doc["source"]})
            batch_ids.append(f"doc_{idx}")
            idx += 1

    # Add in small batches to avoid timeout
    batch_size = 20
    for i in range(0, len(batch_texts), batch_size):
        b_texts = batch_texts[i:i+batch_size]
        b_metas = batch_metas[i:i+batch_size]
        b_ids = batch_ids[i:i+batch_size]
        b_embs = [_embed(t) for t in b_texts]
        collection.add(documents=b_texts, embeddings=b_embs,
                       metadatas=b_metas, ids=b_ids)


# ── Video management section ──────────────────────────────────────────────────

def render_video_management():
    st.markdown("#### Manage Existing Videos")
    index = _load_index()
    videos = index.get("videos", [])

    if not videos:
        st.info("No videos in index.")
        return

    for i, v in enumerate(videos):
        with st.expander(f"{v['title_en']}  `{v['id']}`"):
            col1, col2 = st.columns(2)
            with col1:
                new_title_en = st.text_input("English title", value=v["title_en"],
                                             key=f"mgmt_en_{v['id']}")
                new_title_zh = st.text_input("Chinese title", value=v.get("title_zh", ""),
                                             key=f"mgmt_zh_{v['id']}")
            with col2:
                new_tags = st.text_input("Tags", value=", ".join(v.get("tags", [])),
                                         key=f"mgmt_tags_{v['id']}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save changes", key=f"mgmt_save_{v['id']}"):
                    index["videos"][i]["title_en"] = new_title_en
                    index["videos"][i]["title_zh"] = new_title_zh
                    index["videos"][i]["tags"] = [t.strip() for t in new_tags.split(",") if t.strip()]
                    _save_index(index)
                    st.success("Saved.")
                    st.rerun()
            with c2:
                if st.button("🗑 Delete", key=f"mgmt_del_{v['id']}"):
                    index["videos"] = [x for x in index["videos"] if x["id"] != v["id"]]
                    _save_index(index)
                    st.warning(f"Removed {v['id']} from index. Video file not deleted.")
                    st.rerun()
