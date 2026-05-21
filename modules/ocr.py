# modules/ocr.py
"""
OCR module: parse medical report images or PDFs via ZhipuAI Vision API.
Supports uploading 1-5 reports (one per year), returns a list of dicts.
"""
import base64
import json
import re
import io
from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt sent to the vision model
# ---------------------------------------------------------------------------
_EXTRACTION_PROMPT = """You are a medical data extraction assistant.
From the medical examination report image provided, extract the following fields ONLY if clearly present.
Return a valid JSON object with these keys (use null for missing values):
{
  "age":    <integer or null>,
  "gender": <"Male" | "Female" | null>,
  "alb":    <float, Albumin in g/L, or null>,
  "bmi":    <float, Body Mass Index, or null>,
  "hdl":    <float, HDL Cholesterol in mmol/L, or null>,
  "sbp":    <float, Systolic Blood Pressure in mmHg, or null>,
  "dbp":    <float, Diastolic Blood Pressure in mmHg, or null>,
  "scr":    <float, Serum Creatinine in μmol/L, or null>
}
Return ONLY the JSON object. Do not add explanation or markdown fences."""


def _image_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _pdf_first_page_to_image(pdf_bytes: bytes) -> bytes:
    """Convert the first page of a PDF to PNG bytes using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")


def _call_vision_api(image_b64: str, api_key: str) -> dict:
    """Send image to ZhipuAI GLM-4V and return extracted fields dict."""
    client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def extract_from_file(uploaded_file, api_key: str) -> dict:
    """
    Extract from a single Streamlit UploadedFile.
    Returns a dict with extracted values (None for missing fields).
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        image_bytes = _pdf_first_page_to_image(file_bytes)
    else:
        image_bytes = file_bytes

    image_b64 = _image_bytes_to_base64(image_bytes)
    return _call_vision_api(image_b64, api_key)


def extract_from_files(uploaded_files: list, api_key: str) -> list:
    """
    Extract from multiple Streamlit UploadedFile objects (1-5 files).
    Files should be ordered from oldest to newest (Y1 → Y5).
    Returns a list of dicts, one per file, in the same order.
    """
    results = []
    for f in uploaded_files:
        try:
            results.append(extract_from_file(f, api_key))
        except Exception as e:
            results.append({
                "age": None, "gender": None,
                "alb": None, "bmi": None, "hdl": None,
                "sbp": None, "dbp": None, "scr": None,
                "_error": str(e),
            })
    return results
