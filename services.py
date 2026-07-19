import base64
import csv
import importlib.util
import io
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree
import zipfile

import requests
from supabase import create_client


# ────────────────────────────────────────────────────────────────
# Supabase client (replaces local SQLite + local disk uploads)
# ────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()  # use the service_role key here
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "caspam-files").strip()

_supabase_client = None


def sb():
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_KEY are not set. Add them in Render's Environment tab."
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ────────────────────────────────────────────────────────────────
# OCR.space (replaces local pytesseract/tesseract binary, which
# Render's native Python environment cannot install via Aptfile)
# ────────────────────────────────────────────────────────────────
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "").strip()
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".mov", ".avi", ".mkv",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DOCUMENT_EXTENSIONS = ALLOWED_EXTENSIONS - IMAGE_EXTENSIONS - AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
CHUNK_CHARS = int(os.getenv("DOCUMENT_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "220"))
MAX_RAG_CHUNKS = int(os.getenv("MAX_RAG_CHUNKS", "10"))
LATEST_PROMPT_VERSION = os.getenv("SYSTEM_PROMPT_VERSION", "2026-07-10")
GROQ_AUDIO_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "25"))
VISION_IMAGE_MAX_BYTES = int(os.getenv("VISION_IMAGE_MAX_BYTES", str(900_000)))
VISION_IMAGE_MAX_DIMENSION = int(os.getenv("VISION_IMAGE_MAX_DIMENSION", "1000"))
PDF_VISION_MAX_PAGES = int(os.getenv("PDF_VISION_MAX_PAGES", "2"))


class TextOnlyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data and data.strip():
            self.parts.append(data.strip())

    def text(self):
        return " ".join(self.parts)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def dependency_status():
    return {
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "ocr_space_configured": bool(OCR_SPACE_API_KEY),
        "pymupdf": bool(importlib.util.find_spec("fitz")),
        "pillow": bool(importlib.util.find_spec("PIL")),
    }


def db():
    """Backward-compatible alias so `from services import db` still works."""
    return sb()


def init_db():
    """Tables live in Supabase now (see supabase_schema.sql). This just
    verifies the connection works and logs a clear warning if not."""
    try:
        sb().table("profiles").select("user_id").limit(1).execute()
    except Exception as exc:
        print(f"[services] Supabase connection check failed: {exc}", flush=True)


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def clean_env_value(value):
    value = (value or "").strip().strip("\"'")
    if "=" in value:
        value = value.split("=", 1)[1].strip().strip("\"'")
    return value


def get_user_id(request):
    return (
        request.headers.get("X-User-Id")
        or request.headers.get("X-Firebase-Uid")
        or "local"
    )[:128]


def get_user_role(request):
    role = (request.headers.get("X-User-Role") or "student").lower()
    return role if role in {"student", "teacher", "admin"} else "student"


def log_activity(user_id, action, details=None):
    try:
        sb().table("activity_logs").insert(
            {
                "id": uuid.uuid4().hex,
                "user_id": user_id,
                "action": action,
                "details": json.dumps(details or {}),
                "created_at": utc_now(),
            }
        ).execute()
    except Exception as exc:
        print(f"[services] log_activity failed: {exc}", flush=True)


def upsert_profile(user_id, email="", display_name="", role="student", preferences=None):
    role = role if role in {"student", "teacher", "admin"} else "student"
    sb().table("profiles").upsert(
        {
            "user_id": user_id,
            "email": email,
            "display_name": display_name,
            "role": role,
            "preferences": json.dumps(preferences or {}),
            "updated_at": utc_now(),
        }
    ).execute()
    log_activity(user_id, "profile.upsert", {"role": role})


def get_profile(user_id):
    res = sb().table("profiles").select("*").eq("user_id", user_id).limit(1).execute()
    rows = res.data or []
    if not rows:
        return {"user_id": user_id, "role": "student", "preferences": {}}
    data = dict(rows[0])
    prefs = data.get("preferences") or "{}"
    data["preferences"] = json.loads(prefs) if isinstance(prefs, str) else (prefs or {})
    return data


def allowed_file(filename):
    return Path(filename or "").suffix.lower() in ALLOWED_EXTENSIONS


def safe_filename(filename):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).name).strip("._") or "upload"
    return stem[:120]


def extract_txt(raw):
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def make_segment(text, locator_type="document", locator_label="Document", page_number=None):
    text = clean_text(text)
    if not text:
        return None
    return {
        "text": text,
        "locator_type": locator_type,
        "locator_label": locator_label,
        "page_number": page_number,
    }


def extract_docx(raw):
    text = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in sorted(n for n in zf.namelist() if n.startswith("word/") and n.endswith(".xml")):
            root = ElementTree.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    text.append(node.text)
    return " ".join(text)


def extract_docx_segments(raw):
    return [seg for seg in [make_segment(extract_docx(raw), "document", "Document")] if seg]


def extract_pptx(raw):
    text = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in sorted(n for n in zf.namelist() if n.startswith("ppt/slides/") and n.endswith(".xml")):
            root = ElementTree.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    text.append(node.text)
    return " ".join(text)


def extract_pptx_segments(raw):
    segments = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        slide_names = sorted(n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
        for index, name in enumerate(slide_names, start=1):
            text = []
            root = ElementTree.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    text.append(node.text)
            seg = make_segment(" ".join(text), "slide", f"Slide {index}", index)
            if seg:
                segments.append(seg)
    return segments


def extract_xlsx(raw):
    values = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.iter():
                if item.tag.endswith("}t") and item.text:
                    shared.append(item.text)
        for name in sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")):
            root = ElementTree.fromstring(zf.read(name))
            for cell in root.iter():
                if not cell.tag.endswith("}c"):
                    continue
                cell_type = cell.attrib.get("t")
                raw_value = next((child.text for child in cell if child.tag.endswith("}v") and child.text), "")
                inline_value = " ".join(child.text for child in cell.iter() if child.tag.endswith("}t") and child.text)
                if cell_type == "s" and raw_value.isdigit() and int(raw_value) < len(shared):
                    values.append(shared[int(raw_value)])
                elif inline_value:
                    values.append(inline_value)
                elif raw_value:
                    values.append(raw_value)
    return " ".join(values)


def extract_xlsx_segments(raw):
    segments = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.iter():
                if item.tag.endswith("}t") and item.text:
                    shared.append(item.text)
        sheet_names = sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml"))
        for index, name in enumerate(sheet_names, start=1):
            values = []
            root = ElementTree.fromstring(zf.read(name))
            for cell in root.iter():
                if not cell.tag.endswith("}c"):
                    continue
                cell_type = cell.attrib.get("t")
                raw_value = next((child.text for child in cell if child.tag.endswith("}v") and child.text), "")
                inline_value = " ".join(child.text for child in cell.iter() if child.tag.endswith("}t") and child.text)
                if cell_type == "s" and raw_value.isdigit() and int(raw_value) < len(shared):
                    values.append(shared[int(raw_value)])
                elif inline_value:
                    values.append(inline_value)
                elif raw_value:
                    values.append(raw_value)
            seg = make_segment(" ".join(values), "sheet", f"Sheet {index}", index)
            if seg:
                segments.append(seg)
    return segments


def extract_csv(raw):
    text = extract_txt(raw)
    rows = csv.reader(io.StringIO(text))
    return " ".join(" ".join(cell for cell in row if cell) for row in rows)


def extract_csv_segments(raw):
    return [seg for seg in [make_segment(extract_csv(raw), "sheet", "CSV")] if seg]


def extract_pdf(raw):
    return "\n".join(segment["text"] for segment in extract_pdf_segments(raw))


def extract_pdf_segments(raw):
    segments = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        for index, page in enumerate(reader.pages, start=1):
            seg = make_segment(page.extract_text() or "", "page", f"Page {index}", index)
            if seg:
                segments.append(seg)
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open(stream=raw, filetype="pdf")
        existing_pages = {segment["page_number"] for segment in segments if segment.get("page_number")}
        for index, page in enumerate(doc, start=1):
            if index in existing_pages:
                continue
            seg = make_segment(page.get_text("text") or "", "page", f"Page {index}", index)
            if seg:
                segments.append(seg)
                existing_pages.add(index)

        if len(existing_pages) < len(doc):
            try:
                from PIL import Image

                for index, page in enumerate(doc, start=1):
                    if index in existing_pages or index > PDF_OCR_MAX_PAGES:
                        continue
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = ocr_image_text(image)
                    seg = make_segment(text, "page", f"Page {index} OCR", index)
                    if seg:
                        segments.append(seg)
                        existing_pages.add(index)
            except Exception:
                pass
    except Exception:
        pass

    segments.sort(key=lambda segment: segment.get("page_number") or 0)
    return segments


def ocr_image_bytes_via_api(image_bytes, filename="image.png", language="eng"):
    """Cloud OCR via OCR.space — works on Render's native Python runtime
    without needing the `tesseract` system binary (which Aptfile cannot
    install there; Aptfile is a Heroku convention, not a Render one)."""
    if not OCR_SPACE_API_KEY:
        return ""
    try:
        response = requests.post(
            OCR_SPACE_URL,
            files={"file": (filename, image_bytes)},
            data={
                "apikey": OCR_SPACE_API_KEY,
                "language": language,
                "OCREngine": 2,
                "scale": "true",
                "isOverlayRequired": "false",
            },
            timeout=60,
        )
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            return ""
        parsed = result.get("ParsedResults") or []
        return clean_text(" ".join(p.get("ParsedText", "") for p in parsed))
    except Exception:
        return ""


def ocr_image_text(image):
    """`image` is a PIL Image object (kept for compatibility with existing
    callers); we re-encode it and send it to the OCR.space API."""
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return ocr_image_bytes_via_api(buf.getvalue())
    except Exception:
        return ""


def extract_image(raw):
    try:
        from PIL import Image
    except Exception:
        return ""
    image = Image.open(io.BytesIO(raw))
    return ocr_image_text(image)


def extract_image_segments(raw):
    return [seg for seg in [make_segment(extract_image(raw), "image", "Image OCR")] if seg]


def extract_text_from_upload(filename, raw):
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            return clean_text(extract_txt(raw))
        if ext == ".docx":
            return clean_text(extract_docx(raw))
        if ext == ".pptx":
            return clean_text(extract_pptx(raw))
        if ext == ".xlsx":
            return clean_text(extract_xlsx(raw))
        if ext == ".csv":
            return clean_text(extract_csv(raw))
        if ext == ".pdf":
            return clean_text(extract_pdf(raw))
        if ext in IMAGE_EXTENSIONS:
            return clean_text(extract_image(raw))
    except Exception:
        return ""
    return ""


def extract_segments_from_upload(filename, raw):
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            return [seg for seg in [make_segment(extract_txt(raw), "document", "Text file")] if seg]
        if ext == ".docx":
            return extract_docx_segments(raw)
        if ext == ".pptx":
            return extract_pptx_segments(raw)
        if ext == ".xlsx":
            return extract_xlsx_segments(raw)
        if ext == ".csv":
            return extract_csv_segments(raw)
        if ext == ".pdf":
            return extract_pdf_segments(raw)
        if ext in IMAGE_EXTENSIONS:
            return extract_image_segments(raw)
    except Exception:
        return []
    text = extract_text_from_upload(filename, raw)
    return [seg for seg in [make_segment(text)] if seg]


def sparse_embedding(text, limit=160):
    words = Counter(tokenize(text))
    if not words:
        return {}
    total = sum(words.values()) or 1
    return {word: round(count / total, 6) for word, count in words.most_common(limit)}


def sparse_similarity(query_embedding, chunk_embedding):
    if not query_embedding or not chunk_embedding:
        return 0.0
    dot = sum(weight * chunk_embedding.get(word, 0) for word, weight in query_embedding.items())
    q_norm = sum(weight * weight for weight in query_embedding.values()) ** 0.5
    c_norm = sum(weight * weight for weight in chunk_embedding.values()) ** 0.5
    if not q_norm or not c_norm:
        return 0.0
    return dot / (q_norm * c_norm)


def chunk_segments(segments, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    chunks = []
    for segment in segments:
        text = segment.get("text", "")
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "locator_type": segment.get("locator_type") or "document",
                        "locator_label": segment.get("locator_label") or "Document",
                        "page_number": segment.get("page_number"),
                        "token_count": len(tokenize(chunk_text)),
                        "embedding": sparse_embedding(chunk_text),
                    }
                )
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def transcribe_media_upload(filename, raw, content_type):
    api_key = clean_env_value(os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))
    if not api_key:
        return ""
    try:
        response = requests.post(
            GROQ_AUDIO_TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": GROQ_TRANSCRIPTION_MODEL},
            files={"file": (filename, io.BytesIO(raw), content_type or "application/octet-stream")},
            timeout=90,
        )
        if not response.ok:
            return ""
        data = response.json()
        return clean_text(data.get("text", ""))
    except Exception:
        return ""


def media_kind(filename):
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "document"


def describe_upload(filename, content_type, size_bytes, text):
    kind = media_kind(filename)
    if text:
        return f"{kind.title()} file with {len(text)} extracted text characters."
    if kind == "image":
        return "Image file uploaded. OCR text was not available, so answer using visible context only if the user describes it."
    if kind in {"audio", "video"}:
        return f"{kind.title()} file uploaded. No transcript is available yet; use the filename and any user transcript or description."
    return "Document uploaded, but no readable text was extracted."


# ────────────────────────────────────────────────────────────────
# Document storage — files now live in Supabase Storage (permanent),
# metadata in Supabase Postgres (permanent) instead of local SQLite.
# ────────────────────────────────────────────────────────────────
def save_document(user_id, file_storage, chat_id=""):
    filename = safe_filename(file_storage.filename or "upload")
    if not allowed_file(filename):
        raise ValueError("Unsupported file type.")
    raw = file_storage.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large.")

    doc_id = uuid.uuid4().hex
    storage_path = f"{user_id}/{doc_id}_{filename}"

    try:
        sb().storage.from_(SUPABASE_BUCKET).upload(
            storage_path,
            raw,
            {"content-type": file_storage.mimetype or "application/octet-stream"},
        )
    except Exception as exc:
        raise ValueError(f"Could not save file to Supabase storage: {exc}")

    kind = media_kind(filename)
    segments = extract_segments_from_upload(filename, raw)
    text = clean_text("\n\n".join(segment["text"] for segment in segments))
    if not text and kind in {"audio", "video"}:
        text = transcribe_media_upload(filename, raw, file_storage.mimetype)
        segments = [seg for seg in [make_segment(text, kind, f"{kind.title()} transcript")] if seg]
    chunks = chunk_segments(segments)
    summary = describe_upload(filename, file_storage.mimetype, len(raw), text)
    now = utc_now()

    sb().table("documents").insert(
        {
            "id": doc_id,
            "owner_id": user_id,
            "chat_id": clean_text(chat_id)[:128],
            "filename": filename,
            "content_type": file_storage.mimetype,
            "path": storage_path,
            "text": text,
            "media_kind": kind,
            "size_bytes": len(raw),
            "summary": summary,
            "created_at": now,
        }
    ).execute()

    if chunks:
        chunk_rows = [
            {
                "id": uuid.uuid4().hex,
                "document_id": doc_id,
                "owner_id": user_id,
                "chat_id": clean_text(chat_id)[:128],
                "filename": filename,
                "chunk_index": index,
                "locator_type": chunk["locator_type"],
                "locator_label": chunk["locator_label"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "embedding": json.dumps(chunk["embedding"], separators=(",", ":")),
                "token_count": chunk["token_count"],
                "created_at": now,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        sb().table("document_chunks").insert(chunk_rows).execute()

    log_activity(user_id, "document.upload", {"document_id": doc_id, "filename": filename, "chat_id": chat_id})
    return {
        "id": doc_id,
        "filename": filename,
        "content_type": file_storage.mimetype,
        "media_kind": kind,
        "size_bytes": len(raw),
        "summary": summary,
        "chat_id": clean_text(chat_id)[:128],
        "text_chars": len(text),
        "chunks": len(chunks),
        "ocr_used": Path(filename).suffix.lower() in IMAGE_EXTENSIONS,
    }


def list_documents(user_id, chat_id=None):
    query = sb().table("documents").select("*").in_("owner_id", [user_id, "shared"])
    if chat_id:
        query = query.or_(f"chat_id.eq.{chat_id},owner_id.eq.shared")
    docs = (query.order("created_at", desc=True).execute().data) or []
    if not docs:
        return []

    doc_ids = [d["id"] for d in docs]
    chunk_rows = (
        sb()
        .table("document_chunks")
        .select("document_id,page_number")
        .in_("document_id", doc_ids)
        .execute()
        .data
    ) or []
    counts = Counter(row["document_id"] for row in chunk_rows)
    max_pages = {}
    for row in chunk_rows:
        pn = row.get("page_number")
        if pn is not None:
            max_pages[row["document_id"]] = max(max_pages.get(row["document_id"], 0), pn)

    results = []
    for doc in docs:
        results.append(
            {
                "id": doc["id"],
                "chat_id": doc.get("chat_id"),
                "filename": doc.get("filename"),
                "content_type": doc.get("content_type"),
                "media_kind": doc.get("media_kind"),
                "size_bytes": doc.get("size_bytes"),
                "summary": doc.get("summary"),
                "text_chars": len(doc.get("text") or ""),
                "created_at": doc.get("created_at"),
                "chunks": counts.get(doc["id"], 0),
                "pages": max_pages.get(doc["id"]),
            }
        )
    return results


def prepare_vision_image(raw, content_type):
    if len(raw) <= VISION_IMAGE_MAX_BYTES and content_type in {"image/jpeg", "image/png", "image/webp"}:
        return raw, content_type or "image/jpeg"
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        image.thumbnail((VISION_IMAGE_MAX_DIMENSION, VISION_IMAGE_MAX_DIMENSION))
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        for quality in (85, 75, 65, 55):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            prepared = output.getvalue()
            if len(prepared) <= VISION_IMAGE_MAX_BYTES or quality == 55:
                return prepared, "image/jpeg"
    except Exception:
        pass
    return raw, content_type or "image/jpeg"


def recent_image_attachments(user_id, chat_id=None, limit=4):
    if not chat_id:
        return []
    rows = (
        sb()
        .table("documents")
        .select("id,filename,content_type,path,size_bytes,created_at")
        .in_("owner_id", [user_id, "shared"])
        .eq("chat_id", chat_id)
        .eq("media_kind", "image")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    ) or []
    images = []
    for row in rows:
        try:
            if (row.get("size_bytes") or 0) > MAX_UPLOAD_BYTES:
                continue
            raw = sb().storage.from_(SUPABASE_BUCKET).download(row["path"])
            raw, content_type = prepare_vision_image(raw, row.get("content_type") or "image/jpeg")
            if len(raw) > VISION_IMAGE_MAX_BYTES:
                continue
            images.append(
                {
                    "document_id": row["id"],
                    "filename": row["filename"],
                    "content_type": content_type,
                    "data": base64.b64encode(raw).decode("ascii"),
                    "size_bytes": row.get("size_bytes") or len(raw),
                }
            )
        except Exception:
            continue
    return images


def recent_pdf_page_attachments(user_id, chat_id=None, limit=1, max_pages=PDF_VISION_MAX_PAGES):
    if not chat_id:
        return []
    rows = (
        sb()
        .table("documents")
        .select("id,filename,content_type,path,size_bytes,created_at,text")
        .in_("owner_id", [user_id, "shared"])
        .eq("chat_id", chat_id)
        .eq("media_kind", "document")
        .ilike("filename", "%.pdf")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    ) or []
    pages = []
    try:
        import fitz
    except Exception:
        return pages
    for row in rows:
        try:
            if (row.get("size_bytes") or 0) > MAX_UPLOAD_BYTES:
                continue
            raw = sb().storage.from_(SUPABASE_BUCKET).download(row["path"])
            doc = fitz.open(stream=raw, filetype="pdf")
            for page_index, page in enumerate(doc, start=1):
                if page_index > max_pages:
                    break
                pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                page_raw, content_type = prepare_vision_image(pix.tobytes("png"), "image/png")
                if len(page_raw) > VISION_IMAGE_MAX_BYTES:
                    continue
                pages.append(
                    {
                        "document_id": row["id"],
                        "filename": f"{row['filename']} page {page_index}",
                        "content_type": content_type,
                        "data": base64.b64encode(page_raw).decode("ascii"),
                        "size_bytes": len(page_raw),
                        "page_number": page_index,
                        "source_filename": row["filename"],
                    }
                )
        except Exception:
            continue
    return pages


def refresh_document_text(user_id, document_id, chat_id=None):
    document_id = clean_text(document_id)[:128]
    if not document_id:
        return False
    rows = (
        sb()
        .table("documents")
        .select("*")
        .eq("id", document_id)
        .in_("owner_id", [user_id, "shared"])
        .limit(1)
        .execute()
        .data
    ) or []
    if not rows:
        return False
    row = rows[0]
    if chat_id and row.get("owner_id") != "shared" and clean_text(row.get("chat_id") or "") != clean_text(chat_id):
        return False

    try:
        raw = sb().storage.from_(SUPABASE_BUCKET).download(row["path"])
        if len(raw) > MAX_UPLOAD_BYTES:
            return False
        filename = row["filename"]
        kind = row.get("media_kind") or media_kind(filename)
        segments = extract_segments_from_upload(filename, raw)
        text = clean_text("\n\n".join(segment["text"] for segment in segments))
        if not text and kind in {"audio", "video"}:
            text = transcribe_media_upload(filename, raw, row.get("content_type"))
            segments = [seg for seg in [make_segment(text, kind, f"{kind.title()} transcript")] if seg]
        chunks = chunk_segments(segments)
        summary = describe_upload(filename, row.get("content_type"), len(raw), text)
        now = utc_now()

        sb().table("documents").update(
            {"text": text, "media_kind": kind, "size_bytes": len(raw), "summary": summary}
        ).eq("id", document_id).execute()

        sb().table("document_chunks").delete().eq("document_id", document_id).execute()
        if chunks:
            chunk_rows = [
                {
                    "id": uuid.uuid4().hex,
                    "document_id": document_id,
                    "owner_id": row.get("owner_id"),
                    "chat_id": row.get("chat_id"),
                    "filename": filename,
                    "chunk_index": index,
                    "locator_type": chunk["locator_type"],
                    "locator_label": chunk["locator_label"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "embedding": json.dumps(chunk["embedding"], separators=(",", ":")),
                    "token_count": chunk["token_count"],
                    "created_at": now,
                }
                for index, chunk in enumerate(chunks, start=1)
            ]
            sb().table("document_chunks").insert(chunk_rows).execute()
        return bool(text)
    except Exception:
        return False


def document_context_for_ids(user_id, chat_id, document_ids, limit=16):
    ids = [clean_text(doc_id)[:128] for doc_id in document_ids or [] if clean_text(doc_id)]
    if not ids:
        return "", []

    def fetch_docs():
        q = (
            sb()
            .table("documents")
            .select("id,filename,media_kind,summary,text,created_at,owner_id,chat_id")
            .in_("owner_id", [user_id, "shared"])
            .in_("id", ids)
        )
        if chat_id:
            q = q.or_(f"chat_id.eq.{chat_id},owner_id.eq.shared")
        return (q.order("created_at", desc=True).execute().data) or []

    docs = fetch_docs()
    for doc in docs:
        if not clean_text(doc.get("text") or ""):
            refresh_document_text(user_id, doc["id"], chat_id)
    docs = fetch_docs()

    q = sb().table("document_chunks").select("*").in_("owner_id", [user_id, "shared"]).in_("document_id", ids)
    if chat_id:
        q = q.or_(f"chat_id.eq.{chat_id},owner_id.eq.shared")
    chunks = (q.order("created_at", desc=True).limit(limit).execute().data) or []
    chunks.sort(key=lambda c: (c.get("document_id") or "", c.get("chunk_index") or 0))

    doc_lookup = {row["id"]: row for row in docs}
    sources = []
    parts = []
    if chunks:
        for data in chunks:
            doc = doc_lookup.get(data["document_id"], {})
            title = data.get("filename") or doc.get("filename") or "attached file"
            citation = chunk_citation(data)
            parts.append(
                f"Attached file: {title} ({citation}, chunk {data['chunk_index']})\n{data['text'][:CHUNK_CHARS]}"
            )
            sources.append(
                {
                    "document_id": data["document_id"],
                    "title": title,
                    "chunk": data["chunk_index"],
                    "page": data.get("page_number"),
                    "citation": citation,
                    "kind": doc.get("media_kind"),
                }
            )
    else:
        for doc in docs:
            text = clean_text(doc.get("text") or "")
            if not text:
                continue
            parts.append(f"Attached file: {doc['filename']}\n{text[:CHUNK_CHARS]}")
            sources.append({"document_id": doc["id"], "title": doc["filename"], "kind": doc.get("media_kind")})

    if not parts:
        lines = [
            f"- {doc['filename']} ({doc.get('media_kind') or 'file'}): no readable text or transcript could be extracted"
            for doc in docs
        ]
        return (
            "The user's latest message includes attached files, but the backend could not extract readable text "
            "or a transcript from them even after retrying extraction. Tell the user OCR/transcription failed for "
            "these files and ask for a clearer text-based file or pasted text.\n\n"
            + "\n".join(lines)
        ), [{"document_id": doc["id"], "title": doc["filename"], "kind": doc.get("media_kind")} for doc in docs]

    instruction = (
        "The user's latest message includes these attached file contents. "
        "For short prompts like 'explain', 'summarize', 'what is this', or 'tell me', answer from these attached files directly. "
        "Do not ask what to explain when attached content is available.\n\n"
    )
    return instruction + "\n\n---\n\n".join(parts), sources


def tokenize(text):
    return re.findall(r"[A-Za-z0-9\u0600-\u06FF]{3,}", (text or "").lower())


def score_text(query, text):
    q = Counter(tokenize(query))
    if not q:
        return 0
    words = Counter(tokenize(text))
    return sum(words.get(word, 0) * weight for word, weight in q.items())


def wants_document_answer(query):
    q = query.lower()
    return any(
        phrase in q
        for phrase in [
            "uploaded", "attached", "this pdf", "this document", "lecture note", "lecture notes",
            "my notes", "notes", "document", "pdf", "summarize", "summary", "chapter",
            "definition", "find", "from the file", "from this file",
        ]
    )


def load_chunk_embedding(raw):
    try:
        data = json.loads(raw or "{}")
        return {str(key): float(value) for key, value in data.items()}
    except Exception:
        return {}


def chunk_citation(chunk):
    label = chunk.get("locator_label") or "Document"
    if chunk.get("page_number") and chunk.get("locator_type") == "page":
        return f"page {chunk['page_number']}"
    return label


def best_document_chunks(user_id, query, chat_id=None, limit=MAX_RAG_CHUNKS):
    query_embedding = sparse_embedding(query)

    q = sb().table("document_chunks").select("*").in_("owner_id", [user_id, "shared"])
    if chat_id:
        q = q.or_(f"chat_id.eq.{chat_id},owner_id.eq.shared")
    chunk_rows = (q.execute().data) or []

    rows = []
    if not chunk_rows:
        qd = sb().table("documents").select("id,filename,text,created_at").in_("owner_id", [user_id, "shared"])
        if chat_id:
            qd = qd.or_(f"chat_id.eq.{chat_id},owner_id.eq.shared")
        docs = (qd.order("created_at", desc=True).execute().data) or []
        rows = [d for d in docs if clean_text(d.get("text") or "")]

    if chunk_rows:
        scored = []
        for data in chunk_rows:
            text = data.get("text") or ""
            lexical = score_text(query, text)
            semantic = sparse_similarity(query_embedding, load_chunk_embedding(data.get("embedding")))
            score = lexical + (semantic * 4)
            if score:
                scored.append((score, data))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            selected = [item[1] for item in scored[:limit]]
        elif wants_document_answer(query):
            selected = chunk_rows[:limit]
        else:
            selected = []
        return [
            {
                "document_id": row["document_id"],
                "title": row["filename"],
                "chunk": row["chunk_index"],
                "content": (row.get("text") or "")[:CHUNK_CHARS],
                "locator_type": row.get("locator_type"),
                "locator_label": row.get("locator_label"),
                "page_number": row.get("page_number"),
                "citation": chunk_citation(row),
            }
            for row in selected
        ]

    scored = []
    for row in rows:
        text = row.get("text") or ""
        parts = [text[i : i + 1200] for i in range(0, min(len(text), 24000), 1000)]
        for idx, part in enumerate(parts):
            score = score_text(query, part)
            if score:
                scored.append(
                    (
                        score,
                        {
                            "document_id": row["id"],
                            "title": row["filename"],
                            "chunk": idx + 1,
                            "content": part[:1200],
                            "citation": f"chunk {idx + 1}",
                        },
                    )
                )
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [item[1] for item in scored[:limit]]

    if not wants_document_answer(query):
        return []

    fallback = []
    for row in rows[:limit]:
        text = row.get("text") or ""
        if text:
            fallback.append({"document_id": row["id"], "title": row["filename"], "chunk": 1, "content": text[:1200], "citation": "chunk 1"})
    return fallback


def attachment_context(user_id, chat_id):
    docs = list_documents(user_id, chat_id) if chat_id else []
    if not docs:
        return "", []
    lines = []
    sources = []
    for doc in docs[:12]:
        text_chars = doc.get("text_chars", 0)
        if doc.get("media_kind") == "image":
            readability = (
                "OCR text is available and the image may also be available to a vision model."
                if text_chars
                else "Image is attached. Use vision input if it is available; if no vision input is provided, explain that the image file is attached but cannot be visually inspected by the current provider."
            )
        else:
            readability = (
                "Readable text is available."
                if text_chars
                else "No readable text was extracted. If the user asks to summarize or answer from this file, say the file is attached but appears scanned/image-only/unreadable on the server; ask for a text-based PDF, OCR copy, or pasted text."
            )
        lines.append(
            f"- {doc.get('filename')} ({doc.get('media_kind') or 'file'}, {text_chars} text chars): {doc.get('summary') or ''} {readability}"
        )
        sources.append({"document_id": doc.get("id"), "title": doc.get("filename"), "kind": doc.get("media_kind")})
    return "Files attached to this conversation:\n" + "\n".join(lines), sources


def rag_context(user_id, query, chat_id=None):
    chunks = best_document_chunks(user_id, query, chat_id)
    file_context, file_sources = attachment_context(user_id, chat_id)
    if not chunks:
        return file_context, file_sources
    context = "\n\n".join(
        f"Document: {c['title']} ({c.get('citation') or 'chunk ' + str(c['chunk'])}, chunk {c['chunk']})\n{c['content']}" for c in chunks
    )
    sources = [
        {
            "document_id": c["document_id"],
            "title": c["title"],
            "chunk": c["chunk"],
            "page": c.get("page_number"),
            "citation": c.get("citation"),
        }
        for c in chunks
    ]
    doc_rule = (
        "Use these uploaded document excerpts as the authoritative source for document questions. "
        "When the user asks to summarize, define, find, or answer from an uploaded file, answer only from these excerpts. "
        "If the answer is not present in the excerpts, say it is not available in the uploaded document context. "
        "Cite the document title and page/slide/sheet/chunk shown with each excerpt.\n\n"
    )
    full_context = "\n\n".join(part for part in [file_context, doc_rule + context] if part)
    return full_context, [*file_sources, *sources]


def create_or_update_chat(user_id, chat_id, title, messages, system_prompt, prompt_version, metadata=None):
    now = utc_now()
    existing_rows = (
        sb().table("chats").select("system_prompt,prompt_version,created_at").eq("id", chat_id).limit(1).execute().data
    ) or []
    existing = existing_rows[0] if existing_rows else None
    if existing:
        frozen_prompt = existing["system_prompt"]
        frozen_version = existing["prompt_version"]
        created_at = existing["created_at"]
    else:
        frozen_prompt = system_prompt
        frozen_version = prompt_version
        created_at = now

    sb().table("chats").upsert(
        {
            "id": chat_id,
            "owner_id": user_id,
            "title": title,
            "role": None,
            "prompt_version": frozen_version,
            "system_prompt": frozen_prompt,
            "metadata": json.dumps(metadata or {}),
            "created_at": created_at,
            "updated_at": now,
        }
    ).execute()

    sb().table("messages").delete().eq("chat_id", chat_id).execute()
    if messages:
        rows = [
            {
                "id": uuid.uuid4().hex,
                "chat_id": chat_id,
                "role": message["role"],
                "content": message["content"],
                "sources": json.dumps(message.get("sources", [])),
                "created_at": utc_now(),
            }
            for message in messages
        ]
        sb().table("messages").insert(rows).execute()

    return {"system_prompt": frozen_prompt, "prompt_version": frozen_version}


def set_chat_flag(user_id, chat_id, flag, value):
    if flag not in {"pinned", "bookmarked"}:
        raise ValueError("Unknown chat flag.")
    sb().table("chats").update({flag: bool(value), "updated_at": utc_now()}).eq("id", chat_id).eq(
        "owner_id", user_id
    ).execute()
    log_activity(user_id, f"chat.{flag}", {"chat_id": chat_id, "value": value})


def export_chat(user_id, chat_id, fmt="md"):
    chats = (
        sb().table("chats").select("*").eq("id", chat_id).eq("owner_id", user_id).limit(1).execute().data
    ) or []
    if not chats:
        raise ValueError("Chat not found.")
    chat = chats[0]
    rows = (
        sb().table("messages").select("role,content,sources,created_at").eq("chat_id", chat_id).order("created_at").execute().data
    ) or []
    if fmt == "json":
        return "application/json", json.dumps({"chat": chat, "messages": rows}, indent=2)
    lines = [f"# {chat.get('title') or 'CASPAM Chat'}", ""]
    for row in rows:
        lines.append(f"## {row['role'].title()}")
        lines.append(row["content"])
        if row.get("sources"):
            lines.append("")
            lines.append(f"Sources: {row['sources']}")
        lines.append("")
    return "text/markdown", "\n".join(lines)


def suggested_questions(messages, sources=None):
    latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    words = [w for w in tokenize(latest) if len(w) > 4]
    common = [w for w, _ in Counter(words).most_common(3)]
    if not common:
        return ["Can you explain this with an example?", "What should I study next?", "Can you make a short quiz from this?"]
    topic = " ".join(common[:2])
    return [f"Can you give an example about {topic}?", f"What are common mistakes in {topic}?", "Can you summarize this for exam revision?"]


def dashboard(user_id, role):
    docs = len((sb().table("documents").select("id").eq("owner_id", user_id).execute().data) or [])
    chats_rows = (sb().table("chats").select("id").eq("owner_id", user_id).execute().data) or []
    chats = len(chats_rows)
    chat_ids = [row["id"] for row in chats_rows]
    messages = 0
    if chat_ids:
        messages = len((sb().table("messages").select("id").in_("chat_id", chat_ids).execute().data) or [])
    users = None
    if role == "admin":
        users = len((sb().table("profiles").select("user_id").execute().data) or [])
    cards = [
        {"label": "Chats", "value": chats},
        {"label": "Messages", "value": messages},
        {"label": "Documents", "value": docs},
    ]
    if users is not None:
        cards.append({"label": "Users", "value": users})
    return {"role": role, "cards": cards, "notifications": notifications(user_id, role)}


def analytics(user_id, role):
    if role == "admin":
        rows = (sb().table("activity_logs").select("action").execute().data) or []
    else:
        rows = (sb().table("activity_logs").select("action").eq("user_id", user_id).execute().data) or []
    counts = Counter(row["action"] for row in rows)
    return {"events": [{"action": action, "count": count} for action, count in counts.items()]}


def notifications(user_id, role):
    notes = []
    profile = get_profile(user_id)
    if profile.get("role") == "student":
        notes.append({"type": "tip", "message": "Upload lecture notes to ask source-cited questions."})
    if role in {"teacher", "admin"}:
        notes.append({"type": "dashboard", "message": "Dashboard analytics and activity logs are available."})
    return notes


def activity(user_id, role, limit=100):
    q = sb().table("activity_logs").select("*")
    if role != "admin":
        q = q.eq("user_id", user_id)
    rows = (q.order("created_at", desc=True).limit(limit).execute().data) or []
    return rows


def save_lecture_version(user_id, lecture_id, title, content):
    rows = (
        sb()
        .table("lecture_versions")
        .select("version")
        .eq("lecture_id", lecture_id)
        .eq("owner_id", user_id)
        .order("version", desc=True)
        .limit(1)
        .execute()
        .data
    ) or []
    version = (rows[0]["version"] if rows else 0) + 1
    sb().table("lecture_versions").insert(
        {
            "id": uuid.uuid4().hex,
            "owner_id": user_id,
            "lecture_id": lecture_id,
            "title": title,
            "content": content,
            "version": version,
            "created_at": utc_now(),
        }
    ).execute()
    log_activity(user_id, "lecture.version", {"lecture_id": lecture_id, "version": version})
    return {"lecture_id": lecture_id, "version": version}


def backup_data(user_id, role):
    if role != "admin":
        raise PermissionError("Only admins can create backups.")
    backup_id = f"backup_{int(time.time())}.json"
    payload = {}
    for table in ("profiles", "documents", "document_chunks", "chats", "messages", "activity_logs", "lecture_versions"):
        payload[table] = (sb().table(table).select("*").execute().data) or []
    data = json.dumps(payload, indent=2).encode("utf-8")
    storage_path = f"_backups/{backup_id}"
    sb().storage.from_(SUPABASE_BUCKET).upload(storage_path, data, {"content-type": "application/json"})
    log_activity(user_id, "backup.create", {"backup": backup_id})
    return {"backup_id": backup_id, "path": storage_path}


def restore_data(user_id, role, payload):
    if role != "admin":
        raise PermissionError("Only admins can restore backups.")
    allowed_tables = {"profiles", "documents", "document_chunks", "chats", "messages", "activity_logs", "lecture_versions"}
    for table, rows in payload.items():
        if table not in allowed_tables or not isinstance(rows, list) or not rows:
            continue
        sb().table(table).upsert(rows).execute()
    log_activity(user_id, "backup.restore", {})
