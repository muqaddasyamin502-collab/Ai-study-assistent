import base64
import csv
import importlib.util
import io
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import requests


DATA_DIR = Path(os.getenv("CASPAM_DATA_DIR", "instance")).resolve()
DB_PATH = DATA_DIR / "caspam.db"
UPLOAD_DIR = DATA_DIR / "uploads"
BACKUP_DIR = DATA_DIR / "backups"

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
        "pymupdf": bool(importlib.util.find_spec("fitz")),
        "pillow": bool(importlib.util.find_spec("PIL")),
        "pytesseract": bool(importlib.util.find_spec("pytesseract")),
        "tesseract_binary": bool(shutil.which("tesseract")),
    }


def ensure_data_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def db():
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            create table if not exists documents (
                id text primary key,
                owner_id text,
                chat_id text,
                filename text not null,
                content_type text,
                path text not null,
                text text,
                media_kind text,
                size_bytes integer default 0,
                summary text,
                created_at text not null
            );
            create table if not exists document_chunks (
                id text primary key,
                document_id text not null,
                owner_id text,
                chat_id text,
                filename text,
                chunk_index integer not null,
                locator_type text,
                locator_label text,
                page_number integer,
                text text not null,
                embedding text,
                token_count integer default 0,
                created_at text not null
            );
            create index if not exists idx_document_chunks_owner_chat
                on document_chunks(owner_id, chat_id, document_id);
            create index if not exists idx_document_chunks_document
                on document_chunks(document_id, chunk_index);
            create table if not exists chats (
                id text primary key,
                owner_id text,
                title text,
                role text,
                prompt_version text,
                system_prompt text,
                pinned integer default 0,
                bookmarked integer default 0,
                metadata text,
                created_at text not null,
                updated_at text not null
            );
            create table if not exists messages (
                id text primary key,
                chat_id text not null,
                role text not null,
                content text not null,
                sources text,
                created_at text not null
            );
            create table if not exists profiles (
                user_id text primary key,
                email text,
                display_name text,
                role text default 'student',
                preferences text,
                updated_at text not null
            );
            create table if not exists activity_logs (
                id text primary key,
                user_id text,
                action text not null,
                details text,
                created_at text not null
            );
            create table if not exists lecture_versions (
                id text primary key,
                owner_id text,
                lecture_id text not null,
                title text,
                content text,
                version integer not null,
                created_at text not null
            );
            """
        )
        cols = {row["name"] for row in conn.execute("pragma table_info(documents)").fetchall()}
        migrations = {
            "chat_id": "alter table documents add column chat_id text",
            "media_kind": "alter table documents add column media_kind text",
            "size_bytes": "alter table documents add column size_bytes integer default 0",
            "summary": "alter table documents add column summary text",
        }
        for col, sql in migrations.items():
            if col not in cols:
                conn.execute(sql)


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
    with db() as conn:
        conn.execute(
            "insert into activity_logs values (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, user_id, action, json.dumps(details or {}), utc_now()),
        )


def upsert_profile(user_id, email="", display_name="", role="student", preferences=None):
    role = role if role in {"student", "teacher", "admin"} else "student"
    with db() as conn:
        conn.execute(
            """
            insert into profiles (user_id, email, display_name, role, preferences, updated_at)
            values (?, ?, ?, ?, ?, ?)
            on conflict(user_id) do update set
                email=excluded.email,
                display_name=excluded.display_name,
                role=excluded.role,
                preferences=excluded.preferences,
                updated_at=excluded.updated_at
            """,
            (user_id, email, display_name, role, json.dumps(preferences or {}), utc_now()),
        )
    log_activity(user_id, "profile.upsert", {"role": role})


def get_profile(user_id):
    with db() as conn:
        row = conn.execute("select * from profiles where user_id = ?", (user_id,)).fetchone()
    if not row:
        return {"user_id": user_id, "role": "student", "preferences": {}}
    data = dict(row)
    data["preferences"] = json.loads(data.get("preferences") or "{}")
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


def ocr_image_text(image):
    try:
        import pytesseract
    except Exception:
        return ""
    for lang in ("eng+urd", "eng", None):
        try:
            return pytesseract.image_to_string(image, lang=lang) if lang else pytesseract.image_to_string(image)
        except Exception:
            continue
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
                chunks.append({
                    "text": chunk_text,
                    "locator_type": segment.get("locator_type") or "document",
                    "locator_label": segment.get("locator_label") or "Document",
                    "page_number": segment.get("page_number"),
                    "token_count": len(tokenize(chunk_text)),
                    "embedding": sparse_embedding(chunk_text),
                })
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


def save_document(user_id, file_storage, chat_id=""):
    filename = safe_filename(file_storage.filename or "upload")
    if not allowed_file(filename):
        raise ValueError("Unsupported file type.")
    raw = file_storage.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large.")
    doc_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{doc_id}_{filename}"
    target.write_bytes(raw)
    kind = media_kind(filename)
    segments = extract_segments_from_upload(filename, raw)
    text = clean_text("\n\n".join(segment["text"] for segment in segments))
    if not text and kind in {"audio", "video"}:
        text = transcribe_media_upload(filename, raw, file_storage.mimetype)
        segments = [seg for seg in [make_segment(text, kind, f"{kind.title()} transcript")] if seg]
    chunks = chunk_segments(segments)
    summary = describe_upload(filename, file_storage.mimetype, len(raw), text)
    now = utc_now()
    with db() as conn:
        conn.execute(
            """
            insert into documents (id, owner_id, chat_id, filename, content_type, path, text, media_kind, size_bytes, summary, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, user_id, clean_text(chat_id)[:128], filename, file_storage.mimetype, str(target), text, kind, len(raw), summary, now),
        )
        for index, chunk in enumerate(chunks, start=1):
            conn.execute(
                """
                insert into document_chunks (
                    id, document_id, owner_id, chat_id, filename, chunk_index,
                    locator_type, locator_label, page_number, text, embedding, token_count, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    doc_id,
                    user_id,
                    clean_text(chat_id)[:128],
                    filename,
                    index,
                    chunk["locator_type"],
                    chunk["locator_label"],
                    chunk["page_number"],
                    chunk["text"],
                    json.dumps(chunk["embedding"], separators=(",", ":")),
                    chunk["token_count"],
                    now,
                ),
            )
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
    with db() as conn:
        if chat_id:
            rows = conn.execute(
                """
                select d.id, d.chat_id, d.filename, d.content_type, d.media_kind, d.size_bytes, d.summary,
                       length(coalesce(d.text,'')) as text_chars, d.created_at,
                       count(c.id) as chunks, max(c.page_number) as pages
                from documents d
                left join document_chunks c on c.document_id = d.id
                where d.owner_id in (?, 'shared') and (d.chat_id = ? or d.owner_id = 'shared')
                group by d.id
                order by d.created_at desc
                """,
                (user_id, chat_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select d.id, d.chat_id, d.filename, d.content_type, d.media_kind, d.size_bytes, d.summary,
                       length(coalesce(d.text,'')) as text_chars, d.created_at,
                       count(c.id) as chunks, max(c.page_number) as pages
                from documents d
                left join document_chunks c on c.document_id = d.id
                where d.owner_id in (?, 'shared')
                group by d.id
                order by d.created_at desc
                """,
                (user_id,),
            ).fetchall()
    return [dict(row) for row in rows]


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
    with db() as conn:
        rows = conn.execute(
            """
            select id, filename, content_type, path, size_bytes, created_at
            from documents
            where owner_id in (?, 'shared') and chat_id = ? and media_kind = 'image'
            order by created_at desc
            limit ?
            """,
            (user_id, chat_id, limit),
        ).fetchall()
    images = []
    for row in rows:
        path = Path(row["path"] or "")
        try:
            if not path.exists() or path.stat().st_size > MAX_UPLOAD_BYTES:
                continue
            raw, content_type = prepare_vision_image(path.read_bytes(), row["content_type"] or "image/jpeg")
            if len(raw) > VISION_IMAGE_MAX_BYTES:
                continue
            images.append(
                {
                    "document_id": row["id"],
                    "filename": row["filename"],
                    "content_type": content_type,
                    "data": base64.b64encode(raw).decode("ascii"),
                    "size_bytes": row["size_bytes"] or path.stat().st_size,
                }
            )
        except Exception:
            continue
    return images


def recent_pdf_page_attachments(user_id, chat_id=None, limit=1, max_pages=PDF_VISION_MAX_PAGES):
    if not chat_id:
        return []
    with db() as conn:
        rows = conn.execute(
            """
            select id, filename, content_type, path, size_bytes, created_at, length(coalesce(text,'')) as text_chars
            from documents
            where owner_id in (?, 'shared') and chat_id = ? and media_kind = 'document'
                  and lower(filename) like '%.pdf'
            order by created_at desc
            limit ?
            """,
            (user_id, chat_id, limit),
        ).fetchall()
    pages = []
    try:
        import fitz
    except Exception:
        return pages
    for row in rows:
        path = Path(row["path"] or "")
        try:
            if not path.exists() or path.stat().st_size > MAX_UPLOAD_BYTES:
                continue
            doc = fitz.open(path)
            for page_index, page in enumerate(doc, start=1):
                if page_index > max_pages:
                    break
                pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                raw, content_type = prepare_vision_image(pix.tobytes("png"), "image/png")
                if len(raw) > VISION_IMAGE_MAX_BYTES:
                    continue
                pages.append(
                    {
                        "document_id": row["id"],
                        "filename": f"{row['filename']} page {page_index}",
                        "content_type": content_type,
                        "data": base64.b64encode(raw).decode("ascii"),
                        "size_bytes": len(raw),
                        "page_number": page_index,
                        "source_filename": row["filename"],
                    }
                )
        except Exception:
            continue
    return pages


def document_context_for_ids(user_id, chat_id, document_ids, limit=16):
    ids = [clean_text(doc_id)[:128] for doc_id in document_ids or [] if clean_text(doc_id)]
    if not ids:
        return "", []
    placeholders = ",".join("?" for _ in ids)
    params = [user_id, *ids]
    chat_clause = ""
    if chat_id:
        chat_clause = " and (chat_id = ? or owner_id = 'shared')"
        params.append(chat_id)

    with db() as conn:
        chunks = conn.execute(
            f"""
            select * from document_chunks
            where owner_id in (?, 'shared') and document_id in ({placeholders}){chat_clause}
            order by created_at desc, document_id, chunk_index asc
            limit ?
            """,
            [*params, limit],
        ).fetchall()
        docs = conn.execute(
            f"""
            select id, filename, media_kind, summary, text, created_at
            from documents
            where owner_id in (?, 'shared') and id in ({placeholders}){chat_clause}
            order by created_at desc
            """,
            params,
        ).fetchall()

    doc_lookup = {row["id"]: dict(row) for row in docs}
    sources = []
    parts = []
    if chunks:
        for row in chunks:
            data = dict(row)
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
            text = clean_text(doc["text"] or "")
            if not text:
                continue
            parts.append(f"Attached file: {doc['filename']}\n{text[:CHUNK_CHARS]}")
            sources.append({"document_id": doc["id"], "title": doc["filename"], "kind": doc["media_kind"]})

    if not parts:
        return "", [{"document_id": doc["id"], "title": doc["filename"], "kind": doc["media_kind"]} for doc in docs]

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
    with db() as conn:
        if chat_id:
            chunk_rows = conn.execute(
                """
                select c.* from document_chunks c
                where c.owner_id in (?, 'shared') and (c.chat_id = ? or c.owner_id = 'shared')
                order by c.created_at desc, c.chunk_index asc
                """,
                (user_id, chat_id),
            ).fetchall()
        else:
            chunk_rows = conn.execute(
                """
                select c.* from document_chunks c
                where c.owner_id in (?, 'shared')
                order by c.created_at desc, c.chunk_index asc
                """,
                (user_id,),
            ).fetchall()

        if not chunk_rows:
            if chat_id:
                rows = conn.execute(
                    """
                    select id, filename, text, created_at from documents
                    where owner_id in (?, 'shared') and (chat_id = ? or owner_id = 'shared') and coalesce(text,'') != ''
                    order by created_at desc
                    """,
                    (user_id, chat_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "select id, filename, text, created_at from documents where owner_id in (?, 'shared') and coalesce(text,'') != '' order by created_at desc",
                    (user_id,),
                ).fetchall()
        else:
            rows = []

    if chunk_rows:
        scored = []
        for row in chunk_rows:
            data = dict(row)
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
            selected = [dict(row) for row in chunk_rows[:limit]]
        else:
            selected = []
        return [
            {
                "document_id": row["document_id"],
                "title": row["filename"],
                "chunk": row["chunk_index"],
                "content": row["text"][:CHUNK_CHARS],
                "locator_type": row["locator_type"],
                "locator_label": row["locator_label"],
                "page_number": row["page_number"],
                "citation": chunk_citation(row),
            }
            for row in selected
        ]

    scored = []
    for row in rows:
        text = row["text"] or ""
        parts = [text[i : i + 1200] for i in range(0, min(len(text), 24000), 1000)]
        for idx, part in enumerate(parts):
            score = score_text(query, part)
            if score:
                scored.append((score, {"document_id": row["id"], "title": row["filename"], "chunk": idx + 1, "content": part[:1200], "citation": f"chunk {idx + 1}"}))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [item[1] for item in scored[:limit]]

    if not wants_document_answer(query):
        return []

    fallback = []
    for row in rows[:limit]:
        text = row["text"] or ""
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
    with db() as conn:
        existing = conn.execute("select system_prompt, prompt_version, created_at from chats where id = ?", (chat_id,)).fetchone()
        if existing:
            frozen_prompt = existing["system_prompt"]
            frozen_version = existing["prompt_version"]
            created_at = existing["created_at"]
        else:
            frozen_prompt = system_prompt
            frozen_version = prompt_version
            created_at = now
        conn.execute(
            """
            insert into chats (id, owner_id, title, role, prompt_version, system_prompt, metadata, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(id) do update set
                title=excluded.title,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (chat_id, user_id, title, None, frozen_version, frozen_prompt, json.dumps(metadata or {}), created_at, now),
        )
        conn.execute("delete from messages where chat_id = ?", (chat_id,))
        for message in messages:
            conn.execute(
                "insert into messages values (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, chat_id, message["role"], message["content"], json.dumps(message.get("sources", [])), utc_now()),
            )
    return {"system_prompt": frozen_prompt, "prompt_version": frozen_version}


def set_chat_flag(user_id, chat_id, flag, value):
    if flag not in {"pinned", "bookmarked"}:
        raise ValueError("Unknown chat flag.")
    with db() as conn:
        conn.execute(f"update chats set {flag} = ?, updated_at = ? where id = ? and owner_id = ?", (1 if value else 0, utc_now(), chat_id, user_id))
    log_activity(user_id, f"chat.{flag}", {"chat_id": chat_id, "value": value})


def export_chat(user_id, chat_id, fmt="md"):
    with db() as conn:
        chat = conn.execute("select * from chats where id = ? and owner_id = ?", (chat_id, user_id)).fetchone()
        rows = conn.execute("select role, content, sources, created_at from messages where chat_id = ? order by created_at", (chat_id,)).fetchall()
    if not chat:
        raise ValueError("Chat not found.")
    if fmt == "json":
        return "application/json", json.dumps({"chat": dict(chat), "messages": [dict(r) for r in rows]}, indent=2)
    lines = [f"# {chat['title'] or 'CASPAM Chat'}", ""]
    for row in rows:
        lines.append(f"## {row['role'].title()}")
        lines.append(row["content"])
        if row["sources"]:
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
    with db() as conn:
        docs = conn.execute("select count(*) as c from documents where owner_id = ?", (user_id,)).fetchone()["c"]
        chats = conn.execute("select count(*) as c from chats where owner_id = ?", (user_id,)).fetchone()["c"]
        messages = conn.execute(
            "select count(*) as c from messages where chat_id in (select id from chats where owner_id = ?)",
            (user_id,),
        ).fetchone()["c"]
        users = conn.execute("select count(*) as c from profiles").fetchone()["c"] if role == "admin" else None
    cards = [
        {"label": "Chats", "value": chats},
        {"label": "Messages", "value": messages},
        {"label": "Documents", "value": docs},
    ]
    if users is not None:
        cards.append({"label": "Users", "value": users})
    return {"role": role, "cards": cards, "notifications": notifications(user_id, role)}


def analytics(user_id, role):
    with db() as conn:
        rows = conn.execute("select action, count(*) as count from activity_logs where user_id = ? group by action", (user_id,)).fetchall()
        if role == "admin":
            rows = conn.execute("select action, count(*) as count from activity_logs group by action").fetchall()
    return {"events": [dict(row) for row in rows]}


def notifications(user_id, role):
    notes = []
    profile = get_profile(user_id)
    if profile.get("role") == "student":
        notes.append({"type": "tip", "message": "Upload lecture notes to ask source-cited questions."})
    if role in {"teacher", "admin"}:
        notes.append({"type": "dashboard", "message": "Dashboard analytics and activity logs are available."})
    return notes


def activity(user_id, role, limit=100):
    with db() as conn:
        if role == "admin":
            rows = conn.execute("select * from activity_logs order by created_at desc limit ?", (limit,)).fetchall()
        else:
            rows = conn.execute("select * from activity_logs where user_id = ? order by created_at desc limit ?", (user_id, limit)).fetchall()
    return [dict(row) for row in rows]


def save_lecture_version(user_id, lecture_id, title, content):
    with db() as conn:
        row = conn.execute("select max(version) as version from lecture_versions where lecture_id = ? and owner_id = ?", (lecture_id, user_id)).fetchone()
        version = (row["version"] or 0) + 1
        conn.execute(
            "insert into lecture_versions values (?, ?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, user_id, lecture_id, title, content, version, utc_now()),
        )
    log_activity(user_id, "lecture.version", {"lecture_id": lecture_id, "version": version})
    return {"lecture_id": lecture_id, "version": version}


def backup_data(user_id, role):
    if role != "admin":
        raise PermissionError("Only admins can create backups.")
    backup_id = f"backup_{int(time.time())}.json"
    target = BACKUP_DIR / backup_id
    payload = {}
    with db() as conn:
        for table in ("profiles", "documents", "document_chunks", "chats", "messages", "activity_logs", "lecture_versions"):
            payload[table] = [dict(row) for row in conn.execute(f"select * from {table}").fetchall()]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_activity(user_id, "backup.create", {"backup": backup_id})
    return {"backup_id": backup_id, "path": str(target)}


def restore_data(user_id, role, payload):
    if role != "admin":
        raise PermissionError("Only admins can restore backups.")
    allowed = {
        "profiles": {"user_id", "email", "display_name", "role", "preferences", "updated_at"},
        "documents": {"id", "owner_id", "chat_id", "filename", "content_type", "path", "text", "media_kind", "size_bytes", "summary", "created_at"},
        "document_chunks": {"id", "document_id", "owner_id", "chat_id", "filename", "chunk_index", "locator_type", "locator_label", "page_number", "text", "embedding", "token_count", "created_at"},
        "chats": {"id", "owner_id", "title", "role", "prompt_version", "system_prompt", "pinned", "bookmarked", "metadata", "created_at", "updated_at"},
        "messages": {"id", "chat_id", "role", "content", "sources", "created_at"},
        "activity_logs": {"id", "user_id", "action", "details", "created_at"},
        "lecture_versions": {"id", "owner_id", "lecture_id", "title", "content", "version", "created_at"},
    }
    with db() as conn:
        for table, rows in payload.items():
            if table not in allowed or not isinstance(rows, list):
                continue
            if not rows:
                continue
            cols = [col for col in rows[0].keys() if col in allowed[table]]
            if not cols:
                continue
            placeholders = ",".join("?" for _ in cols)
            for row in rows:
                conn.execute(
                    f"insert or replace into {table} ({','.join(cols)}) values ({placeholders})",
                    [row.get(col) for col in cols],
                )
    log_activity(user_id, "backup.restore", {})
