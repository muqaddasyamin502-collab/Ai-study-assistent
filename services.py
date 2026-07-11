import base64
import csv
import io
import json
import os
import re
import sqlite3
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


DATA_DIR = Path(os.getenv("CASPAM_DATA_DIR", "instance")).resolve()
DB_PATH = DATA_DIR / "caspam.db"
UPLOAD_DIR = DATA_DIR / "uploads"
BACKUP_DIR = DATA_DIR / "backups"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
LATEST_PROMPT_VERSION = os.getenv("SYSTEM_PROMPT_VERSION", "2026-07-10")


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
                filename text not null,
                content_type text,
                path text not null,
                text text,
                created_at text not null
            );
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


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


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


def extract_docx(raw):
    text = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in sorted(n for n in zf.namelist() if n.startswith("word/") and n.endswith(".xml")):
            root = ElementTree.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    text.append(node.text)
    return " ".join(text)


def extract_pptx(raw):
    text = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in sorted(n for n in zf.namelist() if n.startswith("ppt/slides/") and n.endswith(".xml")):
            root = ElementTree.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    text.append(node.text)
    return " ".join(text)


def extract_pdf(raw):
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_image(raw):
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""
    image = Image.open(io.BytesIO(raw))
    return pytesseract.image_to_string(image)


def extract_text_from_upload(filename, raw):
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            return clean_text(extract_txt(raw))
        if ext == ".docx":
            return clean_text(extract_docx(raw))
        if ext == ".pptx":
            return clean_text(extract_pptx(raw))
        if ext == ".pdf":
            return clean_text(extract_pdf(raw))
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            return clean_text(extract_image(raw))
    except Exception:
        return ""
    return ""


def save_document(user_id, file_storage):
    filename = safe_filename(file_storage.filename or "upload")
    if not allowed_file(filename):
        raise ValueError("Unsupported file type.")
    raw = file_storage.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large.")
    doc_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{doc_id}_{filename}"
    target.write_bytes(raw)
    text = extract_text_from_upload(filename, raw)
    with db() as conn:
        conn.execute(
            "insert into documents values (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, user_id, filename, file_storage.mimetype, str(target), text, utc_now()),
        )
    log_activity(user_id, "document.upload", {"document_id": doc_id, "filename": filename})
    return {"id": doc_id, "filename": filename, "text_chars": len(text), "ocr_used": Path(filename).suffix.lower() not in {".txt", ".docx", ".pptx", ".pdf"}}


def list_documents(user_id):
    with db() as conn:
        rows = conn.execute(
            "select id, filename, content_type, length(coalesce(text,'')) as text_chars, created_at from documents where owner_id in (?, 'shared') order by created_at desc",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def tokenize(text):
    return re.findall(r"[A-Za-z0-9\u0600-\u06FF]{3,}", (text or "").lower())


def score_text(query, text):
    q = Counter(tokenize(query))
    if not q:
        return 0
    words = Counter(tokenize(text))
    return sum(words.get(word, 0) * weight for word, weight in q.items())


def best_document_chunks(user_id, query, limit=5):
    with db() as conn:
        rows = conn.execute(
            "select id, filename, text, created_at from documents where owner_id in (?, 'shared') and coalesce(text,'') != '' order by created_at desc",
            (user_id,),
        ).fetchall()
    scored = []
    for row in rows:
        text = row["text"] or ""
        parts = [text[i : i + 1200] for i in range(0, min(len(text), 24000), 1000)]
        for idx, part in enumerate(parts):
            score = score_text(query, part)
            if score:
                scored.append((score, {"document_id": row["id"], "title": row["filename"], "chunk": idx + 1, "content": part[:1200]}))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [item[1] for item in scored[:limit]]

    q = query.lower()
    wants_uploaded_notes = any(
        phrase in q
        for phrase in [
            "uploaded",
            "lecture note",
            "lecture notes",
            "my notes",
            "notes",
            "document",
            "pdf",
            "summarize",
            "summary",
        ]
    )
    if not wants_uploaded_notes:
        return []

    fallback = []
    for row in rows[:limit]:
        text = row["text"] or ""
        if text:
            fallback.append({"document_id": row["id"], "title": row["filename"], "chunk": 1, "content": text[:1200]})
    return fallback


def rag_context(user_id, query):
    chunks = best_document_chunks(user_id, query)
    if not chunks:
        return "", []
    context = "\n\n".join(
        f"Document: {c['title']} (chunk {c['chunk']})\n{c['content']}" for c in chunks
    )
    sources = [{"document_id": c["document_id"], "title": c["title"], "chunk": c["chunk"]} for c in chunks]
    return "Use these uploaded document excerpts when relevant and cite them by document title and chunk.\n\n" + context, sources


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
        for table in ("profiles", "documents", "chats", "messages", "activity_logs", "lecture_versions"):
            payload[table] = [dict(row) for row in conn.execute(f"select * from {table}").fetchall()]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_activity(user_id, "backup.create", {"backup": backup_id})
    return {"backup_id": backup_id, "path": str(target)}


def restore_data(user_id, role, payload):
    if role != "admin":
        raise PermissionError("Only admins can restore backups.")
    allowed = {
        "profiles": {"user_id", "email", "display_name", "role", "preferences", "updated_at"},
        "documents": {"id", "owner_id", "filename", "content_type", "path", "text", "created_at"},
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
