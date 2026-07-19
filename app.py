import os
import re
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from services import (
    LATEST_PROMPT_VERSION,
    activity,
    analytics,
    backup_data,
    create_or_update_chat,
    dashboard,
    db,
    dependency_status,
    export_chat,
    get_profile,
    get_user_id,
    get_user_role,
    init_db,
    list_documents,
    log_activity,
    rag_context,
    recent_image_attachments,
    recent_pdf_page_attachments,
    restore_data,
    save_document,
    save_lecture_version,
    set_chat_flag,
    suggested_questions,
    upsert_profile,
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
CORS(app)
init_db()


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if request.path.startswith("/api/"):
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.name, "details": exc.description}), exc.code or 500
        return jsonify({"error": "Backend error", "details": str(exc)}), 500
    raise exc


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        request.headers.get("Access-Control-Request-Headers")
        or "Content-Type, Authorization, X-User-Id, X-Firebase-Uid, X-User-Email, X-User-Role"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PATCH"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=(), payment=()"
    return response

def clean_env_value(value):
    value = (value or "").strip().strip("\"'")
    if "=" in value:
        value = value.split("=", 1)[1].strip().strip("\"'")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", value):
        value = value.split(":", 1)[1].strip().strip("\"',{} ")
    return value


GROQ_API_KEY = clean_env_value(os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))
GROQ_MODEL = clean_env_value(os.getenv("GROQ_MODEL"))
if not GROQ_MODEL:
    old_model = clean_env_value(os.getenv("GROK_MODEL"))
    GROQ_MODEL = old_model if old_model and not old_model.lower().startswith("grok-") else "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = clean_env_value(os.getenv("GROQ_VISION_MODEL")) or "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = clean_env_value(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = clean_env_value(os.getenv("GEMINI_MODEL")) or "gemini-3.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CLAUDE_API_KEY = clean_env_value(os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = clean_env_value(os.getenv("CLAUDE_MODEL")) or "claude-haiku-4-5"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

MAX_OUTPUT_TOKENS = int(clean_env_value(os.getenv("MAX_OUTPUT_TOKENS")) or "600")

FIREBASE_CONFIG = {
    "apiKey": clean_env_value(os.getenv("FIREBASE_API_KEY")),
    "authDomain": clean_env_value(os.getenv("FIREBASE_AUTH_DOMAIN")),
    "projectId": clean_env_value(os.getenv("FIREBASE_PROJECT_ID")),
    "storageBucket": clean_env_value(os.getenv("FIREBASE_STORAGE_BUCKET")),
    "messagingSenderId": clean_env_value(os.getenv("FIREBASE_MESSAGING_SENDER_ID")),
    "appId": clean_env_value(os.getenv("FIREBASE_APP_ID")),
}

KNOWLEDGE_SOURCES = {
    "HEC Pakistan": "https://www.hec.gov.pk/",
    "BZU Multan": "https://www.bzu.edu.pk/",
    "BZU Admissions": "https://admissions.bzu.edu.pk/",
    "BZU CASPAM": "https://www.bzu.edu.pk/caspam",
}

ALLOWED_DOMAINS = {
    "hec.gov.pk",
    "www.hec.gov.pk",
    "bzu.edu.pk",
    "www.bzu.edu.pk",
    "admissions.bzu.edu.pk",
}

SYSTEM_PROMPT = f"""
You are CASPAM-Bot, the official AI assistant for CASPAM
(Centre for Advanced Studies in Pure and Applied Mathematics), BZU Multan, Pakistan.

Official knowledge links:
- HEC Pakistan: {KNOWLEDGE_SOURCES["HEC Pakistan"]}
- BZU Multan: {KNOWLEDGE_SOURCES["BZU Multan"]}
- BZU Admissions: {KNOWLEDGE_SOURCES["BZU Admissions"]}
- BZU CASPAM: {KNOWLEDGE_SOURCES["BZU CASPAM"]}

Rules:
1. Always reply in English, even if the student writes Urdu or Roman Urdu.
2. Answer exactly what the student asked. Keep short answers short.
3. Use official HEC/BZU information when relevant.
4. If live website context is missing or uncertain, say that the student should verify on the official link.
5. Use LaTeX for math: inline $formula$, display $$formula$$.
6. Use fenced code blocks with a language tag for code.
7. Be helpful, friendly, and encouraging for BZU students.
8. If asked who created you, who made you, who built you, your owner, your developer, or your author, answer only with this identity: "Muqaddas Yamin is my author. She made me to help CASPAM and BZU students." Do not mention Meta, OpenAI, Groq, Anthropic, or the base model in that answer.
""".strip()


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def is_allowed_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in ALLOWED_DOMAINS
    except Exception:
        return False


def fetch_page_text(url, timeout=10, max_chars=3500):
    if not is_allowed_url(url):
        return ""

    headers = {
        "User-Agent": "CASPAM-Bot/1.0 educational chatbot",
        "Accept": "text/html,application/xhtml+xml",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else url
    body = clean_text(soup.get_text(" "))
    return f"Source: {url}\nTitle: {title}\nContent: {body[:max_chars]}"


def pick_sources_for_query(query):
    q = query.lower()
    selected = [KNOWLEDGE_SOURCES["BZU Multan"], KNOWLEDGE_SOURCES["HEC Pakistan"]]

    if any(word in q for word in ["admission", "apply", "merit", "fee", "prospectus"]):
        selected.insert(0, KNOWLEDGE_SOURCES["BZU Admissions"])

    if any(word in q for word in ["caspam", "math", "mathematics", "department"]):
        selected.insert(0, KNOWLEDGE_SOURCES["BZU CASPAM"])

    if any(word in q for word in ["hec", "degree", "attestation", "recognition", "equivalence"]):
        selected.insert(0, KNOWLEDGE_SOURCES["HEC Pakistan"])

    unique = []
    for source in selected:
        if source not in unique:
            unique.append(source)
    return unique[:4]


def build_website_context(query):
    q = query.lower()
    needs_context = any(
        word in q
        for word in [
            "bzu",
            "caspam",
            "admission",
            "apply",
            "merit",
            "fee",
            "prospectus",
            "hec",
            "degree",
            "attestation",
            "recognition",
            "equivalence",
            "department",
        ]
    )
    if not needs_context:
        return ""

    chunks = []
    for source in pick_sources_for_query(query):
        try:
            text = fetch_page_text(source)
            if text:
                chunks.append(text)
        except Exception as exc:
            chunks.append(f"Source: {source}\nStatus: Could not fetch live content ({exc}).")
    return "\n\n---\n\n".join(chunks)


def normalize_messages(messages):
    clean_messages = []
    for msg in messages[-16:]:
        role = msg.get("role")
        content = clean_text(msg.get("content", ""))
        if role in {"user", "assistant"} and content:
            clean_messages.append({"role": role, "content": content})
    return clean_messages


def readable_groq_error(error_detail):
    if isinstance(error_detail, dict):
        error = error_detail.get("error")
        if isinstance(error, dict):
            return error.get("message") or str(error)
        return error_detail.get("message") or str(error_detail)
    return str(error_detail)


def user_safe_provider_error(provider, exc, vision_attachments=None):
    text = str(exc)
    lower = text.lower()
    if provider == "Gemini" and ("api_key_invalid" in lower or "api key not valid" in lower):
        return "Gemini API key is invalid. Add a valid GEMINI_API_KEY in Render environment variables."
    if provider == "Gemini" and ("not found" in lower or "model" in lower and "invalid" in lower):
        return f"Gemini model is not available: {GEMINI_MODEL}. Update GEMINI_MODEL in Render."
    if provider == "Claude" and ("credit balance" in lower or "billing" in lower):
        return "Claude account has no available credits. Add credits or remove CLAUDE_API_KEY from Render."
    if provider == "Groq" and "groq_vision_model" in lower:
        return "Groq image vision is not configured. Set GROQ_VISION_MODEL to a Groq vision-capable model."
    if "api key" in lower:
        return f"{provider} API key is invalid or missing."
    if vision_attachments:
        return f"{provider} could not process the uploaded image."
    return f"{provider} is not available right now."


def all_providers_failed_message(errors, vision_attachments=None):
    if vision_attachments:
        return (
            "Groq image understanding is connected, but Groq could not answer this image request.\n\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n\nCheck Render env vars: GROQ_API_KEY must be valid and GROQ_VISION_MODEL should be "
            "meta-llama/llama-4-scout-17b-16e-instruct. Then redeploy and upload the image again."
        )
    return (
        "The AI providers are not available right now.\n\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\nPlease check the API keys and billing settings in Render."
    )


def context_message(website_context):
    if not website_context:
        return "No live website context was needed for this question."
    return (
        "Use this official website context when it is relevant. "
        "Do not invent details that are not present here.\n\n"
        f"{website_context}"
    )


def asks_about_visual_file(query):
    q = query.lower()
    return any(
        phrase in q
        for phrase in [
            "image",
            "picture",
            "photo",
            "screenshot",
            "diagram",
            "chart",
            "graph",
            "explain this",
            "what is in this",
            "describe this",
            "attached file",
        ]
    )


def latest_user_has_image_attachment(raw_messages):
    for msg in reversed(raw_messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        attachments = msg.get("attachments") if isinstance(msg.get("attachments"), list) else []
        return any(
            isinstance(att, dict)
            and clean_text(att.get("media_kind") or "").lower() == "image"
            for att in attachments
        )
    return False


def asks_about_uploaded_notes(query):
    q = query.lower()
    return any(
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


def latest_user_has_document_attachment(raw_messages):
    for msg in reversed(raw_messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        attachments = msg.get("attachments") if isinstance(msg.get("attachments"), list) else []
        return any(
            isinstance(att, dict)
            and clean_text(att.get("media_kind") or "").lower() == "document"
            for att in attachments
        )
    return False


def uploaded_document_status_context(user_id, query, document_context, chat_id=None):
    if document_context or not asks_about_uploaded_notes(query):
        return ""

    docs = list_documents(user_id, chat_id)
    if not docs:
        return (
            "The student is asking about uploaded lecture notes, but no uploaded documents "
            "are available for this user. Ask them to upload notes with the plus button first."
        )

    listed = "\n".join(
        f"- {doc.get('filename', 'uploaded file')} ({doc.get('text_chars', 0)} extracted text characters)"
        for doc in docs[:5]
    )
    return (
        "The student has uploaded document files, but no searchable text was extracted from them. "
        "Do not say that no files were uploaded. Explain that the uploaded file may be scanned, image-only, "
        "or unreadable on the server, and ask for a text-based PDF, DOCX, TXT, or pasted text.\n\n"
        f"Uploaded documents:\n{listed}"
    )


def client_attachment_status_context(raw_messages, user_id, chat_id):
    attachments = []
    for msg in raw_messages or []:
        if isinstance(msg, dict) and isinstance(msg.get("attachments"), list):
            attachments.extend(att for att in msg["attachments"] if isinstance(att, dict))
    if not attachments:
        return ""

    server_docs = list_documents(user_id, chat_id) if chat_id else []
    if server_docs:
        return ""

    listed = "\n".join(
        f"- {clean_text(att.get('filename') or att.get('name') or 'attached file')} "
        f"({clean_text(att.get('media_kind') or 'file')}, {int(att.get('text_chars') or 0)} text chars reported by browser)"
        for att in attachments[:8]
    )
    return (
        "The chat history contains attachment cards, but the server cannot find the uploaded file records/chunks "
        "for this conversation. This commonly happens after a Render redeploy/restart when uploads are stored on "
        "ephemeral disk, or when a chat was restored from browser localStorage without server-side files. "
        "Do not say the user never attached a file. Explain that the attachment card is visible but the backend "
        "cannot access the file content anymore, and ask the user to re-upload the PDF after deployment or enable "
        "persistent storage.\n\n"
        f"Attachment cards seen in chat history:\n{listed}"
    )


def groq_messages(messages, context_prompt, system_prompt=None, vision_attachments=None):
    payload_messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "system", "content": context_prompt},
        *messages,
    ]
    if vision_attachments:
        payload_messages = [dict(message) for message in payload_messages]
        for message in reversed(payload_messages):
            if message.get("role") == "user":
                content = [{"type": "text", "text": message.get("content", "")}]
                for image in vision_attachments:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image['content_type']};base64,{image['data']}"
                            },
                        }
                    )
                message["content"] = content
                break
    return payload_messages


def call_groq(messages, context_prompt, system_prompt=None, vision_attachments=None):
    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key is missing.")
    if vision_attachments and not GROQ_VISION_MODEL:
        raise RuntimeError("GROQ_VISION_MODEL is missing for image questions.")

    payload = {
        "model": GROQ_VISION_MODEL if vision_attachments else GROQ_MODEL,
        "messages": groq_messages(messages, context_prompt, system_prompt, vision_attachments),
        "temperature": 0.4,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise RuntimeError(readable_groq_error(error_detail))
    result = response.json()
    return result["choices"][0]["message"]["content"]


def call_gemini(messages, context_prompt, system_prompt=None, vision_attachments=None):
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is missing.")

    contents = []
    image_parts_added = False
    for index, msg in enumerate(messages):
        parts = [{"text": msg["content"]}]
        if vision_attachments and msg["role"] == "user" and not image_parts_added and index == len(messages) - 1:
            for image in vision_attachments:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": image["content_type"],
                            "data": image["data"],
                        }
                    }
                )
            image_parts_added = True
        contents.append(
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": parts,
            }
        )

    payload = {
        "systemInstruction": {
            "parts": [{"text": f"{system_prompt or SYSTEM_PROMPT}\n\n{context_prompt}"}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    response = requests.post(
        GEMINI_API_URL.format(model=GEMINI_MODEL),
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise RuntimeError(str(error_detail))
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def claude_messages(messages, vision_attachments=None):
    payload_messages = []
    image_parts_added = False
    for index, msg in enumerate(messages):
        content = msg["content"]
        if vision_attachments and msg["role"] == "user" and not image_parts_added and index == len(messages) - 1:
            content = [{"type": "text", "text": msg["content"]}]
            for image in vision_attachments:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image["content_type"],
                            "data": image["data"],
                        },
                    }
                )
            image_parts_added = True
        payload_messages.append({"role": msg["role"], "content": content})
    return payload_messages


def call_claude(messages, context_prompt, system_prompt=None, vision_attachments=None):
    if not CLAUDE_API_KEY:
        raise RuntimeError("Claude API key is missing.")

    payload = {
        "model": CLAUDE_MODEL,
        "system": f"{system_prompt or SYSTEM_PROMPT}\n\n{context_prompt}",
        "messages": claude_messages(messages, vision_attachments),
        "temperature": 0.4,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    response = requests.post(
        CLAUDE_API_URL,
        headers={
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise RuntimeError(str(error_detail))
    result = response.json()
    return "".join(block.get("text", "") for block in result.get("content", []))


def generate_reply(messages, context_prompt, system_prompt=None, vision_attachments=None):
    providers = [
        ("Groq", call_groq),
    ] if vision_attachments else [
        ("Groq", call_groq),
        ("Gemini", call_gemini),
        ("Claude", call_claude),
    ]
    errors = []
    for provider, caller in providers:
        try:
            reply = caller(messages, context_prompt, system_prompt, vision_attachments)
            return provider, reply
        except Exception as exc:
            errors.append(user_safe_provider_error(provider, exc, vision_attachments))
            print(f"{provider} API error: {exc}", flush=True)
    raise RuntimeError(all_providers_failed_message(errors, vision_attachments))


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "providers": {
                "groq": {"model": GROQ_MODEL, "has_api_key": bool(GROQ_API_KEY)},
                "groq_vision": {"model": GROQ_VISION_MODEL, "has_api_key": bool(GROQ_API_KEY and GROQ_VISION_MODEL)},
                "gemini": {"model": GEMINI_MODEL, "has_api_key": bool(GEMINI_API_KEY)},
                "claude": {"model": CLAUDE_MODEL, "has_api_key": bool(CLAUDE_API_KEY)},
            },
            "vision_ready": bool(GROQ_API_KEY and GROQ_VISION_MODEL),
            "dependencies": dependency_status(),
            "setup_notes": {
                "image_questions": "Groq image questions use GROQ_API_KEY plus GROQ_VISION_MODEL.",
                "render_free_storage": "Uploads on Render free tier are temporary; re-upload files after redeploy/restart.",
            },
            "sources": KNOWLEDGE_SOURCES,
            "prompt_version": LATEST_PROMPT_VERSION,
        }
    )


@app.get("/api/firebase-config")
def firebase_config():
    enabled = all(
        FIREBASE_CONFIG.get(key)
        for key in ["apiKey", "authDomain", "projectId", "appId"]
    )
    return jsonify({"enabled": enabled, "config": FIREBASE_CONFIG if enabled else {}})


@app.post("/api/chat")
def chat():
    if not any([GROQ_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY]):
        return jsonify({"error": "Add at least one API key: GROQ_API_KEY, GEMINI_API_KEY, or CLAUDE_API_KEY."}), 500

    data = request.get_json(silent=True) or {}
    raw_messages = data.get("messages", [])
    messages = normalize_messages(raw_messages)
    chat_id = clean_text(data.get("chat_id", ""))[:128]
    title = clean_text(data.get("title", ""))[:160]
    user_id = get_user_id(request)

    if not messages:
        return jsonify({"error": "No messages received."}), 400

    latest_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not re.search(r"[A-Za-z0-9\u0600-\u06FF]", latest_user_message):
        return jsonify({"reply": "Please type a clear question so I can help you."})

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    chat_system_prompt = SYSTEM_PROMPT
    if metadata.get("legacy_chat") and clean_text(metadata.get("system_prompt", "")):
        chat_system_prompt = metadata["system_prompt"]
    frozen = None
    if chat_id:
        frozen = create_or_update_chat(
            user_id,
            chat_id,
            title or latest_user_message[:40],
            messages,
            chat_system_prompt,
            LATEST_PROMPT_VERSION,
            metadata,
        )

    website_context = build_website_context(latest_user_message)
    document_context, sources = rag_context(user_id, latest_user_message, chat_id)
    document_question = asks_about_uploaded_notes(latest_user_message) or latest_user_has_document_attachment(raw_messages)
    visual_question = asks_about_visual_file(latest_user_message) or latest_user_has_image_attachment(raw_messages)
    vision_attachments = recent_image_attachments(user_id, chat_id) if visual_question else []
    vision_context = ""
    if vision_attachments:
        vision_context = (
            "The latest user question appears to ask about an uploaded image. "
            "The image file is included as vision input for providers that support images. "
            "Describe the visible content directly and mention uncertainty only where needed.\n\n"
            + "\n".join(
                f"- {image['filename']} ({image['content_type']}, {image['size_bytes']} bytes)"
                for image in vision_attachments
            )
        )
        sources.extend(
            {
                "document_id": image["document_id"],
                "title": image["filename"],
                "kind": "image",
                "vision": True,
            }
            for image in vision_attachments
        )
    elif visual_question:
        image_docs = [doc for doc in list_documents(user_id, chat_id) if doc.get("media_kind") == "image"]
        if image_docs:
            listed = "\n".join(
                f"- {doc.get('filename')} ({doc.get('size_bytes', 0)} bytes)"
                for doc in image_docs[:4]
            )
            vision_context = (
                "The student is asking about an uploaded image, but the backend could not load the image bytes "
                "for vision analysis. This can happen after a Render restart/redeploy because free-tier local "
                "uploads are temporary. Do not claim to see the image. Ask the student to re-upload it and then "
                "answer from the newly uploaded image.\n\n"
                f"Image records found:\n{listed}"
            )
    if not vision_attachments and document_question:
        pdf_pages = recent_pdf_page_attachments(user_id, chat_id)
        if pdf_pages:
            vision_attachments = pdf_pages
            vision_context = (
                "The uploaded PDF did not provide enough searchable text, so selected PDF pages are included "
                "as vision images. Read the visible page content and answer the user's PDF question from these "
                "page images. If only some pages are included, say that the answer is based on the visible pages "
                "provided to vision.\n\n"
                + "\n".join(
                    f"- {page['source_filename']} page {page['page_number']} ({page['content_type']}, {page['size_bytes']} bytes)"
                    for page in pdf_pages
                )
            )
            sources.extend(
                {
                    "document_id": page["document_id"],
                    "title": page["source_filename"],
                    "kind": "pdf_page_image",
                    "page": page["page_number"],
                    "vision": True,
                }
                for page in pdf_pages
            )
    has_text_source = any(source.get("chunk") for source in sources)
    upload_status_context = uploaded_document_status_context(
        user_id,
        latest_user_message,
        document_context if has_text_source or vision_attachments else "",
        chat_id,
    )
    client_attachment_context = client_attachment_status_context(raw_messages, user_id, chat_id)
    combined_context = "\n\n---\n\n".join(
        part for part in [website_context, document_context, vision_context, upload_status_context, client_attachment_context] if part
    )
    context_prompt = context_message(combined_context)

    try:
        provider, reply = generate_reply(
            messages,
            context_prompt,
            (frozen or {}).get("system_prompt") or SYSTEM_PROMPT,
            vision_attachments,
        )
        updated_messages = [*messages, {"role": "assistant", "content": reply, "sources": sources}]
        if chat_id:
            create_or_update_chat(
                user_id,
                chat_id,
                title or latest_user_message[:40],
                updated_messages,
                chat_system_prompt,
                LATEST_PROMPT_VERSION,
                metadata,
            )
        log_activity(user_id, "chat.message", {"chat_id": chat_id, "provider": provider})
        return jsonify(
            {
                "provider": provider,
                "reply": reply,
                "sources": sources,
                "suggested": suggested_questions(updated_messages, sources),
                "prompt_version": (frozen or {}).get("prompt_version") or LATEST_PROMPT_VERSION,
            }
        )

    except requests.Timeout:
        return jsonify({"error": "AI request timed out. Please try again."}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/search")
def search():
    query = clean_text(request.args.get("q", ""))
    if not query:
        return jsonify({"results": []})

    results = []
    for name, url in KNOWLEDGE_SOURCES.items():
        if query.lower() in name.lower() or any(part in url.lower() for part in query.lower().split()):
            results.append({"title": name, "url": url})

    if not results:
        results = [{"title": name, "url": url} for name, url in KNOWLEDGE_SOURCES.items()]

    return jsonify({"query": query, "results": results[:8]})


@app.post("/api/profile")
def profile_save():
    user_id = get_user_id(request)
    data = request.get_json(silent=True) or {}
    upsert_profile(
        user_id,
        clean_text(data.get("email", "")),
        clean_text(data.get("display_name", "")),
        clean_text(data.get("role", "student")),
        data.get("preferences") if isinstance(data.get("preferences"), dict) else {},
    )
    return jsonify({"profile": get_profile(user_id)})


@app.get("/api/profile")
def profile_get():
    return jsonify({"profile": get_profile(get_user_id(request))})


@app.post("/api/upload")
def upload():
    user_id = get_user_id(request)
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No file uploaded."}), 400
    try:
        chat_id = clean_text(request.form.get("chat_id", ""))[:128]
        docs = [save_document(user_id, file, chat_id) for file in files if file and file.filename]
        return jsonify({"document": docs[0] if docs else None, "documents": list_documents(user_id, chat_id), "uploaded": docs})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/documents")
def documents():
    chat_id = clean_text(request.args.get("chat_id", ""))[:128]
    return jsonify({"documents": list_documents(get_user_id(request), chat_id or None)})


@app.get("/api/dashboard")
def dashboard_route():
    user_id = get_user_id(request)
    return jsonify(dashboard(user_id, get_user_role(request)))


@app.get("/api/analytics")
def analytics_route():
    user_id = get_user_id(request)
    return jsonify(analytics(user_id, get_user_role(request)))


@app.get("/api/activity")
def activity_route():
    user_id = get_user_id(request)
    return jsonify({"activity": activity(user_id, get_user_role(request))})


@app.patch("/api/chats/<chat_id>/flags")
def chat_flags(chat_id):
    user_id = get_user_id(request)
    data = request.get_json(silent=True) or {}
    for flag in ("pinned", "bookmarked"):
        if flag in data:
            set_chat_flag(user_id, chat_id, flag, bool(data[flag]))
    return jsonify({"ok": True})


@app.get("/api/chats/<chat_id>/export")
def chat_export(chat_id):
    fmt = request.args.get("format", "md").lower()
    try:
        mimetype, content = export_chat(get_user_id(request), chat_id, fmt)
        extension = "json" if fmt == "json" else "md"
        return Response(
            content,
            mimetype=mimetype,
            headers={"Content-Disposition": f"attachment; filename=caspam-chat-{chat_id}.{extension}"},
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/lectures/<lecture_id>/versions")
def lecture_version(lecture_id):
    user_id = get_user_id(request)
    data = request.get_json(silent=True) or {}
    return jsonify(
        save_lecture_version(
            user_id,
            lecture_id,
            clean_text(data.get("title", "")),
            data.get("content", ""),
        )
    )


@app.post("/api/backup")
def backup_route():
    try:
        return jsonify(backup_data(get_user_id(request), get_user_role(request)))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403


@app.post("/api/restore")
def restore_route():
    try:
        restore_data(get_user_id(request), get_user_role(request), request.get_json(silent=True) or {})
        return jsonify({"ok": True})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403


@app.get("/api/bzu/")
@app.get("/api/bzu/<path:subpath>")
def bzu_info(subpath=""):
    url = urljoin(KNOWLEDGE_SOURCES["BZU Multan"], quote_plus(subpath).replace("%2F", "/"))
    if subpath and not is_allowed_url(url):
        return jsonify({"error": "Only official BZU pages are allowed."}), 400

    try:
        content = fetch_page_text(url if subpath else KNOWLEDGE_SOURCES["BZU Multan"], max_chars=6000)
        return jsonify({"url": url, "content": content, "links": KNOWLEDGE_SOURCES})
    except Exception as exc:
        return jsonify({"error": str(exc), "links": KNOWLEDGE_SOURCES}), 502


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
