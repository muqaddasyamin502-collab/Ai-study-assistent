import os
import re
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

def clean_env_value(value):
    value = (value or "").strip().strip("\"'")
    if "=" in value:
        value = value.split("=", 1)[1].strip().strip("\"'")
    if ":" in value:
        value = value.split(":", 1)[1].strip().strip("\"',{} ")
    return value


GROQ_API_KEY = clean_env_value(os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))
GROQ_MODEL = clean_env_value(os.getenv("GROQ_MODEL"))
if not GROQ_MODEL:
    old_model = clean_env_value(os.getenv("GROK_MODEL"))
    GROQ_MODEL = old_model if old_model and not old_model.lower().startswith("grok-") else "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = clean_env_value(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = clean_env_value(os.getenv("GEMINI_MODEL")) or "gemini-3.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CLAUDE_API_KEY = clean_env_value(os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = clean_env_value(os.getenv("CLAUDE_MODEL")) or "claude-haiku-4-5"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

MAX_OUTPUT_TOKENS = int(clean_env_value(os.getenv("MAX_OUTPUT_TOKENS")) or "600")

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


def context_message(website_context):
    if not website_context:
        return "No live website context was needed for this question."
    return (
        "Use this official website context when it is relevant. "
        "Do not invent details that are not present here.\n\n"
        f"{website_context}"
    )


def call_groq(messages, context_prompt):
    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key is missing.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_prompt},
            *messages,
        ],
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


def call_gemini(messages, context_prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is missing.")

    contents = []
    for msg in messages:
        contents.append(
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}],
            }
        )

    payload = {
        "systemInstruction": {
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{context_prompt}"}]
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


def call_claude(messages, context_prompt):
    if not CLAUDE_API_KEY:
        raise RuntimeError("Claude API key is missing.")

    payload = {
        "model": CLAUDE_MODEL,
        "system": f"{SYSTEM_PROMPT}\n\n{context_prompt}",
        "messages": messages,
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


def generate_reply(messages, context_prompt):
    providers = [
        ("Groq", call_groq),
        ("Gemini", call_gemini),
        ("Claude", call_claude),
    ]
    errors = []
    for provider, caller in providers:
        try:
            reply = caller(messages, context_prompt)
            return provider, reply
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            print(f"{provider} API error: {exc}", flush=True)
    raise RuntimeError(" | ".join(errors))


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "providers": {
                "groq": {"model": GROQ_MODEL, "has_api_key": bool(GROQ_API_KEY)},
                "gemini": {"model": GEMINI_MODEL, "has_api_key": bool(GEMINI_API_KEY)},
                "claude": {"model": CLAUDE_MODEL, "has_api_key": bool(CLAUDE_API_KEY)},
            },
            "sources": KNOWLEDGE_SOURCES,
        }
    )


@app.post("/api/chat")
def chat():
    if not any([GROQ_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY]):
        return jsonify({"error": "Add at least one API key: GROQ_API_KEY, GEMINI_API_KEY, or CLAUDE_API_KEY."}), 500

    data = request.get_json(silent=True) or {}
    messages = normalize_messages(data.get("messages", []))

    if not messages:
        return jsonify({"error": "No messages received."}), 400

    latest_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not re.search(r"[A-Za-z0-9\u0600-\u06FF]", latest_user_message):
        return jsonify({"reply": "Please type a clear question so I can help you."})

    context_prompt = context_message(build_website_context(latest_user_message))

    try:
        provider, reply = generate_reply(messages, context_prompt)
        return jsonify({"provider": provider, "reply": reply})

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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
