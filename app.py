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

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or "").strip()
GROQ_MODEL = (os.getenv("GROQ_MODEL") or "").strip()
if not GROQ_MODEL:
    old_model = os.getenv("GROK_MODEL", "").strip()
    GROQ_MODEL = old_model if old_model and not old_model.lower().startswith("grok-") else "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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
            "provider": "Groq",
            "model": GROQ_MODEL,
            "has_api_key": bool(GROQ_API_KEY),
            "sources": KNOWLEDGE_SOURCES,
        }
    )


@app.post("/api/chat")
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is missing. Add it in your .env file."}), 500

    data = request.get_json(silent=True) or {}
    messages = normalize_messages(data.get("messages", []))

    if not messages:
        return jsonify({"error": "No messages received."}), 400

    latest_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    website_context = build_website_context(latest_user_message)

    context_prompt = (
        "Use this official website context when it is relevant. "
        "Do not invent details that are not present here.\n\n"
        f"{website_context}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_prompt},
            *messages,
        ],
        "temperature": 0.4,
        "max_completion_tokens": 1200,
        "stream": False,
    }

    try:
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
            return jsonify({"error": "Groq API error", "details": error_detail}), response.status_code

        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    except requests.Timeout:
        return jsonify({"error": "Groq API request timed out. Please try again."}), 504
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
