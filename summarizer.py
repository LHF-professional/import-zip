#!/usr/bin/env python3
"""
Summarizer bot: Telegram → classify → enrich → LLM summary → Notion page.

Usage:
  python summarizer.py --setup <NOTION_PARENT_PAGE_ID>   # create Notion DB
  python summarizer.py                                   # run bot

Inputs acceptés :
  - URL d'article
  - Lien magnet  (magnet:?xt=urn:btih:...)
  - Fichier .torrent envoyé en pièce jointe Telegram
  - film: <titre>  /  livre: <titre>  /  concept: <sujet>

Env vars required:
  TELEGRAM_TOKEN, OPENAI_API_KEY, NOTION_TOKEN, NOTION_DATABASE_ID
Optional:
  TMDB_API_KEY, GOOGLE_BOOKS_API_KEY, TORRENT_TIMEOUT (défaut: 300s)

Dépendances système :
  apt install aria2
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import re
import subprocess
import tempfile
import time
from datetime import datetime, date
from html.parser import HTMLParser
from pathlib import Path

import requests
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import openai

# ─── Config ─────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")
TORRENT_TIMEOUT = int(os.environ.get("TORRENT_TIMEOUT", "300"))

CACHE_FILE = Path("cache.json")
MAX_ARTICLE_CHARS = 32000  # ~8000 tokens


# ─── Cache ───────────────────────────────────────────────────────────────────

def _load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_key(content_type, identifier):
    raw = f"{content_type}:{identifier.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Classification ──────────────────────────────────────────────────────────

_TYPE_MAP = {
    "film": "film", "movie": "film",
    "livre": "livre", "book": "livre",
    "concept": "concept",
    "article": "article",
}


def classify(text):
    """Returns (content_type, identifier).
    content_type can be 'ambiguous' if no explicit prefix and not a URL/magnet.
    """
    text = text.strip()
    # Explicit prefix: film:/livre:/concept:/article:
    m = re.match(r'^(film|movie|livre|book|concept|article)\s*:\s*(.+)$', text, re.IGNORECASE)
    if m:
        return _TYPE_MAP[m.group(1).lower()], m.group(2).strip()
    # Magnet link
    if re.match(r'magnet:\?', text, re.IGNORECASE):
        return "torrent", text
    # .torrent file path (set by document handler)
    if text.endswith(".torrent") and os.path.exists(text):
        return "torrent", text
    # HTTP(S) article
    if re.match(r'https?://', text):
        return "article", text
    return "ambiguous", text


# ─── Enrichment ──────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strips HTML tags, skips script/style/nav blocks."""
    _SKIP = {"script", "style", "nav", "header", "footer", "aside"}

    def __init__(self):
        super().__init__()
        self._depth = 0
        self.texts = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            s = data.strip()
            if s:
                self.texts.append(s)


def fetch_article(url):
    resp = requests.get(
        url, timeout=12,
        headers={"User-Agent": "Mozilla/5.0 (compatible; summarizer-bot/1.0)"}
    )
    resp.raise_for_status()
    parser = _TextExtractor()
    parser.feed(resp.text)
    return " ".join(parser.texts)[:MAX_ARTICLE_CHARS]


def enrich_film(title):
    if not TMDB_API_KEY:
        return {}
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": TMDB_API_KEY, "query": title},
        timeout=10,
    )
    results = r.json().get("results", [])
    if not results:
        return {}
    m = results[0]
    return {
        "title": m.get("title", title),
        "year": (m.get("release_date") or "")[:4],
        "overview": m.get("overview", ""),
        "source": f"https://www.themoviedb.org/movie/{m['id']}",
    }


def enrich_book(title):
    params = {"q": title, "maxResults": 1}
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY
    r = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=10)
    items = r.json().get("items", [])
    if not items:
        return {}
    info = items[0].get("volumeInfo", {})
    return {
        "title": info.get("title", title),
        "author": ", ".join(info.get("authors", [])),
        "year": (info.get("publishedDate") or "")[:4],
        "description": info.get("description", ""),
        "source": info.get("infoLink", ""),
    }


# ─── Torrent fetch ─────────────────────────────────────────────────────────────

def fetch_torrent(magnet_or_torrent_path):
    """Download via aria2c with seeding fully disabled. Returns extracted text."""
    with tempfile.TemporaryDirectory(prefix="summarizer_") as tmpdir:
        cmd = [
            "aria2c",
            "--seed-time=0",           # stop seeding immediately after download
            "--bt-max-upload-slots=0", # no upload slots allocated
            "--max-upload-limit=1",    # hard cap 1 byte/s (belt + suspenders)
            "--dir", tmpdir,
            "--quiet",
            "--console-log-level=warn",
            magnet_or_torrent_path,
        ]
        subprocess.run(cmd, timeout=TORRENT_TIMEOUT, check=True)
        return _extract_from_dir(tmpdir)


def _extract_from_dir(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            all_files.append(os.path.join(root, f))

    if not all_files:
        raise ValueError("Aucun fichier téléchargé")

    for ext in (".epub", ".pdf", ".txt", ".text", ".md"):
        matches = [f for f in all_files if f.lower().endswith(ext)]
        if matches:
            return _extract_text_file(matches[0])

    names = [os.path.basename(f) for f in all_files]
    raise ValueError(f"Format non supporté parmi : {names}")


def _extract_text_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".epub":
        return _extract_epub(path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:MAX_ARTICLE_CHARS]


def _extract_pdf(path):
    from pdfminer.high_level import extract_text as pdf_to_text
    return (pdf_to_text(path) or "")[:MAX_ARTICLE_CHARS]


def _extract_epub(path):
    import ebooklib
    from ebooklib import epub as epub_lib
    book = epub_lib.read_epub(path, options={"ignore_ncx": True})
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        p = _TextExtractor()
        p.feed(item.get_content().decode("utf-8", errors="ignore"))
        parts.append(" ".join(p.texts))
    return " ".join(parts)[:MAX_ARTICLE_CHARS]


# ─── LLM summary ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Tu es un assistant de synthèse structurée. Tu réponds UNIQUEMENT avec un objet JSON valide,
sans markdown, sans texte autour.

Schéma JSON à respecter exactement :
{
  "very_short": "1 phrase max",
  "main_summary": "3-5 phrases",
  "key_ideas": [{"theme": "string", "points": ["string"]}],
  "keywords": ["string"],
  "useful_concepts": ["string"],
  "actionable": ["string"],
  "best_quote": {"text": "string", "is_real_quote": false},
  "personal_interpretation_draft": "string",
  "related_draft": ["string"],
  "metadata": {"title": "string", "author": "string", "year": "string", "source": "string"},
  "tags": ["string"],
  "domain": "string",
  "content_type": "article|livre|film|concept"
}

RÈGLES CRITIQUES :
- best_quote.is_real_quote = false TOUJOURS (ne jamais inventer une citation textuelle)
- personal_interpretation_draft commence TOUJOURS par "[BROUILLON] "
- related_draft : titres ou concepts liés, sans certitude requise
"""


def _build_user_prompt(content_type, identifier, enriched):
    if content_type in ("article", "torrent"):
        intro = (
            "Résume cet article.\nURL: " + identifier
            if content_type == "article"
            else "Identifie et résume ce document (livre, film, concept ou article) pour une fiche Notion."
        )
        return intro + "\n\nContenu:\n" + enriched.get("content", "")
    if content_type == "film":
        return (
            f"Résume ce film pour une fiche Notion personnelle.\n"
            f"Titre: {enriched.get('title', identifier)}\n"
            f"Année: {enriched.get('year', '')}\n"
            f"Synopsis: {enriched.get('overview', '')}"
        )
    if content_type == "livre":
        return (
            f"Résume ce livre pour une fiche Notion personnelle.\n"
            f"Titre: {enriched.get('title', identifier)}\n"
            f"Auteur: {enriched.get('author', '')}\n"
            f"Année: {enriched.get('year', '')}\n"
            f"Description: {enriched.get('description', '')}"
        )
    # concept
    return f"Explique ce concept pour une fiche Notion personnelle.\nConcept: {identifier}"


def summarize_with_llm(content_type, identifier, enriched):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(content_type, identifier, enriched)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ─── Notion ───────────────────────────────────────────────────────────────────

_NOTION_API = "https://api.notion.com/v1"


def _notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }


def _rt(text):
    return [{"type": "text", "text": {"content": str(text)[:2000]}}]


def _heading(text, level=2):
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rt(text)}}


def _para(text):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}


def _bullet(text):
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _rt(text)}}


def _todo(text):
    return {"object": "block", "type": "to_do",
            "to_do": {"rich_text": _rt(text), "checked": False}}


def _quote(text):
    return {"object": "block", "type": "quote", "quote": {"rich_text": _rt(text)}}


def _toggle(text, children):
    return {"object": "block", "type": "toggle",
            "toggle": {"rich_text": _rt(text), "children": children}}


def _callout(text, emoji="✏️"):
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": _rt(text), "icon": {"type": "emoji", "emoji": emoji}}}


def _build_notion_page(db_id, summary):
    meta = summary.get("metadata", {})
    title = meta.get("title") or summary.get("very_short", "Untitled")
    year_str = meta.get("year", "")
    year_num = int(year_str) if year_str and year_str.isdigit() else None

    props = {
        "Name": {"title": _rt(title)},
        "Content Type": {"select": {"name": summary.get("content_type", "concept").capitalize()}},
        "Domain": {"select": {"name": summary.get("domain", "Général")}},
        "Status": {"select": {"name": "À lire"}},
        "Priority": {"select": {"name": "Moyenne"}},
        "Date Consumed": {"date": {"start": date.today().isoformat()}},
    }
    if meta.get("author"):
        props["Author"] = {"rich_text": _rt(meta["author"])}
    if year_num:
        props["Year"] = {"number": year_num}
    if meta.get("source"):
        props["Source"] = {"url": meta["source"][:2000]}
    if summary.get("tags"):
        props["Tags"] = {"multi_select": [{"name": t} for t in summary["tags"][:5]]}

    bq = summary.get("best_quote", {})
    quote_text = bq.get("text", "")
    if quote_text and not bq.get("is_real_quote", True):
        quote_text += " *(non vérifié)*"

    children = [
        _heading("Overview"),
        _para(summary.get("very_short", "")),
        _para(summary.get("main_summary", "")),
        _heading("Key Ideas"),
    ]
    for idea in summary.get("key_ideas", []):
        children.append(_toggle(idea.get("theme", ""), [_bullet(p) for p in idea.get("points", [])]))

    children.append(_heading("Keywords"))
    for kw in summary.get("keywords", []):
        children.append(_bullet(kw))

    children.append(_heading("Useful Concepts"))
    for c in summary.get("useful_concepts", []):
        children.append(_bullet(c))

    children.append(_heading("Actionable"))
    for a in summary.get("actionable", []):
        children.append(_todo(a))

    children += [
        _heading("Best Quote / Moment"),
        _quote(quote_text),
        _heading("Personal Interpretation"),
        _callout(summary.get("personal_interpretation_draft", "[BROUILLON] À compléter."), "✏️"),
        _heading("Related Notes"),
    ]
    for r in summary.get("related_draft", []):
        children.append(_bullet(r))

    return {"parent": {"database_id": db_id}, "properties": props, "children": children}


def _create_notion_page(db_id, summary):
    payload = _build_notion_page(db_id, summary)
    r = requests.post(f"{_NOTION_API}/pages", headers=_notion_headers(), json=payload)
    r.raise_for_status()
    return r.json()["url"]


def setup_notion(parent_page_id):
    """Create the Notion database with all required properties."""
    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": "Bibliothèque personnelle"}}],
        "properties": {
            "Name": {"title": {}},
            "Content Type": {"select": {"options": [
                {"name": "Article"}, {"name": "Livre"},
                {"name": "Film"}, {"name": "Concept"},
            ]}},
            "Domain": {"select": {"options": [
                {"name": "Technologie"}, {"name": "Science"}, {"name": "Philosophie"},
                {"name": "Business"}, {"name": "Psychologie"}, {"name": "Histoire"},
                {"name": "Art"}, {"name": "Santé"}, {"name": "Général"},
            ]}},
            "Author": {"rich_text": {}},
            "Year": {"number": {}},
            "Source": {"url": {}},
            "Tags": {"multi_select": {}},
            "Status": {"select": {"options": [
                {"name": "À lire"}, {"name": "En cours"},
                {"name": "Lu"}, {"name": "Archivé"},
            ]}},
            "Priority": {"select": {"options": [
                {"name": "Haute"}, {"name": "Moyenne"}, {"name": "Basse"},
            ]}},
            "Date Consumed": {"date": {}},
        },
    }
    r = requests.post(f"{_NOTION_API}/databases", headers=_notion_headers(), json=payload)
    r.raise_for_status()
    db = r.json()
    print(f"✅ Base Notion créée")
    print(f"   ID  : {db['id']}")
    print(f"   URL : {db.get('url', '')}")
    print(f"\nAjoute dans ton .env :")
    print(f"   NOTION_DATABASE_ID={db['id']}")


# ─── Core pipeline ────────────────────────────────────────────────────────────

def process(text):
    """
    Full pipeline for a user message.
    Returns (notion_url, from_cache, content_type, identifier).
    content_type == 'ambiguous' → needs clarification, notion_url is None.
    """
    content_type, identifier = classify(text)

    if content_type == "ambiguous":
        return None, False, "ambiguous", identifier

    ck = _cache_key(content_type, identifier)
    cache = _load_cache()
    if ck in cache:
        return cache[ck]["notion_url"], True, content_type, identifier

    enriched = {}
    if content_type == "article":
        enriched["content"] = fetch_article(identifier)
    elif content_type == "torrent":
        enriched["content"] = fetch_torrent(identifier)
    elif content_type == "film":
        enriched = enrich_film(identifier)
    elif content_type == "livre":
        enriched = enrich_book(identifier)

    summary = summarize_with_llm(content_type, identifier, enriched)

    # Safety net: force is_real_quote = false
    if "best_quote" in summary:
        summary["best_quote"]["is_real_quote"] = False

    notion_url = _create_notion_page(NOTION_DATABASE_ID, summary)

    cache[ck] = {"notion_url": notion_url, "created_at": datetime.now().isoformat()}
    _save_cache(cache)

    return notion_url, False, content_type, identifier


# ─── Telegram bot ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_HELP = (
    "Envoie-moi :\n"
    "• Une URL d'article\n"
    "• Un lien magnet (magnet:?...)\n"
    "• Un fichier .torrent en pièce jointe\n"
    "• `film: <titre>` · `livre: <titre>` · `concept: <sujet>`\n\n"
    "Pour lever une ambiguïté :\n"
    "  `film: Dune`  vs  `livre: Dune`"
)

_LABELS = {
    "article": "📰 Article",
    "film": "🎬 Film",
    "livre": "📚 Livre",
    "concept": "💡 Concept",
    "torrent": "🧲 Torrent",
}


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text or text in ("/start", "/help"):
        await update.message.reply_text(_HELP)
        return
    await _run_pipeline(update, text)


async def _handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not (doc.file_name or "").endswith(".torrent"):
        await update.message.reply_text("❌ Seuls les fichiers .torrent sont acceptés.")
        return

    tg_file = await context.bot.get_file(doc.file_id)
    with tempfile.NamedTemporaryFile(suffix=".torrent", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        torrent_path = tmp.name

    try:
        await _run_pipeline(update, torrent_path)
    finally:
        try:
            os.unlink(torrent_path)
        except OSError:
            pass


async def _run_pipeline(update: Update, text: str):
    await update.message.reply_text("⏳ Traitement en cours…")
    t0 = time.monotonic()
    try:
        notion_url, from_cache, content_type, identifier = process(text)
    except Exception as exc:
        logger.exception("pipeline error")
        await update.message.reply_text(f"❌ Erreur : {exc}")
        return

    if content_type == "ambiguous":
        await update.message.reply_text(
            f"❓ Ambiguïté : *{identifier}*\n\nPrécise le type :\n"
            f"• `film: {identifier}`\n"
            f"• `livre: {identifier}`\n"
            f"• `concept: {identifier}`",
            parse_mode="Markdown",
        )
        return

    elapsed = round(time.monotonic() - t0, 1)
    note = " *(cache)*" if from_cache else f" *({elapsed}s)*"
    label = _LABELS.get(content_type, content_type)
    await update.message.reply_text(
        f"✅ {label}{note}\n{notion_url}",
        parse_mode="Markdown",
    )


def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, _handle_text))
    app.add_handler(MessageHandler(filters.Document.FileExtension("torrent"), _handle_document))
    logger.info("Summarizer bot started")
    app.run_polling()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarizer bot")
    parser.add_argument(
        "--setup", metavar="PARENT_PAGE_ID",
        help="Créer la base de données Notion dans la page indiquée",
    )
    args = parser.parse_args()

    if args.setup:
        setup_notion(args.setup)
    else:
        if not NOTION_DATABASE_ID:
            sys.exit("NOTION_DATABASE_ID manquant. Lance d'abord --setup <page_id>")
        run_bot()
