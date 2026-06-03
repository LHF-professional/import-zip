#!/usr/bin/env python3
"""
Summarizer bot: Telegram → classify → enrich → LLM summary → Notion page.

Usage:
  python summarizer.py --setup <NOTION_PARENT_PAGE_ID>   # create Notion DB
  python summarizer.py                                   # run bot

Inputs acceptés :
  URL                          → fetch direct + Wayback fallback
  article: <URL>               → idem
  article: <mots-clés>         → GPT web search (gpt-4o-search-preview)
  livre: <titre>               → Google Books meta + Libgen → TPB → 1337x
  film: <titre>                → TMDB + LLM
  concept: <sujet>             → LLM
  magnet:?...                  → aria2c seed=0
  fichier .torrent en PJ       → idem

Env vars required:
  TELEGRAM_TOKEN, OPENAI_API_KEY, NOTION_TOKEN, NOTION_DATABASE_ID
Optional:
  TMDB_API_KEY, GOOGLE_BOOKS_API_KEY, TORRENT_TIMEOUT (défaut 300s)

Dép. système : apt install aria2
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
from urllib.parse import quote as url_quote

import requests
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import openai

# ─── Config ─────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN      = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN        = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID  = os.environ.get("NOTION_DATABASE_ID", "")
TMDB_API_KEY        = os.environ.get("TMDB_API_KEY", "")
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")
TORRENT_TIMEOUT     = int(os.environ.get("TORRENT_TIMEOUT", "300"))

CACHE_FILE        = Path("cache.json")
MAX_ARTICLE_CHARS = 32000  # ~8000 tokens
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; summarizer-bot/1.0)"}

_TPB_TRACKERS = "&".join([
    "tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce",
    "tr=udp%3A%2F%2Fopen.tracker.cl%3A1337%2Fannounce",
    "tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce",
])


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
    "film": "film",   "movie": "film",
    "livre": "livre", "book": "livre",
    "concept": "concept",
    "article": "article",
}

def classify(text):
    text = text.strip()
    m = re.match(r'^(film|movie|livre|book|concept|article)\s*:\s*(.+)$', text, re.IGNORECASE)
    if m:
        kind, identifier = _TYPE_MAP[m.group(1).lower()], m.group(2).strip()
        if kind == "article" and not re.match(r'https?://', identifier):
            return "article_search", identifier
        return kind, identifier
    if re.match(r'https?://', text):              return "article", text
    if re.match(r'magnet:\?', text, re.IGNORECASE): return "torrent", text
    if text.endswith(".torrent") and os.path.exists(text): return "torrent", text
    return "ambiguous", text


# ─── HTML extractor ────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "header", "footer", "aside"}
    def __init__(self):
        super().__init__()
        self._depth = 0
        self.texts  = []
    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP: self._depth += 1
    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth: self._depth -= 1
    def handle_data(self, data):
        if not self._depth:
            s = data.strip()
            if s: self.texts.append(s)

def _html_to_text(html):
    p = _TextExtractor()
    p.feed(html)
    return " ".join(p.texts)[:MAX_ARTICLE_CHARS]


# ─── Article fetch ───────────────────────────────────────────────────────────────

def _wayback_url(url):
    try:
        snap = requests.get("https://archive.org/wayback/available",
                            params={"url": url}, timeout=10).json()\
                       .get("archived_snapshots", {}).get("closest", {})
        if snap.get("available"): return snap["url"]
    except Exception:
        pass
    return None

def _fetch_raw(url):
    resp = requests.get(url, timeout=12, headers=_HEADERS)
    resp.raise_for_status()
    return _html_to_text(resp.text)

def fetch_article(url):
    try:
        return _fetch_raw(url)
    except Exception as err:
        archived = _wayback_url(url)
        if archived:
            try: return _fetch_raw(archived)
            except Exception: pass
        raise err


# ─── Article search (keywords → GPT web search) ─────────────────────────────────

_SEARCH_PROMPT = """\
Find the following article and return its full content.
Article: {keywords}

Reply in this exact format (no extra text):
SOURCE_URL: <url>
TITLE: <title>
AUTHOR: <author or empty>
DATE: <YYYY-MM-DD or empty>
CONTENT:
<full article text>
"""

def fetch_article_by_keywords(keywords):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={},
        messages=[{"role": "user", "content": _SEARCH_PROMPT.format(keywords=keywords)}],
    )
    raw = resp.choices[0].message.content or ""
    def _f(pat): m = re.search(pat, raw, re.IGNORECASE); return m.group(1).strip() if m else ""
    cm = re.search(r'CONTENT:\n(.+)', raw, re.DOTALL | re.IGNORECASE)
    return {
        "content": (cm.group(1).strip() if cm else raw)[:MAX_ARTICLE_CHARS],
        "url":    _f(r'SOURCE_URL:\s*(\S+)'),
        "title":  _f(r'TITLE:\s*(.+)'),
        "author": _f(r'AUTHOR:\s*(.+)'),
        "date":   _f(r'DATE:\s*(\S+)'),
    }


# ─── Book: Google Books + Libgen → TPB → 1337x ────────────────────────────────

def enrich_book(title):
    """Google Books: title / author / year / ISBN / description."""
    params = {"q": title, "maxResults": 1}
    if GOOGLE_BOOKS_API_KEY: params["key"] = GOOGLE_BOOKS_API_KEY
    r     = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=10)
    items = r.json().get("items", [])
    if not items: return {}
    info   = items[0].get("volumeInfo", {})
    idents = info.get("industryIdentifiers", [])
    isbn   = next((x["identifier"] for x in idents
                   if x["type"] in ("ISBN_13", "ISBN_10")), "")
    return {
        "title":       info.get("title", title),
        "author":      ", ".join(info.get("authors", [])),
        "year":        (info.get("publishedDate") or "")[:4],
        "description": info.get("description", ""),
        "source":      info.get("infoLink", ""),
        "isbn":        isbn,
    }


def _libgen_search(query):
    r   = requests.get(
        "http://libgen.rs/search.php",
        params={"req": query, "res": 5, "view": "simple", "phrase": 1, "column": "def"},
        timeout=15, headers=_HEADERS,
    )
    ids = re.findall(r'<td>\s*(\d{7,})\s*</td>', r.text)
    if not ids: return []
    r2 = requests.get(
        "http://libgen.rs/json.php",
        params={"ids": ",".join(ids[:5]), "fields": "id,title,author,year,extension,md5"},
        timeout=10,
    )
    return r2.json()

def _libgen_download_url(md5):
    page = requests.get(f"https://library.lol/main/{md5}", timeout=12, headers=_HEADERS)
    m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*GET\s*</a>', page.text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'href="(https?://[^"]*get\.php[^"]*md5=[^"]+)"', page.text, re.IGNORECASE)
    return m.group(1) if m else None

def _download_and_extract(url, ext="pdf"):
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        resp = requests.get(url, timeout=60, stream=True, headers=_HEADERS)
        resp.raise_for_status()
        for chunk in resp.iter_content(8192): tmp.write(chunk)
        path = tmp.name
    try:    return _extract_text_file(path)
    finally: os.unlink(path)

def _tpb_magnet(query):
    """The Pirate Bay JSON API, category 601 = ebooks."""
    r       = requests.get("https://apibay.org/q.php",
                           params={"q": query, "cat": "601"}, timeout=10, headers=_HEADERS)
    results = r.json()
    # Empty result set is [{"id":"0",...}]
    if not results or results[0].get("id") == "0": return None
    best = results[0]
    return (f"magnet:?xt=urn:btih:{best['info_hash']}"
            f"&dn={url_quote(best['name'])}&{_TPB_TRACKERS}")

def _1337x_magnet(query):
    """Scrape 1337x search for first ebook result magnet."""
    r = requests.get(
        f"https://1337x.to/search/{url_quote(query)}/1/",
        timeout=12, headers=_HEADERS,
    )
    m = re.search(r'href="(/torrent/\d+/[^"]+)"', r.text)
    if not m: return None
    page = requests.get(f"https://1337x.to{m.group(1)}", timeout=12, headers=_HEADERS)
    mag  = re.search(r'href="(magnet:\?[^"]+)"', page.text)
    return mag.group(1) if mag else None

def fetch_book_content(title, author="", isbn=""):
    """
    Fallback chain: Libgen (HTTP) → TPB (torrent) → 1337x (torrent).
    Returns extracted text. Raises ValueError if all sources fail.
    """
    query = isbn if isbn else f"{title} {author}".strip()

    # 1. Libgen direct HTTP
    try:
        results = _libgen_search(query)
        if results:
            results.sort(key=lambda b: {"epub": 0, "pdf": 1}.get(b.get("extension","").lower(), 2))
            dl = _libgen_download_url(results[0].get("md5", ""))
            if dl:
                logger.info("Book source: libgen")
                return _download_and_extract(dl, results[0].get("extension", "pdf"))
    except Exception as e:
        logger.warning("Libgen failed: %s", e)

    # 2. The Pirate Bay (cat 601 = ebooks)
    try:
        magnet = _tpb_magnet(query)
        if magnet:
            logger.info("Book source: TPB")
            return fetch_torrent(magnet)
    except Exception as e:
        logger.warning("TPB failed: %s", e)

    # 3. 1337x
    try:
        magnet = _1337x_magnet(query)
        if magnet:
            logger.info("Book source: 1337x")
            return fetch_torrent(magnet)
    except Exception as e:
        logger.warning("1337x failed: %s", e)

    raise ValueError(f"Livre introuvable sur toutes les sources (libgen / TPB / 1337x) : {query}")


# ─── Film enrichment ─────────────────────────────────────────────────────────────

def enrich_film(title):
    if not TMDB_API_KEY: return {}
    r = requests.get("https://api.themoviedb.org/3/search/movie",
                     params={"api_key": TMDB_API_KEY, "query": title}, timeout=10)
    results = r.json().get("results", [])
    if not results: return {}
    m = results[0]
    return {"title": m.get("title", title), "year": (m.get("release_date") or "")[:4],
            "overview": m.get("overview", ""),
            "source": f"https://www.themoviedb.org/movie/{m['id']}"}


# ─── Torrent fetch (aria2c, seeding disabled) ─────────────────────────────────────

def fetch_torrent(magnet_or_path):
    with tempfile.TemporaryDirectory(prefix="summarizer_") as tmpdir:
        subprocess.run([
            "aria2c",
            "--seed-time=0", "--bt-max-upload-slots=0", "--max-upload-limit=1",
            "--dir", tmpdir, "--quiet", "--console-log-level=warn",
            magnet_or_path,
        ], timeout=TORRENT_TIMEOUT, check=True)
        return _extract_from_dir(tmpdir)

def _extract_from_dir(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for f in files: all_files.append(os.path.join(root, f))
    if not all_files: raise ValueError("Aucun fichier téléchargé")
    for ext in (".epub", ".pdf", ".txt", ".text", ".md"):
        hits = [f for f in all_files if f.lower().endswith(ext)]
        if hits: return _extract_text_file(hits[0])
    raise ValueError(f"Format non supporté : {[os.path.basename(f) for f in all_files]}")

def _extract_text_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":  return _extract_pdf(path)
    if ext == ".epub": return _extract_epub(path)
    with open(path, encoding="utf-8", errors="ignore") as f: return f.read()[:MAX_ARTICLE_CHARS]

def _extract_pdf(path):
    from pdfminer.high_level import extract_text as _pdf
    return (_pdf(path) or "")[:MAX_ARTICLE_CHARS]

def _extract_epub(path):
    import ebooklib
    from ebooklib import epub as epub_lib
    book  = epub_lib.read_epub(path, options={"ignore_ncx": True})
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
- best_quote.is_real_quote = false TOUJOURS
- personal_interpretation_draft commence TOUJOURS par "[BROUILLON] "
- related_draft : titres ou concepts liés, sans certitude requise
"""

def _build_user_prompt(content_type, identifier, enriched):
    if content_type == "article":
        return f"Résume cet article.\nURL: {identifier}\n\nContenu:\n" + enriched.get("content", "")
    if content_type == "article_search":
        return (
            f"Résume cet article (trouvé par recherche web).\n"
            f"Titre: {enriched.get('title', identifier)}\nAuteur: {enriched.get('author','')}\n"
            f"Date: {enriched.get('date','')}\nURL: {enriched.get('url','')}\n\nContenu:\n"
            + enriched.get("content", "")
        )
    if content_type == "livre":
        if enriched.get("content"):
            return (
                f"Résume ce livre.\n"
                f"Titre: {enriched.get('title', identifier)}\nAuteur: {enriched.get('author','')}\n"
                f"Année: {enriched.get('year','')}\n\n"
                f"Contenu (tronqué à 32k):\n" + enriched["content"]
            )
        return (
            f"Résume ce livre pour une fiche Notion.\n"
            f"Titre: {enriched.get('title', identifier)}\nAuteur: {enriched.get('author','')}\n"
            f"Année: {enriched.get('year','')}\nDescription: {enriched.get('description','')}"
        )
    if content_type == "torrent":
        return "Identifie et résume ce document.\n\nContenu:\n" + enriched.get("content", "")
    if content_type == "film":
        return (
            f"Résume ce film pour une fiche Notion.\n"
            f"Titre: {enriched.get('title', identifier)}\nAnnée: {enriched.get('year','')}\n"
            f"Synopsis: {enriched.get('overview','')}"
        )
    return f"Explique ce concept pour une fiche Notion.\nConcept: {identifier}"

def summarize_with_llm(content_type, identifier, enriched):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(content_type, identifier, enriched)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ─── Notion ───────────────────────────────────────────────────────────────────

_NOTION_API = "https://api.notion.com/v1"
def _notion_headers(): return {"Authorization": f"Bearer {NOTION_TOKEN}",
                               "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
def _rt(t):    return [{"type":"text","text":{"content":str(t)[:2000]}}]
def _h(t, lv=2): k=f"heading_{lv}"; return {"object":"block","type":k,k:{"rich_text":_rt(t)}}
def _p(t):     return {"object":"block","type":"paragraph","paragraph":{"rich_text":_rt(t)}}
def _b(t):     return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":_rt(t)}}
def _td(t):    return {"object":"block","type":"to_do","to_do":{"rich_text":_rt(t),"checked":False}}
def _q(t):     return {"object":"block","type":"quote","quote":{"rich_text":_rt(t)}}
def _tog(t,ch):return {"object":"block","type":"toggle","toggle":{"rich_text":_rt(t),"children":ch}}
def _cl(t,e="✏️"): return {"object":"block","type":"callout","callout":{"rich_text":_rt(t),"icon":{"type":"emoji","emoji":e}}}

def _build_notion_page(db_id, summary):
    meta     = summary.get("metadata", {})
    title    = meta.get("title") or summary.get("very_short", "Untitled")
    year_str = meta.get("year", "")
    year_num = int(year_str) if year_str and year_str.isdigit() else None
    props = {
        "Name":          {"title": _rt(title)},
        "Content Type":  {"select": {"name": summary.get("content_type","concept").capitalize()}},
        "Domain":        {"select": {"name": summary.get("domain","Général")}},
        "Status":        {"select": {"name": "À lire"}},
        "Priority":      {"select": {"name": "Moyenne"}},
        "Date Consumed": {"date":   {"start": date.today().isoformat()}},
    }
    if meta.get("author"): props["Author"] = {"rich_text": _rt(meta["author"])}
    if year_num:           props["Year"]   = {"number": year_num}
    if meta.get("source"): props["Source"] = {"url": meta["source"][:2000]}
    if summary.get("tags"): props["Tags"]  = {"multi_select": [{"name":t} for t in summary["tags"][:5]]}
    bq  = summary.get("best_quote", {})
    qt  = bq.get("text", "")
    if qt and not bq.get("is_real_quote", True): qt += " *(non vérifié)*"
    ch = [_h("Overview"), _p(summary.get("very_short","")), _p(summary.get("main_summary","")), _h("Key Ideas")]
    for idea in summary.get("key_ideas",[]):
        ch.append(_tog(idea.get("theme",""), [_b(p) for p in idea.get("points",[])]))
    ch.append(_h("Keywords"))
    for kw in summary.get("keywords",[]): ch.append(_b(kw))
    ch.append(_h("Useful Concepts"))
    for c in summary.get("useful_concepts",[]): ch.append(_b(c))
    ch.append(_h("Actionable"))
    for a in summary.get("actionable",[]): ch.append(_td(a))
    ch += [_h("Best Quote / Moment"), _q(qt), _h("Personal Interpretation"),
           _cl(summary.get("personal_interpretation_draft","[BROUILLON] À compléter."),"✏️"),
           _h("Related Notes")]
    for r in summary.get("related_draft",[]): ch.append(_b(r))
    return {"parent":{"database_id":db_id},"properties":props,"children":ch}

def _create_notion_page(db_id, summary):
    r = requests.post(f"{_NOTION_API}/pages", headers=_notion_headers(),
                      json=_build_notion_page(db_id, summary))
    r.raise_for_status()
    return r.json()["url"]

def setup_notion(parent_page_id):
    payload = {
        "parent": {"type":"page_id","page_id":parent_page_id},
        "title":  [{"type":"text","text":{"content":"Bibliothèque personnelle"}}],
        "properties": {
            "Name":{"title":{}},
            "Content Type":{"select":{"options":[{"name":"Article"},{"name":"Livre"},{"name":"Film"},{"name":"Concept"}]}},
            "Domain":{"select":{"options":[{"name":"Technologie"},{"name":"Science"},{"name":"Philosophie"},{"name":"Business"},{"name":"Psychologie"},{"name":"Histoire"},{"name":"Art"},{"name":"Santé"},{"name":"Général"}]}},
            "Author":{"rich_text":{}},"Year":{"number":{}},"Source":{"url":{}},"Tags":{"multi_select":{}},
            "Status":{"select":{"options":[{"name":"À lire"},{"name":"En cours"},{"name":"Lu"},{"name":"Archivé"}]}},
            "Priority":{"select":{"options":[{"name":"Haute"},{"name":"Moyenne"},{"name":"Basse"}]}},
            "Date Consumed":{"date":{}},
        },
    }
    r  = requests.post(f"{_NOTION_API}/databases", headers=_notion_headers(), json=payload)
    r.raise_for_status()
    db = r.json()
    print(f"✅ Base Notion créée\n   ID  : {db['id']}\n   URL : {db.get('url','')}")
    print(f"\nAjoute dans ton .env :\n   NOTION_DATABASE_ID={db['id']}")


# ─── Core pipeline ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

def process(text):
    content_type, identifier = classify(text)
    if content_type == "ambiguous":
        return None, False, "ambiguous", identifier
    ck    = _cache_key(content_type, identifier)
    cache = _load_cache()
    if ck in cache:
        return cache[ck]["notion_url"], True, content_type, identifier
    enriched = {}
    if content_type == "article":
        enriched["content"] = fetch_article(identifier)
    elif content_type == "article_search":
        enriched = fetch_article_by_keywords(identifier)
    elif content_type == "livre":
        enriched = enrich_book(identifier)
        try:
            enriched["content"] = fetch_book_content(
                enriched.get("title", identifier),
                enriched.get("author", ""),
                enriched.get("isbn", ""),
            )
        except Exception as e:
            logger.warning("Toutes les sources de livre ont échoué (%s) — métadonnées seules", e)
    elif content_type == "film":
        enriched = enrich_film(identifier)
    elif content_type == "torrent":
        enriched["content"] = fetch_torrent(identifier)
    summary = summarize_with_llm(content_type, identifier, enriched)
    if "best_quote" in summary:
        summary["best_quote"]["is_real_quote"] = False
    if content_type == "article_search" and enriched.get("url"):
        summary.setdefault("metadata", {})["source"] = enriched["url"]
    notion_url = _create_notion_page(NOTION_DATABASE_ID, summary)
    cache[ck]  = {"notion_url": notion_url, "created_at": datetime.now().isoformat()}
    _save_cache(cache)
    return notion_url, False, content_type, identifier


# ─── Telegram ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_HELP = (
    "Envoie-moi :\n"
    "• URL d’article → fetch + Wayback fallback\n"
    "• `article: politico - melanchon 2027` → recherche web\n"
    "• `livre: Atomic Habits` → Google Books + Libgen → TPB → 1337x\n"
    "• `film: Inception` · `concept: biais cognitifs`\n"
    "• Lien magnet ou fichier .torrent en PJ"
)
_LABELS = {
    "article":"\U0001f4f0 Article", "article_search":"\U0001f50d Article (recherche)",
    "film":"\U0001f3ac Film", "livre":"\U0001f4da Livre",
    "concept":"\U0001f4a1 Concept", "torrent":"\U0001f9f2 Torrent",
}

async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text or text in ("/start","/help"):
        await update.message.reply_text(_HELP); return
    await _run_pipeline(update, text)

async def _handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not (doc.file_name or "").endswith(".torrent"):
        await update.message.reply_text("❌ Seuls les fichiers .torrent sont acceptés."); return
    tg_file = await context.bot.get_file(doc.file_id)
    with tempfile.NamedTemporaryFile(suffix=".torrent", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name); path = tmp.name
    try: await _run_pipeline(update, path)
    finally:
        try: os.unlink(path)
        except OSError: pass

async def _run_pipeline(update: Update, text: str):
    await update.message.reply_text("⏳ Traitement en cours…")
    t0 = time.monotonic()
    try:
        notion_url, from_cache, content_type, identifier = process(text)
    except Exception as exc:
        logger.exception("pipeline error")
        await update.message.reply_text(f"❌ Erreur : {exc}"); return
    if content_type == "ambiguous":
        await update.message.reply_text(
            f"❓ Ambiguïté : *{identifier}*\n\nPrécise :\n"
            f"• `film: {identifier}`\n• `livre: {identifier}`\n• `concept: {identifier}`",
            parse_mode="Markdown"); return
    elapsed = round(time.monotonic() - t0, 1)
    label   = _LABELS.get(content_type, content_type)
    note    = " *(cache)*" if from_cache else f" *({elapsed}s)*"
    await update.message.reply_text(f"✅ {label}{note}\n{notion_url}", parse_mode="Markdown")

def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, _handle_text))
    app.add_handler(MessageHandler(filters.Document.FileExtension("torrent"), _handle_document))
    logger.info("Summarizer bot started")
    app.run_polling()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", metavar="PARENT_PAGE_ID", help="Créer la base Notion")
    args = parser.parse_args()
    if args.setup:
        setup_notion(args.setup)
    else:
        if not NOTION_DATABASE_ID:
            sys.exit("NOTION_DATABASE_ID manquant. Lance d'abord --setup <page_id>")
        run_bot()
