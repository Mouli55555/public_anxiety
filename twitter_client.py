# twitter_client.py
# Legacy module name retained so the rest of the app does not need a wider refactor.
# Live post fetches now use Bluesky's public AppView search instead of the paid X API.

import math
import re

import pandas as pd
import requests

import config

REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_BLUESKY_API_BASE_URL = "https://api.bsky.app"
PUBLIC_BLUESKY_API_BASE_URL = "https://public.api.bsky.app"
SEARCH_ENDPOINT = "/xrpc/app.bsky.feed.searchPosts"
REQUEST_HEADERS = {
    "User-Agent": "PublicAnxietyDashboard/1.0"
}
QUERY_STOP_WORDS = {
    "and",
    "or",
    "not",
    "lang",
    "from",
    "since",
    "until",
    "near",
    "filter",
    "is",
}


def _api_base_urls():
    configured = (config.BLUESKY_API_BASE_URL or DEFAULT_BLUESKY_API_BASE_URL).strip()
    ordered = [configured, DEFAULT_BLUESKY_API_BASE_URL, PUBLIC_BLUESKY_API_BASE_URL]
    deduped = []
    seen = set()

    for url in ordered:
        normalized = url.rstrip("/")
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)

    return deduped


def _extract_search_terms(query):
    quoted_terms = [match.strip().lower() for match in re.findall(r'"([^"]+)"', query) if match.strip()]
    token_terms = []

    for token in re.findall(r"[A-Za-z0-9_#@'-]+", query.lower()):
        normalized = token.strip("()[]{}.,:;!?")
        normalized = normalized.lstrip("#@")
        if len(normalized) < 2 or normalized in QUERY_STOP_WORDS:
            continue
        token_terms.append(normalized)

    deduped = []
    seen = set()
    for term in quoted_terms + token_terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)

    return deduped[:5]


def _build_candidate_queries(query):
    raw_query = (query or "").strip()
    search_terms = _extract_search_terms(raw_query)
    if not search_terms:
        return [raw_query] if raw_query else []

    if len(search_terms) == 1:
        return [search_terms[0]]

    return search_terms


def _matches_query(text, query):
    lowered_text = (text or "").lower()
    raw_query = (query or "").strip().lower()
    if raw_query and raw_query in lowered_text:
        return True

    search_terms = _extract_search_terms(query)
    if not search_terms:
        return True

    require_all_terms = " and " in raw_query and " or " not in raw_query
    if require_all_terms:
        return all(term in lowered_text for term in search_terms)

    return any(term in lowered_text for term in search_terms)


def _looks_english(post):
    langs = (((post or {}).get("record") or {}).get("langs") or [])
    return not langs or "en" in langs


def _format_post(post):
    record = (post or {}).get("record") or {}
    author = (post or {}).get("author") or {}
    text = (record.get("text") or "").strip()
    if not text:
        return None

    handle = (author.get("handle") or "unknown.bsky.social").strip()
    email_local = re.sub(r"[^A-Za-z0-9._-]", "_", handle) or "unknown"

    return {
        "Name": (author.get("displayName") or handle).strip(),
        "Email": f"{email_local}@bsky.invalid",
        "tweet": text,
    }


def _fetch_search_results(session, search_query, limit):
    params = {
        "q": search_query,
        "lang": "en",
        "limit": max(1, min(int(limit), 100)),
    }
    errors = []

    for base_url in _api_base_urls():
        url = f"{base_url}{SEARCH_ENDPOINT}"
        try:
            response = session.get(
                url,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            errors.append(f"{base_url}: {exc}")
            continue

        if response.status_code == 200:
            return response.json(), None

        errors.append(f"{base_url}: HTTP {response.status_code}")

    if errors:
        return None, (
            "The live X(twitter) source could not be reached right now. "
            + " Tried: "
            + "; ".join(errors)
        )

    return None, "The live X(twitter) source could not be reached right now."


def fetch_recent_tweets(query, max_results=50):
    """
    Fetches recent public X(twitter) posts for the live analysis tab.
    Returns a tuple: (DataFrame, error_message).
    If successful, error_message will be None.
    If fails, DataFrame will be None.
    """
    raw_query = (query or "").strip()
    if not raw_query:
        return None, "Enter a search query before fetching live posts."

    candidate_queries = _build_candidate_queries(raw_query)
    if not candidate_queries:
        return None, "Could not extract a usable search query."

    max_results = max(1, min(int(max_results), 100))
    per_query_limit = min(100, max(10, math.ceil((max_results * 2) / len(candidate_queries))))

    rows = []
    seen_uris = set()
    had_successful_request = False
    last_error = None

    with requests.Session() as session:
        for candidate_query in candidate_queries:
            payload, error = _fetch_search_results(session, candidate_query, per_query_limit)
            if error:
                last_error = error
                continue

            had_successful_request = True

            for post in payload.get("posts", []):
                uri = post.get("uri")
                if not uri or uri in seen_uris or not _looks_english(post):
                    continue

                formatted = _format_post(post)
                if not formatted or not _matches_query(formatted["tweet"], raw_query):
                    continue

                rows.append(formatted)
                seen_uris.add(uri)

                if len(rows) >= max_results:
                    return pd.DataFrame(rows), None

    if rows:
        return pd.DataFrame(rows), None

    if had_successful_request:
        return pd.DataFrame(columns=["Name", "Email", "tweet"]), None

    return None, last_error or "Unable to fetch live posts right now."
