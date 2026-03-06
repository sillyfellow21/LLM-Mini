import requests, re, sys, os
sys.path.insert(0, os.path.expanduser("~/MiniLLM"))

SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
HEADERS     = {"User-Agent": "MiniLLM/1.0 (educational project)"}

SKIP_TASKS        = {"story", "poetry"}
DIRECT_WIKI_TASKS = {"qa"}
HYBRID_TASKS      = {"farmer"}


def clean_query(user_input):
    """
    Remove question words for better Wikipedia search.
    'what is earth' → 'earth'
    'who is einstein' → 'einstein'
    """
    text = user_input.lower().strip()
    prefixes = [
        "what is a ", "what is an ", "what is the ", "what is ",
        "what are the ", "what are ",
        "who is the ", "who is a ", "who is ",
        "who was the ", "who was ",
        "where is the ", "where is ",
        "when is the ", "when is ",
        "when was the ", "when was ",
        "why is the ", "why is ",
        "how is the ", "how is ",
        "how does ", "how do ",
        "tell me about the ", "tell me about ",
        "explain the ", "explain ",
        "define the ", "define ",
        "describe the ", "describe ",
        "what was the ", "what was ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text.rstrip("?!.,").strip()


def search_wikipedia(query, max_chars=500):
    """Search Wikipedia and return a clean summary."""
    try:
        cleaned = clean_query(query)
        print(f"[Wiki] Searching: '{cleaned}' (from: '{query}')")

        data = requests.get(SEARCH_URL, headers=HEADERS, timeout=8, params={
            "action": "opensearch",
            "search": cleaned,
            "limit":  3,
            "format": "json"
        }).json()

        titles = data[1] if len(data) > 1 else []
        if not titles:
            print(f"[Wiki] No results for '{cleaned}'")
            return None

        for title in titles:
            resp    = requests.get(
                f"{SUMMARY_URL}/{requests.utils.quote(title)}",
                headers=HEADERS, timeout=8
            ).json()
            extract = resp.get("extract", "").strip()
            extract = re.sub(r"\s+", " ", extract)

            if len(extract) < 50:
                continue

            # Clean pronunciation guides and parentheticals
            extract = re.sub(r"\(\/.*?\/\)", "", extract)
            extract = re.sub(r"\(.*?\)", "", extract)
            extract = re.sub(r"\s+", " ", extract).strip()

            # Trim to max_chars at sentence boundary
            if len(extract) > max_chars:
                extract = extract[:max_chars]
                last    = extract.rfind(".")
                extract = extract[:last + 1] if last > 100 else extract

            print(f"[Wiki] Found: '{title}' — {len(extract)} chars")
            return extract

    except Exception as e:
        print(f"[Wiki] Failed: {e}")
    return None


def format_wiki_answer(wiki_text):
    """Format Wikipedia result as a clean direct answer."""
    if not wiki_text:
        return "I couldn't find reliable information on that topic. Please try rephrasing your question."
    answer = wiki_text.strip()
    if not answer.endswith("."):
        answer += "."
    return answer


def build_farmer_prompt(wiki_text, user_input, prompts):
    """Hybrid: inject wiki facts into farmer prompt."""
    cfg = prompts["farmer"]
    if wiki_text:
        return (
            f"### Context: {wiki_text}\n\n"
            f"{cfg['prefix']}{user_input}{cfg['response']}"
        )
    return cfg["prefix"] + user_input + cfg["response"]


def build_prompt(task, user_input, prompts):
    """
    Main entry point called by chat_controller.
    Returns: (prompt_or_answer, used_wiki, is_direct)

    is_direct=True  → return answer directly, skip model
    is_direct=False → pass prompt to model for generation
    """
    # Story / Poetry — model only
    if task in SKIP_TASKS:
        cfg = prompts[task]
        return cfg["prefix"] + user_input + cfg["response"], False, False

    # QA — Wikipedia directly, skip model
    if task in DIRECT_WIKI_TASKS:
        wiki = search_wikipedia(user_input, max_chars=600)
        if wiki:
            return format_wiki_answer(wiki), True, True
        # Fallback to model if wiki fails
        cfg = prompts[task]
        return cfg["prefix"] + user_input + cfg["response"], False, False

    # Farmer — hybrid: wiki context + model advice
    if task in HYBRID_TASKS:
        wiki   = search_wikipedia(user_input, max_chars=400)
        prompt = build_farmer_prompt(wiki, user_input, prompts)
        return prompt, wiki is not None, False

    # Default
    cfg = prompts[task]
    return cfg["prefix"] + user_input + cfg["response"], False, False