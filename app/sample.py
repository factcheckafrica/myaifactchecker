# views_factcheck_api.py
import re
import time
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# LLM & search (English only)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from langchain.schema import SystemMessage, HumanMessage

from .serializers import FactcheckRequestSerializer, FactcheckResponseSerializer


# ---------------------------
# Helpers & configuration
# ---------------------------

# Greeting detector
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|what'?s up|howdy)\b",
    re.IGNORECASE,
)

def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text or ""))

# URL detection
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Normalized verdict set
_VERDICT_WORDS = ("True", "False", "Misleading", "Unverifiable")


def _split_title_url(s: str):
    """Input item is 'Title - URL' or just 'URL'."""
    if " - " in s:
        title, url = s.split(" - ", 1)
    else:
        title, url = "Source", s
    return title.strip(), url.strip()


def _score_source_credibility(url: str) -> int:
    """
    Score source credibility (1-5) based on domain.
    5 = highly credible, 1 = use with caution
    """
    url_lower = url.lower()
    
    # Tier 1: Official/Primary sources (5)
    tier1_domains = [
        '.gov.ng', 'inec.gov.ng', 'cbn.gov.ng', 'nigeriaembassyusa.org',
        'statehouse.gov.ng', 'ng.undp.org', 'nigeria.gov.ng'
    ]
    if any(d in url_lower for d in tier1_domains):
        return 5
    
    # Tier 2: Established international outlets (4)
    tier2_domains = [
        'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com', 'aljazeera.com',
        'theguardian.com', 'cnn.com', 'bloomberg.com', 'nytimes.com'
    ]
    if any(d in url_lower for d in tier2_domains):
        return 4
    
    # Tier 3: Established Nigerian outlets (4)
    tier3_domains = [
        'channelstv.com', 'premiumtimesng.com', 'thecable.ng',
        'punchng.com', 'vanguardngr.com', 'thenationonlineng.net',
        'dailytrust.com', 'saharareporters.com', 'businessday.ng',
        'thisdaylive.com', 'tribuneonlineng.com'
    ]
    if any(d in url_lower for d in tier3_domains):
        return 4
    
    # Tier 4: General news sites (3)
    if any(ext in url_lower for ext in ['.com', '.org', '.ng', '.net']):
        return 3
    
    # Tier 5: Social media, blogs, unknown (2)
    low_cred = ['facebook.com', 'twitter.com', 'x.com', 'instagram.com', 
                'tiktok.com', 'blog', 'wordpress.com', 'medium.com']
    if any(d in url_lower for d in low_cred):
        return 2
    
    return 2  # Default: use with caution


def _format_sources_links(sources: list, max_n: int = 3) -> str:
    """Return a compact Sources block with clickable links, prioritized by credibility"""
    if not sources:
        return ""
    
    # Score and sort sources by credibility
    scored_sources = []
    for s in sources:
        title, url = _split_title_url(s)
        score = _score_source_credibility(url)
        scored_sources.append((score, s, title, url))
    
    # Sort by credibility score (descending), then take top N
    scored_sources.sort(key=lambda x: x[0], reverse=True)
    
    items = []
    for score, _, title, url in scored_sources[:max_n]:
        items.append(f"- {title} — {url}")
    
    return "Sources:\n" + "\n".join(items) if items else ""


def _append_or_fix_sources_block(result_text: str, sources: list) -> str:
    """
    Ensure the final text includes a Sources block WITH direct links.
    If a Sources block exists but has no links, replace it with a linked block.
    """
    if not sources:
        return result_text

    formatted = _format_sources_links(sources, max_n=3)
    if not formatted:
        return result_text

    # Already has a Sources section?
    if "Sources:" in result_text:
        tail = result_text.split("Sources:", 1)[1]
        if not _URL_RE.search(tail):
            # Replace existing Sources section (everything after 'Sources:') with our linked block
            return re.sub(r"Sources:\s*(.|\n)*$", formatted, result_text, flags=re.IGNORECASE)
        return result_text  # it already has URLs
    # Otherwise append ours
    return (result_text.rstrip() + "\n\n" + formatted).strip()


def _ensure_structure(text: str, fallback_sources: list) -> str:
    """
    Enforce consistent structure and validate verdict quality.
    'Verdict: <...>\n\nExplanation: ...\n\n[Sources: ...]'
    """
    t = (text or "").strip()

    # 1) Ensure valid Verdict with normalization
    verdict_match = re.match(r"^\s*Verdict:\s*(\w+)", t, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).capitalize()
        # Normalize to standard verdicts
        if verdict not in _VERDICT_WORDS:
            # Map common variations
            verdict_map = {
                "Partly": "Misleading",
                "Partial": "Misleading",
                "Partially": "Misleading",
                "Mixed": "Misleading",
                "Uncertain": "Unverifiable",
                "Unknown": "Unverifiable",
                "Unclear": "Unverifiable",
                "Unconfirmed": "Unverifiable",
                "Correct": "True",
                "Accurate": "True",
                "Incorrect": "False",
                "Wrong": "False",
                "Fake": "False",
            }
            verdict = verdict_map.get(verdict, "Unverifiable")
        t = re.sub(r"^\s*Verdict:\s*\w+", f"Verdict: {verdict}", t, flags=re.IGNORECASE)
    else:
        # No verdict found - infer from content
        first_line = (t.splitlines()[0] if t else "").lower()
        inferred = "Unverifiable"
        for v in _VERDICT_WORDS:
            if v.lower() in first_line:
                inferred = v
                break
        t = f"Verdict: {inferred}\n\n{t}"

    # 2) Ensure Explanation section exists
    if "Explanation:" not in t:
        lines = t.splitlines()
        verdict_line = lines[0] if lines else "Verdict: Unverifiable"
        body = "\n".join(lines[1:]).strip() or "Insufficient reliable evidence was found to verify this claim."
        t = f"{verdict_line}\n\nExplanation: {body}"

    # 3) Validate explanation quality
    expl_match = re.search(r"Explanation:\s*(.+?)(?=\n\nSources:|$)", t, re.DOTALL)
    if expl_match:
        explanation = expl_match.group(1).strip()
        
        # Check for lazy patterns and clean them
        lazy_replacements = {
            "based on the sources": "the evidence shows",
            "according to the context": "the available information indicates",
            "the information provided suggests": "the evidence suggests",
            "from the search results": "available information shows",
        }
        
        explanation_lower = explanation.lower()
        for lazy_phrase, replacement in lazy_replacements.items():
            if lazy_phrase in explanation_lower:
                # Find and replace preserving case context
                pattern = re.compile(re.escape(lazy_phrase), re.IGNORECASE)
                explanation = pattern.sub(replacement, explanation)
        
        # Ensure minimum substance (at least 20 words for quality)
        word_count = len(explanation.split())
        if word_count < 20:
            verdict_line = t.splitlines()[0] if t else "Verdict: Unverifiable"
            if "Unverifiable" in verdict_line:
                explanation += " Additional sources or official statements would be needed for verification."
            elif "Misleading" in verdict_line:
                explanation += " The claim requires important context to be properly understood."
            else:
                explanation += " Further details are available in the sources listed below."
        
        # Update the text with cleaned explanation
        t = re.sub(
            r"(Explanation:\s*)(.+?)(?=\n\nSources:|$)",
            f"\\1{explanation}",
            t,
            flags=re.DOTALL
        )

    # 4) Add/repair Sources block with clickable links (prioritized by credibility)
    t = _append_or_fix_sources_block(t, fallback_sources)

    # 5) Final length check (WhatsApp-friendly, max ~1200 chars)
    if len(t) > 1200:
        # Try to trim explanation first while preserving structure
        parts = t.split("\n\nSources:", 1)
        main_text = parts[0]
        sources_block = "\n\nSources:" + parts[1] if len(parts) > 1 else ""
        
        if len(main_text) > 900:
            expl_match = re.search(r"(Verdict:.+?\n\nExplanation:\s*)(.+)", main_text, re.DOTALL)
            if expl_match:
                prefix = expl_match.group(1)
                explanation = expl_match.group(2)
                # Trim explanation to fit, keeping at least 2 sentences
                sentences = re.split(r'(?<=[.!?])\s+', explanation)
                trimmed = sentences[0]
                for s in sentences[1:]:
                    if len(prefix + trimmed + " " + s) <= 700:
                        trimmed += " " + s
                    else:
                        break
                main_text = prefix + trimmed
        
        t = (main_text + sources_block)[:1190].rstrip() + "…"

    return t.strip()


def _build_answer_prompt(combined_context: str, user_query: str, sources: list) -> list:
    """
    Enhanced conversational fact-checking prompt with rigorous verification standards.
    """
    # Build sources list with numbering and credibility indicators
    sources_with_scores = []
    for i, source in enumerate(sources):
        title, url = _split_title_url(source)
        score = _score_source_credibility(url)
        credibility = "🔵 Official" if score == 5 else "✓ Credible" if score >= 4 else ""
        sources_with_scores.append(f"[{i+1}] {source} {credibility}".strip())
    
    sources_text = "\n".join(sources_with_scores) or "N/A"

    sys = SystemMessage(content=(
        "You are a professional fact-checking assistant trained to verify claims with journalistic rigor.\n\n"
        
        "## GREETING HANDLING\n"
        "If the user message is a greeting (hi/hello/hey/good morning/afternoon/evening), "
        "reply with ONE friendly line inviting them to send a claim. Do NOT fact-check greetings.\n\n"
        
        "## FACT-CHECKING PROTOCOL\n"
        "For all other queries, follow this rigorous verification process:\n\n"
        
        "### 1. SOURCE EVALUATION\n"
        "- Prioritize PRIMARY sources (🔵): Official government sites (.gov.ng), institutional statements, original documents\n"
        "- Trust CREDIBLE outlets (✓): Established Nigerian media (Channels, Premium Times, The Cable, Punch) and "
        "international outlets (BBC, Reuters, AP)\n"
        "- Be CAUTIOUS of: Social media claims, partisan blogs, unverified user content, single-source claims\n"
        "- CHECK publication dates: Recent claims need recent sources; outdated sources may not reflect current reality\n"
        "- IDENTIFY conflicts: Note when credible sources contradict each other\n"
        "- CROSS-REFERENCE: Don't rely on a single source for important claims\n\n"
        
        "### 2. CLAIM ANALYSIS\n"
        "- Break down COMPLEX claims into verifiable components\n"
        "- Distinguish FACTS (verifiable data, events, statements) from OPINIONS (not fact-checkable)\n"
        "- Identify MISSING CONTEXT that fundamentally changes meaning\n"
        "- Check for MISLEADING framing: technically true statements presented deceptively\n"
        "- Note EXAGGERATIONS, MINIMIZATIONS, or cherry-picked statistics\n"
        "- Consider TEMPORAL factors: Is this claim about past, present, or future?\n\n"
        
        "### 3. VERDICT CRITERIA (Choose ONE - Be Precise)\n\n"
        
        "**True**: The claim is factually accurate and confirmed by multiple credible sources with strong consensus. "
        "No significant context is missing that would change the meaning.\n\n"
        
        "**False**: The claim is factually incorrect, fabricated, or directly contradicted by credible evidence. "
        "Core facts in the claim are demonstrably wrong.\n\n"
        
        "**Misleading**: The claim contains some factual elements BUT:\n"
        "  - Omits crucial context that changes the meaning significantly\n"
        "  - Cherry-picks data to support a false or skewed narrative\n"
        "  - Presents correlation as causation without justification\n"
        "  - Uses outdated information as if it's current\n"
        "  - Distorts meaning through selective editing, framing, or false implications\n"
        "  - Mixes true facts with false conclusions\n\n"
        
        "**Unverifiable**: Choose this when:\n"
        "  - Insufficient reliable evidence exists to make a determination\n"
        "  - Claim is too recent for adequate verification\n"
        "  - Multiple credible sources contradict each other without clear resolution\n"
        "  - Cannot access primary documentation or sources needed\n"
        "  - Claim is a prediction, opinion, or speculation presented as fact\n"
        "  - Evidence is indirect or circumstantial only\n\n"
        
        "### 4. OUTPUT FORMAT (MANDATORY STRUCTURE)\n\n"
        
        "Verdict: <True | False | Misleading | Unverifiable>\n\n"
        
        "Explanation: Write 2-4 clear, concise sentences (60-120 words) that:\n"
        "- State WHAT the evidence shows directly (avoid 'based on sources' or 'according to context')\n"
        "- Provide SPECIFIC details: dates, numbers, official statements, names when available\n"
        "- Explain WHY you reached this verdict with concrete reasoning\n"
        "- For 'Misleading': Clearly specify what's accurate vs. what's false/missing\n"
        "- For 'Unverifiable': Explain precisely what's lacking (too recent, conflicting data, insufficient evidence)\n"
        "- Use plain, accessible language; avoid jargon and technical terms\n"
        "- Be NEUTRAL and OBJECTIVE: no emotional language, no advocacy, no bias\n"
        "- Write in active voice with confident, declarative statements\n\n"
        
        "Sources: (Include for True/False/Misleading verdicts when sources are available)\n"
        "- List 1-3 most relevant and credible sources\n"
        "- Format each as: Title — URL\n"
        "- Strongly prefer Official (🔵) sources, then Credible (✓) outlets\n"
        "- Choose from the 'Available sources' list below\n"
        "- Each source on a new line with dash prefix\n"
        "- Only include sources that directly support your verdict\n\n"
        
        "## CRITICAL RULES\n"
        "- NEVER make claims beyond what the evidence clearly supports\n"
        "- If credible sources conflict significantly, default to 'Unverifiable' and explain the conflict\n"
        "- Do NOT add personal opinions, assumptions, or speculation\n"
        "- Do NOT use vague phrases: 'based on the sources', 'according to the context', 'the information provided'\n"
        "- ALWAYS prioritize accuracy over certainty—when in doubt, choose 'Unverifiable'\n"
        "- For Nigerian political/electoral claims, demand exceptionally strong evidence from official or primary sources\n"
        "- For statistics or numbers, verify they match the sources exactly\n"
        "- QUESTION: Does my explanation clearly answer WHY someone should believe this verdict?\n"
        "- QUESTION: Have I been specific with facts, or am I being vague?\n"
        "- If the claim involves multiple sub-claims, address the primary/most important one\n\n"
        
        "## SPECIAL CONSIDERATIONS\n"
        "- **Ongoing events**: For developing stories, acknowledge uncertainty and favor 'Unverifiable' unless strong consensus exists\n"
        "- **Old claims resurfacing**: Check if information is outdated; note if context has changed\n"
        "- **Satirical content**: If claim appears to be from satire/parody, note this (verdict depends on whether "
        "it's being shared as factual)\n"
        "- **Partial truths**: Often qualify as 'Misleading' rather than 'True'\n"
        "- **Predictions**: Cannot be fact-checked; mark as 'Unverifiable' and explain why"
    ))

    human = HumanMessage(content=f"""Context snippets:
{combined_context or 'N/A'}

Available sources:
{sources_text}

User Claim: {user_query}

Apply the fact-checking protocol rigorously. Analyze the claim carefully, evaluate the evidence, and return ONLY the specified sections in the exact order. Be thorough but concise.""")
    
    return [sys, human]


# ---------------------------
# Core fact-check function
# ---------------------------

def get_external_api_answer_english(query: str) -> str:
    """
    Enhanced fact-checking with rigorous verification and source prioritization.
    """
    # Greeting path (no search/LLM cost)
    if _is_greeting(query):
        return "Hello! 👋 Send me a claim or headline and I'll fact-check it with sources you can verify."

    # Initialize LLMs (lower temperature for more consistent, factual responses)
    llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Initialize search tools
    serper_tool = GoogleSerperAPIWrapper()
    tavily_tool = TavilySearch()

    # Fetch from Serper with error handling
    try:
        serper_data = serper_tool.results(query)
    except Exception as e:
        serper_data = {"error": str(e), "organic": []}

    # Fetch from Tavily with error handling
    try:
        tavily_data = tavily_tool.invoke({"query": query})
    except Exception as e:
        tavily_data = {"error": str(e), "results": []}

    # Build candidate sources list ("Title - URL")
    sources_raw = []
    
    # Serper top 5 (increased from 3 for better coverage)
    for item in serper_data.get("organic", [])[:5]:
        title = item.get("title", "").strip()
        url = item.get("link", "").strip()
        if title and url:
            sources_raw.append(f"{title} - {url}")
    
    # Tavily top 5
    for item in tavily_data.get("results", [])[:5]:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        if title and url:
            sources_raw.append(f"{title} - {url}")

    # De-duplicate by URL while preserving order
    seen = set()
    unique_sources = []
    for s in sources_raw:
        _, url = _split_title_url(s)
        if url not in seen:
            seen.add(url)
            unique_sources.append(s)
    
    # Sort by credibility score before limiting
    scored_sources = []
    for s in unique_sources:
        _, url = _split_title_url(s)
        score = _score_source_credibility(url)
        scored_sources.append((score, s))
    
    scored_sources.sort(key=lambda x: x[0], reverse=True)
    sources = [s for _, s in scored_sources[:6]]  # Keep top 6 credible sources

    # Collate text content for context (snippets only)
    serper_snippets = []
    for item in serper_data.get("organic", [])[:5]:
        title = item.get('title', 'Unknown Source')
        snippet = item.get('snippet', 'No snippet available')
        if snippet and len(snippet) > 20:  # Only include substantial snippets
            serper_snippets.append(f"From {title}:\n{snippet}")
    
    tavily_snippets = []
    for item in tavily_data.get("results", [])[:5]:
        title = item.get('title', 'Unknown Source')
        content = item.get('content', 'No content available')
        if content and len(content) > 20:  # Only include substantial content
            # Truncate very long content to keep context manageable
            if len(content) > 500:
                content = content[:500] + "..."
            tavily_snippets.append(f"From {title}:\n{content}")
    
    combined_context = "\n\n".join(serper_snippets + tavily_snippets).strip()

    # No context found case
    if not combined_context:
        base = (
            "Verdict: Unverifiable\n\n"
            "Explanation: No reliable information could be found to verify this claim. "
            "This may be a very recent development not yet covered by credible sources, "
            "or the claim may not have sufficient public documentation. "
            "Consider checking official government websites or established news outlets directly."
        )
        # Still try to include any sources we found, even if snippets failed
        return _append_or_fix_sources_block(base, sources)

    # Build messages with enhanced prompt
    messages = _build_answer_prompt(combined_context, query, sources)

    # Call LLM (Groq first, fallback to OpenAI)
    result = None
    try:
        llm_resp = llm_groq.invoke(messages)
        result = (llm_resp.content or "").strip()
    except Exception as groq_error:
        try:
            llm_resp = llm_openai.invoke(messages)
            result = (llm_resp.content or "").strip()
        except Exception as openai_error:
            # Both LLMs failed - return graceful error with sources
            base = (
                "Verdict: Unverifiable\n\n"
                "Explanation: Our fact-checking system is temporarily unavailable. "
                "Please try again in a moment, or verify this claim using the sources provided below. "
                "Check official sources like government websites for the most reliable information."
            )
            return _append_or_fix_sources_block(base, sources)

    # Final normalization: enforce structure, validate quality, ensure clickable links
    result = _ensure_structure(result, sources)
    
    return result


# ---------------------------
# DRF view
# ---------------------------

@method_decorator(csrf_exempt, name="dispatch")
class FactcheckAPIView(APIView):
    """
    POST /api/factcheck/
    Body: { "query": "..." }
    Response: { "query": "...", "result": "...", "latency_ms": 123 }
    """
    def post(self, request):
        s = FactcheckRequestSerializer(data=request.data)
        if not s.is_valid():
            return Response(s.errors, status=status.HTTP_400_BAD_REQUEST)

        query = (s.validated_data.get("query") or "").strip()
        if not query:
            return Response(
                {
                    "detail": "Query cannot be empty.",
                    "example": "Try sending a claim like: 'INEC postponed the election to September 1.'"
                },
                status=400,
            )

        # Optional: Add query length validation
        if len(query) > 500:
            return Response(
                {
                    "detail": "Query is too long. Please keep claims under 500 characters.",
                    "current_length": len(query)
                },
                status=400,
            )

        t0 = time.time()
        result_text = get_external_api_answer_english(query)
        latency_ms = int((time.time() - t0) * 1000)

        payload = {
            "query": query,
            "result": result_text,
            "latency_ms": latency_ms,
        }
        
        return Response(FactcheckResponseSerializer(payload).data, status=200)