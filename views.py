# myapp/views.py
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from urllib.parse import urlparse
from .models import Factcheck, UserReport
from django.views.decorators.csrf import csrf_exempt
import os
import requests
import time
import csv
import re
from dotenv import load_dotenv

# LLM / search
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage

# Translators (classical lib you already use)
from translate import Translator

# Voice (kept imported if used elsewhere)
import speech_recognition as sr
import pyttsx3

# Dates
from htmldate import find_date

# Authenticity checker (your module)
from .authenticity_checker import check_news_authenticity, FactCheck, fact_check_statement

# Sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# DRF
from rest_framework import permissions, viewsets, status
from .serializers import FactcheckSerializer
from rest_framework.views import APIView
from rest_framework.response import Response

# (Legacy imports you had for agents; they remain unused in this snippet but left intact if needed elsewhere)
from tavily import TavilyClient
from langchain_community.tools import TavilySearchResults as LC_TavilySearchResults_A
from retry import retry
from requests.exceptions import ConnectionError
from langchain_community.tools.tavily_search import TavilySearchResults as LC_TavilySearchResults_B
from langchain_community.chat_models import ChatOpenAI as LC_ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage as LC_HumanMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

# Environment-based configuration is loaded before API clients are created.

# -----------------
# Translators
# -----------------
translator          = Translator(to_lang="ha")
french_translator   = Translator(to_lang="fr")
igbo_translator     = Translator(to_lang="ig")
yoruba_translator   = Translator(to_lang="yo")
swahili_translator  = Translator(to_lang="sw")
arabic_translator   = Translator(to_lang="ar")

# ---------------
# Sentiment
# ---------------
analyzer = SentimentIntensityAnalyzer()
SENTIMENT_THRESH_POS = 0.05
SENTIMENT_THRESH_NEG = -0.05

def analyze_sentiment_text(text: str) -> dict:
    """
    English sentiment analysis via VADER.
    """
    scores = analyzer.polarity_scores(text or "")
    comp = scores.get("compound", 0.0)
    if comp > SENTIMENT_THRESH_POS:
        label = "Positive"
    elif comp < SENTIMENT_THRESH_NEG:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "scores": scores}

# ===== Simple one-shot LLM translation (no chunking) =====
from langchain_openai import ChatOpenAI as _COAI
from langchain.schema import SystemMessage as _SysMsg, HumanMessage as _HumMsg

_llm_simple = _COAI(model="gpt-4o-mini", temperature=0)

def llm_translate(text: str, target_lang: str) -> str:
    """Translate full text in a single LLM call."""
    if not text:
        return text
    try:
        sys_msg = _SysMsg(content=(
            "You are a professional translator. Translate the user text faithfully into the target language. "
            "Do not add, remove, or embellish meaning. Keep names/URLs as-is. "
            "If the input is already in the target language, return it unchanged. "
            "Return only the translation—no explanations."
        ))
        usr_msg = _HumMsg(content=f"Target language: {target_lang}\n\nText:\n{text}")
        out = _llm_simple.invoke([sys_msg, usr_msg])
        return out.content.strip()
    except Exception as e:
        # Fallback to classical translator for Hausa (as you had)
        try:
            return translator.translate(text) if target_lang.lower().startswith("hausa") else text
        except Exception:
            print(f"[LLM translate error] {e}")
            return text

def analyze_sentiment_multilingual(text: str, source_lang_name: str = "auto") -> dict:
    """
    Translate to English first (for Hausa/Yoruba/Igbo/Swahili/French/Arabic) then analyze with VADER.
    """
    try:
        english = llm_translate(text or "", target_lang="English")
        return analyze_sentiment_text(english)
    except Exception:
        # fallback on raw text (VADER may still somewhat work)
        return analyze_sentiment_text(text or "")

# -----------------
# Chunking / translation helpers you already had
# -----------------
def chunk_text(text, max_chars=400):
    sentences = re.split(r'(?<=[.!?])\s+', text or "")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def translate_large_text(text, translator_obj, max_chars=4500):
    try:
        chunks = chunk_text(text, max_chars)
        translated_chunks = [translator_obj.translate(chunk) for chunk in chunks]
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text

def translate_to_yoruba(text):  return translate_large_text(text, yoruba_translator)
def translate_to_arabic(text):  return translate_large_text(text, arabic_translator)
def translate_to_swahili(text): return translate_large_text(text, swahili_translator)
def translate_to_igbo(text):    return translate_large_text(text, igbo_translator)
def translate_to_french(text):  return translate_large_text(text, french_translator)
def translate_text(text, translator_obj): return translate_large_text(text, translator_obj)

# -----------------
# Basic pages
# -----------------
def index(request):      return render(request, 'index.html')
def preview(request):    return render(request, "preview.html")
def about(request):      return render(request, 'about.html')
def fact(request):       return render(request, 'factcheck.html')

# Hausa/Yoruba/Swahili/Igbo/French/Arabic pages
def hausa(request):               return render(request, 'hausa_index.html')
def combine(request):             return render(request, 'combine.html')
def hausa_factcheck(request):     return render(request, 'hausa_factcheck.html')
def habout(request):              return render(request, 'hausa_about.html')

def yoruba(request):              return render(request, 'yoruba_index.html')
def yfactcheck(request):          return render(request, 'yoruba_factcheck.html')
def yabout(request):              return render(request, 'yoruba_about.html')

def swahili(request):             return render(request, 'swahili_index.html')
def sfactcheck(request):          return render(request, 'swahili_factcheck.html')
def sabout(request):              return render(request, 'swahili_about.html')

def igbo(request):                return render(request, 'igbo_index.html')
def ifactcheck(request):          return render(request, 'igbo_factcheck.html')
def iabout(request):              return render(request, 'igbo_about.html')

def french(request):              return render(request, 'french_index.html')
def ffactcheck(request):          return render(request, 'french_factcheck.html')
def fabout(request):              return render(request, 'french_about.html')

def arabic(request):              return render(request, 'arabic_index.html')
def arabout(request):             return render(request, 'arabic_about.html')
def arafactcheck(request):        return render(request, 'arabic_factcheck.html')
def araresult(request):           return render(request, 'arabic_preview.html')

def speech_input(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        return JsonResponse({'result': speech_input_result})
    return JsonResponse({'error': 'Invalid request method'})

# -----------------
# Search + LLM summarizer (your existing pattern)
# -----------------
def get_external_api_answer(query):
    # Deterministic for fact checks
    llm_groq   = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    serper_tool = GoogleSerperAPIWrapper()
    tavily_tool = TavilySearch()

    try:
        serper_data = serper_tool.results(query)
    except Exception as e:
        serper_data = {"error": str(e)}

    try:
        tavily_data = tavily_tool.invoke({"query": query})
    except Exception as e:
        tavily_data = {"error": str(e)}

    # Extract URLs
    sources = []
    if "organic" in serper_data:
        sources += [item.get("link") for item in serper_data["organic"] if "link" in item]
    if "results" in tavily_data:
        sources += [item.get("url") for item in tavily_data["results"] if "url" in item]

    # Combine search content (text only)
    serper_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('snippet', '')}"
        for item in serper_data.get("organic", [])
    )
    tavily_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('content', '')}"
        for item in tavily_data.get("results", [])
    )
    combined_context = f"{serper_snippets}\n\n{tavily_snippets}".strip()

    messages = [
        SystemMessage(content=(
            "You are a professional fact-checking assistant. "
            "Verify the user's claim using ONLY the provided information below. "
            "Respond clearly, concisely, in a formal narrative. "
            "Do not mention sources or websites. Avoid bullet points. "
            "State if the claim is true, false, misleading, or unverifiable, and explain why. Do not speculate."
        )),
        HumanMessage(content=f"{combined_context}\n\nUser Claim: {query}")
    ]

    try:
        response = llm_groq.invoke(messages)
    except Exception as e:
        print("[ChatGroq Failed] Switching to OpenAI:", e)
        response = llm_openai.invoke(messages)

    summary = (response.content or "").strip()
    return summary, sources

# -----------------
# Views with SENTIMENT integrated
# -----------------
def result(request):
    if request.method == 'POST':
        start_time = time.time()
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query          = speech_input_result if speech_input_result else request.POST.get('query', '')

        cleaned_summary, sources = get_external_api_answer(user_query)

        # ✅ Sentiment on English query
        sentiment = analyze_sentiment_text(user_query)
        sentiment_label = sentiment["label"]

        # Save
        Factcheck.objects.create(
            user_input_news=user_query,
            fresult=cleaned_summary,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'preview.html', {
            'user_input_news': user_query,
            'tavily_answer': cleaned_summary,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,  # optional display
        })
    return render(request, 'factcheck.html')

# Hausa
def hausa_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_hausa    = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_hausa, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_hausa = llm_translate(cleaned_summary_en, target_lang="Hausa")

        # ✅ Sentiment (translate→analyze)
        sentiment = analyze_sentiment_multilingual(user_query_hausa, source_lang_name="Hausa")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_hausa,
            fresult=cleaned_summary_hausa,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'hausa_preview.html', {
            'user_input_news': user_query_hausa,
            'tavily_answer': cleaned_summary_hausa,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'hausa_factcheck.html')

def result_detail_hausa(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'hausa_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# Yoruba
def yoruba_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_yoruba   = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_yoruba, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_yoruba = llm_translate(cleaned_summary_en, target_lang="Yoruba")

        # ✅ Sentiment
        sentiment = analyze_sentiment_multilingual(user_query_yoruba, source_lang_name="Yoruba")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_yoruba,
            fresult=cleaned_summary_yoruba,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'yoruba_preview.html', {
            'user_input_news': user_query_yoruba,
            'tavily_answer': cleaned_summary_yoruba,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'yoruba_factcheck.html')

def result_detail_yoruba(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'yoruba_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# Igbo
def igbo_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_igbo     = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_igbo, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_igbo = llm_translate(cleaned_summary_en, target_lang="Igbo")

        # ✅ Sentiment
        sentiment = analyze_sentiment_multilingual(user_query_igbo, source_lang_name="Igbo")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_igbo,
            fresult=cleaned_summary_igbo,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'igbo_preview.html', {
            'user_input_news': user_query_igbo,
            'tavily_answer': cleaned_summary_igbo,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'igbo_factcheck.html')

def result_detail_igbo(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'igbo_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# Swahili
def swahili_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_swahili  = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_swahili, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_swahili = llm_translate(cleaned_summary_en, target_lang="Swahili")

        # ✅ Sentiment
        sentiment = analyze_sentiment_multilingual(user_query_swahili, source_lang_name="Swahili")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_swahili,
            fresult=cleaned_summary_swahili,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'swahili_preview.html', {
            'user_input_news': user_query_swahili,
            'tavily_answer': cleaned_summary_swahili,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'swahili_factcheck.html')

def result_detail_swahili(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'swahili_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# French
def french_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_french   = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_french, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_french = llm_translate(cleaned_summary_en, target_lang="French")

        # ✅ Sentiment
        sentiment = analyze_sentiment_multilingual(user_query_french, source_lang_name="French")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_french,
            fresult=cleaned_summary_french,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'french_preview.html', {
            'user_input_news': user_query_french,
            'tavily_answer': cleaned_summary_french,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'french_factcheck.html')

def result_detail_french(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'french_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# Arabic
def arabic_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value  = request.POST.get('hiddenInput', '')
        user_query_arabic   = speech_input_result if speech_input_result else request.POST.get('query', '')

        query_en = llm_translate(user_query_arabic, target_lang="English")
        cleaned_summary_en, sources = get_external_api_answer(query_en)
        cleaned_summary_arabic = llm_translate(cleaned_summary_en, target_lang="Arabic")

        # ✅ Sentiment
        sentiment = analyze_sentiment_multilingual(user_query_arabic, source_lang_name="Arabic")
        sentiment_label = sentiment["label"]

        Factcheck.objects.create(
            user_input_news=user_query_arabic,
            fresult=cleaned_summary_arabic,
            genuine_urls=sources,
            sentiment_label=sentiment_label,
        )

        return render(request, 'arabic_preview.html', {
            'user_input_news': user_query_arabic,
            'tavily_answer': cleaned_summary_arabic,
            'sources': sources,
            'hidden_input_value': hidden_input_value,
            'sentiment': sentiment,
        })
    return render(request, 'arabic_factcheck.html')

def result_detail_arabic(request, slug):
    factcheck = Factcheck.objects.get(slug=slug)
    return render(request, 'arabic_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })

# -----------------
# Lists / CSV export
# -----------------
def all_factchecks(request):
    factchecks = Factcheck.objects.all()
    user_reports = UserReport.objects.all()
    return render(request, 'all_factchecks.html', {
        'factchecks': factchecks,
        'user_reports': user_reports
    })

def export_factchecks_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="factchecks.csv"'
    writer = csv.writer(response)
    writer.writerow([
        'ID', 'User Input News', 'Fact Check Result', 'Sentiment Label',
        'Genuine URLs', 'Non-Authentic URLs', 'Number of Genuine Sources',
        'Genuine URLs and Dates', 'Non-Authentic Sources'
    ])
    factchecks = Factcheck.objects.all()
    for factcheck in factchecks:
        writer.writerow([
            factcheck.id,
            factcheck.user_input_news,
            factcheck.fresult,
            getattr(factcheck, "sentiment_label", ""),
            factcheck.genuine_urls,
            factcheck.non_authentic_urls,
            factcheck.num_genuine_sources,
            factcheck.genuine_urls_and_dates,
            factcheck.non_authentic_sources
        ])
    return response

def reliable_sources(request, user_query):
    authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources = check_news_authenticity(user_query)

    # Extract keywords for each URL & sentiments
    url_keywords_mapping = {}
    keyword_sentiments = {}
    for url in genuine_urls:
        parsed_url = urlparse(url)
        path = parsed_url.path
        keywords = path.split('/')
        non_empty_keywords = [keyword.replace("-", " ") for keyword in keywords if keyword]
        if non_empty_keywords:
            keyword = non_empty_keywords[-1]
            url_keywords_mapping[url] = keyword
            sentiment_scores = analyzer.polarity_scores(keyword)
            keyword_sentiments[keyword] = sentiment_scores
        else:
            url_keywords_mapping[url] = ""

    # Overall user input sentiment (English assumed here)
    sentiment_scores = analyzer.polarity_scores(user_query)
    sentiment_label = "Neutral"
    if sentiment_scores['compound'] > SENTIMENT_THRESH_POS:
        sentiment_label = "Positive"
    elif sentiment_scores['compound'] < SENTIMENT_THRESH_NEG:
        sentiment_label = "Negative"

    return render(request, 'reliable.html', {
        'genuine_sources': genuine_sources,
        'result': authenticity_result,
        'genuine_urls': genuine_urls,
        'num_genuine_sources': num_genuine_sources,
        'non_authentic_urls': non_authentic_urls,
        'n': non_authentic_sources,
        'user_input_news': user_query,
        'url_keywords_mapping': url_keywords_mapping,
        'keyword_sentiments': keyword_sentiments,
        'sentiment_label': sentiment_label
    })

def internet_safety(request):
    return render(request, 'internet_safety.html')

def fact1(request):
    events = Factcheck.objects.all()
    context = {'events': events}
    return render(request, 'fact.html', context)

# -----------------
# Optional: Agent-based Tavily answer (kept from your code)
# -----------------
def get_tavily_answer_with_retry(query):
    llm = LC_ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    search = TavilySearchAPIWrapper()
    tavily_tool = LC_TavilySearchResults_B(api_wrapper=search)

    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    output = agent_chain.run(query)
    return output

class FactcheckAPI(APIView):
    def post(self, request):
        user_query = request.data.get('user_input_news', '').strip()

        # Tavily agent answer
        tavily_answer = get_tavily_answer_with_retry(user_query)

        # Authenticity
        (authenticity_result,
         genuine_sources,
         num_genuine_sources,
         genuine_urls,
         non_authentic_urls,
         non_authentic_sources,
         non_legit_length) = check_news_authenticity(user_query)

        # Dates
        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url, timeout=10).content.decode('utf-8', errors='ignore')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                genuine_urls_and_dates[url] = None

        # ✅ Standardized sentiment (English request body)
        sentiment = analyze_sentiment_text(user_query)
        sentiment_label = sentiment["label"]
        sentiment_scores = sentiment["scores"]

        factcheck = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources=num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources=non_legit_length
        )

        data = FactcheckSerializer(factcheck).data
        # Enrich API response with scores (not persisted unless you add a JSONField)
        data["sentiment"] = {"label": sentiment_label, "scores": sentiment_scores}
        return Response(data, status=status.HTTP_201_CREATED)





# ============================================================
#  🔎 IMAGE FACTCHECK / REVERSE IMAGE SEARCH API (views.py)
# ============================================================
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from urllib.parse import urlparse
from datetime import datetime
import time
import os
import re
import requests
import base64

# -----------------------------------------------------------------------------
# 🔑 Keys: use SERPAPI_API_KEY if set; otherwise fall back to SERPER_API_KEY
# -----------------------------------------------------------------------------
SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPER_API_KEY")
OPENAI_KEY  = os.environ.get("OPENAI_API_KEY")
IMGBB_KEY   = os.environ.get("IMGBB_API_KEY")  # optional, only used by UploadImgbbAPI

# -----------------------------------------------------------------------------
# 🗓️ Date helpers (best-effort parsing of "Sep 30, 2016" / "Nov 2022" / "N/A")
# -----------------------------------------------------------------------------
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1
)}

def _parse_date_str(s: str):
    """
    Try to turn human date strings like 'Sep 30, 2016' or 'Nov 2022' into sortable tuples.
    Returns (YYYY, MM, DD) or None if unknown.
    """
    if not s or s == "N/A":
        return None
    s = s.strip()

    # Patterns:
    # 1) 'Sep 30, 2016'
    m = re.match(r"^(?P<mon>[A-Za-z]{3,})\s+(?P<day>\d{1,2}),\s*(?P<yr>\d{4})$", s)
    if m:
        mon = _MONTHS.get(m.group("mon").lower())
        if mon:
            return (int(m.group("yr")), mon, int(m.group("day")))

    # 2) 'Nov 2022'
    m = re.match(r"^(?P<mon>[A-Za-z]{3,})\s+(?P<yr>\d{4})$", s)
    if m:
        mon = _MONTHS.get(m.group("mon").lower())
        if mon:
            return (int(m.group("yr")), mon, 1)

    # 3) '2022-11-05' (ISO)
    m = re.match(r"^(?P<yr>\d{4})-(?P<mon>\d{2})-(?P<day>\d{2})$", s)
    if m:
        return (int(m.group("yr")), int(m.group("mon")), int(m.group("day")))

    # 4) Just a year
    m = re.match(r"^(?P<yr>\d{4})$", s)
    if m:
        return (int(m.group("yr")), 1, 1)

    return None

# -----------------------------------------------------------------------------
# 🔎 SerpAPI call with retry (works with google_reverse_image or google_lens)
# -----------------------------------------------------------------------------
def reverse_image_serpapi_with_retry(image_url: str,
                                     engine: str = "google_reverse_image",
                                     retries: int = 2,
                                     delay: float = 0.8):
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI key missing (set SERPAPI_API_KEY or SERPER_API_KEY)")

    params = {
        "engine": engine,
        "image_url": image_url,
        "api_key": SERPAPI_KEY,
        "no_cache": True,
        "hl": "en",
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            return GoogleSearch(params).get_dict()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
            else:
                raise last_err

# -----------------------------------------------------------------------------
# 🧹 Normalize SerpAPI image_results
# -----------------------------------------------------------------------------
def extract_and_format_data(results):
    """
    Input: SerpAPI 'image_results' list
    Output:
      cleaned: list of dicts {title, link, snippet, source, domain, date}
      platforms: set()
      domains: set()
      dates: list of strings (as provided by SerpAPI)
    """
    platforms, domains, dates, cleaned = set(), set(), [], []

    for res in results:
        source  = res.get("source", "Unknown")
        link    = res.get("link", "")
        title   = res.get("title", "No Title")
        snippet = res.get("snippet", "No snippet available")
        date    = res.get("date", "N/A")

        parsed = urlparse(link)
        domain = parsed.netloc.replace("www.", "") if parsed.netloc else "unknown"

        if source:
            platforms.add(source)
        if domain:
            domains.add(domain)
        if date and date != "N/A":
            dates.append(date)

        cleaned.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "source": source,
            "domain": domain,
            "date": date,
        })

    return cleaned, platforms, domains, dates

# -----------------------------------------------------------------------------
# 🕵️ Earliest appearance helper
# -----------------------------------------------------------------------------
def find_earliest_appearance(cleaned):
    """
    From cleaned hits, pick the earliest item with a parseable date.
    Returns dict or None.
    """
    dated = []
    for item in cleaned:
        t = _parse_date_str(item.get("date"))
        if t:
            dated.append((t, item))
    if not dated:
        return None
    dated.sort(key=lambda x: x[0])  # earliest first
    _, item = dated[0]
    return {
        "date": item.get("date"),
        "platform": item.get("source"),
        "domain": item.get("domain"),
        "title": item.get("title"),
        "link": item.get("link"),
    }

# -----------------------------------------------------------------------------
# 🤖 AI Summary
# -----------------------------------------------------------------------------
def generate_ai_summary(image_url, data, platforms, domains, dates):
    """
    Uses your existing OpenAI setup (env key) to produce a concise, structured analysis.
    """
    if not OPENAI_KEY:
        return "AI analysis unavailable: OPENAI_API_KEY missing on server."

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0)

    bullets = "\n".join([
        f"- Title: {item['title']}\n"
        f"  Link: {item['link']}\n"
        f"  Source: {item['source']}\n"
        f"  Domain: {item['domain']}\n"
        f"  Date: {item['date']}\n"
        f"  Snippet: {item['snippet']}\n"
        for item in data
    ])

    prompt = f"""
You are a careful reverse-image fact-checking analyst. ONLY use the evidence below. If unknown, say "Unknown".
Do not fabricate dates or sources. Be brief, precise, and neutral.

IMAGE_URL: {image_url}

EVIDENCE (top hits):
{bullets}

Return JSON with these fields:
- summary: 2–4 sentences describing what the image likely represents (evidence-only).
- first_appearance: best-effort earliest (platform, domain, date, title, link) if present; else null.
- platforms: array of unique platform names (strings).
- context_notes: short bullet-like phrases on how/why it appears (news, bio, social, meme, etc).
- credibility: one of ["authentic","reused","manipulated","unknown"], based only on evidence.
"""

    messages = [
        SystemMessage(content="You are an evidence-bound reverse-image fact-checking assistant."),
        HumanMessage(content=prompt)
    ]

    out = llm.invoke(messages)
    return (out.content or "").strip()

# -----------------------------------------------------------------------------
# 📤 Server-side Imgbb upload (optional but recommended for the frontend)
# -----------------------------------------------------------------------------
@method_decorator(csrf_exempt, name="dispatch")
class UploadImgbbAPI(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        if not IMGBB_KEY:
            return Response({"error": "IMGBB_API_KEY missing on server"}, status=500)

        # Accept either multipart file or JSON base64
        if "image" in request.FILES:
            file_obj = request.FILES["image"]
            b64 = base64.b64encode(file_obj.read()).decode("utf-8")
        else:
            try:
                image_b64 = (request.data.get("image_base64") or "").strip()
                if not image_b64:
                    return Response({"error": "Provide multipart 'image' or 'image_base64' in JSON"},
                                    status=400)
                # strip data URL prefix if present
                if image_b64.startswith("data:"):
                    image_b64 = image_b64.split(",", 1)[-1]
                b64 = image_b64
            except Exception:
                return Response({"error": "Invalid request body"}, status=400)

        try:
            r = requests.post(
                "https://api.imgbb.com/1/upload",
                params={"key": IMGBB_KEY},
                data={"image": b64},
                timeout=30
            )
            data = r.json()
            if not r.ok or not data.get("success"):
                return Response({"error": data.get("error", {}).get("message", "Upload failed")},
                                status=502)
            return Response({
                "image_url": data["data"]["url"],
                "display_url": data["data"]["display_url"]
            }, status=200)
        except Exception as e:
            return Response({"error": f"Imgbb upload failed: {str(e)}"}, status=502)

# -----------------------------------------------------------------------------
# 🚀 Reverse Image Factcheck API
# -----------------------------------------------------------------------------
@method_decorator(csrf_exempt, name="dispatch")
class ImageFactcheckAPI(APIView):
    """
    POST /api/image-factcheck
    {
      "image_url": "<direct image URL>",
      "max_results": 15,
      "debug": false
    }
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        image_url   = (request.data.get("image_url") or "").strip()
        max_results = int(request.data.get("max_results") or 15)
        debug       = bool(request.data.get("debug") or False)

        if not image_url:
            return Response({"error": "image_url is required"}, status=400)

        # Basic sanity: discourage imgbb page URLs instead of the direct image
        try:
            parsed = urlparse(image_url)
            if parsed.netloc.endswith("imgbb.com") and not re.search(r"/\.(jpg|jpeg|png|gif|webp)(\?|$)", image_url, re.I):
                return Response({"error": "Please provide a DIRECT image URL, not the Imgbb page URL"}, status=400)
        except Exception:
            pass

        if not SERPAPI_KEY:
            return Response({"error": "SERPAPI_API_KEY (or SERPER_API_KEY alias) missing"}, status=500)

        try:
            # 1) Try Google Reverse Image
            data1 = reverse_image_serpapi_with_retry(image_url, engine="google_reverse_image")
            if isinstance(data1, dict) and "error" in data1:
                return Response({"error": f"SerpAPI error: {data1['error']}"}, status=502)

            results = (data1 or {}).get("image_results") or []

            # 2) Fallback to Google Lens
            if not results:
                data2 = reverse_image_serpapi_with_retry(image_url, engine="google_lens")
                if isinstance(data2, dict) and "error" in data2:
                    return Response({"error": f"SerpAPI error: {data2['error']}"}, status=502)
                results = (data2 or {}).get("image_results") or []

            if not results:
                payload = {"error": "No reverse image results found."}
                if debug:
                    payload["serpapi_debug"] = data1
                return Response(payload, status=404)

            # 3) Normalize / cut to max_results
            formatted_data, platforms, domains, dates = extract_and_format_data(results[:max_results])

            # 4) First appearance (best effort)
            earliest_appearance = find_earliest_appearance(formatted_data)

            # 5) Overview
            # Convert dates to parsed tuples to compute min/max; fall back to string min/max
            parsed_dates = [(d, _parse_date_str(d)) for d in dates if _parse_date_str(d)]
            if parsed_dates:
                earliest = min(parsed_dates, key=lambda x: x[1])[0]
                latest   = max(parsed_dates, key=lambda x: x[1])[0]
            else:
                earliest = min(dates) if dates else None
                latest   = max(dates) if dates else None

            overview = {
                "total_results": len(formatted_data),
                "unique_platforms": sorted(list(platforms)),
                "unique_domains": sorted(list(domains)),
                "date_range": [earliest, latest],
            }

            # 6) AI analysis (string or JSON text)
            ai_summary = generate_ai_summary(image_url, formatted_data, platforms, domains, dates)

            response_data = {
                "image_url": image_url,
                "overview": overview,
                "earliest_appearance": earliest_appearance,
                "ai_analysis": ai_summary,   # may be JSON string if model follows instruction
                "top_hits": formatted_data,
            }
            if debug:
                response_data["serpapi_raw_counts"] = {
                    "initial_results": len((data1 or {}).get("image_results") or []),
                    "final_results": len(results),
                }

            return Response(response_data, status=200)

        except Exception as e:
            # Surface useful message; avoid leaking stack traces
            return Response({"error": str(e)}, status=500)








