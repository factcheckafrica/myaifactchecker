# myapp/views.py
from django.shortcuts import render
from django.http import HttpResponse
from urllib.parse import urlparse
from .models import Factcheck
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
import os
from translate import Translator
import requests
import speech_recognition as sr
import pyttsx3
from django.http import JsonResponse
from htmldate import find_date
from .authenticity_checker import check_news_authenticity, FactCheck, fact_check_statement
from langchain_community.utilities import GoogleSerperAPIWrapper
from tavily import TavilyClient
from langchain_community.tools import TavilySearchResults
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import csv
from django.contrib.auth.models import Group, User
from .models import UserReport
from rest_framework import permissions, viewsets
from .models import Factcheck
from retry import retry
from django.shortcuts import redirect, render
from .models import Factcheck
from requests.exceptions import ConnectionError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools.tavily_search import TavilySearchResults
from .serializers import FactcheckSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import re
import requests
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from langchain.schema import SystemMessage, HumanMessage


# Set environment variables for OpenAI and Tavily API key


analyzer = SentimentIntensityAnalyzer()

translator = Translator(to_lang="ha")
french_translator = Translator(to_lang="fr")
igbo_translator = Translator(to_lang="ig")
yoruba_translator = Translator(to_lang="yo")
swahili_translator = Translator(to_lang="sw")
arabic_translator = Translator(to_lang="ar") 

def all_factchecks(request):
    # Fetch all Factcheck objects
    factchecks = Factcheck.objects.all()

    # Optionally, fetch all UserReport objects
    user_reports = UserReport.objects.all()

    # Pass the data to the template
    return render(request, 'all_factchecks.html', {
        'factchecks': factchecks,
        'user_reports': user_reports
    })


def export_factchecks_csv(request):
    # Create the HTTP response with CSV content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="factchecks.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'ID', 'User Input News', 'Fact Check Result', 'Sentiment Label', 
        'Genuine URLs', 'Non-Authentic URLs', 'Number of Genuine Sources', 
        'Genuine URLs and Dates', 'Non-Authentic Sources'
    ])
    
    # Fetch all Factcheck objects
    factchecks = Factcheck.objects.all()
    
    for factcheck in factchecks:
        writer.writerow([
            factcheck.id,
            factcheck.user_input_news,
            factcheck.fresult,
            factcheck.sentiment_label,
            factcheck.genuine_urls,
            factcheck.non_authentic_urls,
            factcheck.num_genuine_sources,
            factcheck.genuine_urls_and_dates,
            factcheck.non_authentic_sources
        ])
    
    return response



# ✅ Chunking Utility

def chunk_text(text, max_chars=400):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
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

def translate_large_text(text, translator, max_chars=4500):
    try:
        chunks = chunk_text(text, max_chars)
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text


# Replace existing translation functions to use the chunked version

def translate_to_yoruba(text):
    return translate_large_text(text, yoruba_translator)

def translate_to_arabic(text):
    return translate_large_text(text, arabic_translator)

def translate_to_swahili(text):
    return translate_large_text(text, swahili_translator)

def translate_to_igbo(text):
    return translate_large_text(text, igbo_translator)

def translate_to_french(text):
    return translate_large_text(text, french_translator)

def translate_text(text, translator):
    return translate_large_text(text, translator)



def index(request):
    return render(request, 'index.html')

def preview(request):
    return render(request,"preview.html")

def about(request):
    return render(request, 'about.html')

def fact(request):
    return render(request, 'factcheck.html')



# Hausa
def hausa(request):
    return render(request, 'hausa_index.html')

# Hausa
def combine(request):
    return render(request, 'combine.html')
# Hausa
def hausa_factcheck(request):
    return render(request, 'hausa_factcheck.html')

def habout(request):
    return render(request, 'hausa_about.html')


def yoruba(request):
    return render(request, 'yoruba_index.html')

def yfactcheck(request):
    return render(request, 'yoruba_factcheck.html')

def yabout(request):
    return render(request, 'yoruba_about.html')



#Swahili
def swahili(request):
    return render(request, 'swahili_index.html')

def sfactcheck(request):
    return render(request, 'swahili_factcheck.html')

def sabout(request):
    return render(request, 'swahili_about.html')


def igbo(request):
    return render(request, 'igbo_index.html')

def ifactcheck(request):
    return render(request, 'igbo_factcheck.html')

def iabout(request):
    return render(request, 'igbo_about.html')




def french(request):
    return render(request, 'french_index.html')

def ffactcheck(request):
    return render(request, 'french_factcheck.html')

def fabout(request):
    return render(request, 'french_about.html')


def arabic(request):
    return render(request, 'arabic_index.html')

def arabout(request):
    return render(request, 'arabic_about.html')

def arafactcheck(request):
    return render(request, 'arabic_factcheck.html')

def araresult(request):
    return render(request, 'arabic_preview.html')



def speech_input(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        return JsonResponse({'result': speech_input_result})

    return JsonResponse({'error': 'Invalid request method'})




def get_external_api_answer(query):
    # Initialize LLMs
    llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize search tools
    serper_tool = GoogleSerperAPIWrapper()
    tavily_tool = TavilySearch()

    # Run Serper and Tavily
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

    # Combine search content (text only, no links)
    serper_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('snippet', '')}"
        for item in serper_data.get("organic", [])
    )

    tavily_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('content', '')}"
        for item in tavily_data.get("results", [])
    )

    combined_context = f"{serper_snippets}\n\n{tavily_snippets}"

    # System prompt for direct, narrative, source-free fact-checking
    messages = [
        SystemMessage(content="""You are a professional fact-checking assistant. 
Verify the accuracy of the user's claim using the provided information. Respond in a clear, concise, and formal narrative. 
Do not mention sources, websites, or search results. Avoid bullet points. Only state if the claim is true, false, misleading, or unverifiable and explain why based on the evidence. Do not speculate."""),
        HumanMessage(content=f"{combined_context}\n\nUser Claim: {query}")
    ]

    # Try ChatGroq first, fallback to OpenAI
    try:
        response = llm_groq.invoke(messages)
    except Exception as e:
        print("[ChatGroq Failed] Switching to OpenAI:", e)
        response = llm_openai.invoke(messages)

    summary = response.content.strip()
    return summary, sources



# === Main result view ===
def result(request):
    if request.method == 'POST':
        start_time = time.time()

        # Get query from user input (speech or text)
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Get result from external API and sources
        cleaned_summary, sources = get_external_api_answer(user_query)


        # Save to database (optional)
        Factcheck.objects.create(
            user_input_news=user_query,
            fresult=cleaned_summary,
            genuine_urls=sources,
        )



#            genuine_urls=genuine_urls,
        # Render template with simplified data
        return render(request, 'preview.html', {
            'user_input_news': user_query,
            'tavily_answer': cleaned_summary,
            'sources': sources,
            'hidden_input_value': hidden_input_value
        })

    return render(request, 'factcheck.html')



# ===== Simple one-shot LLM translation (no chunking) =====
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

_llm_simple = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_translate(text: str, target_lang: str) -> str:
    """Translate the full text in a single LLM call."""
    if not text:
        return text
    try:
        sys_msg = SystemMessage(content=(
            "You are a professional translator. Translate the user text faithfully into the target language. "
            "Do not add, remove, or embellish meaning. Keep names/URLs as-is. "
            "If the input is already in the target language, return it unchanged. "
            "Return only the translation—no explanations."
        ))
        usr_msg = HumanMessage(content=f"Target language: {target_lang}\n\nText:\n{text}")
        out = _llm_simple.invoke([sys_msg, usr_msg])
        return out.content.strip()
    except Exception as e:
        # Fallback to your existing classic translator if Hausa
        try:
            return translator.translate(text) if target_lang.lower().startswith("hausa") else text
        except Exception:
            print(f"[LLM translate error] {e}")
            return text


def hausa_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_hausa = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Hausa -> English (single LLM call)
        query_en = llm_translate(user_query_hausa, target_lang="English")

        # Run search + summary in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> Hausa (single LLM call)
        cleaned_summary_hausa = llm_translate(cleaned_summary_en, target_lang="Hausa")

        # Save (store original Hausa input + Hausa result)
        Factcheck.objects.create(
            user_input_news=user_query_hausa,
            fresult=cleaned_summary_hausa,
            genuine_urls=sources,
        )

        return render(request, 'hausa_preview.html', {
            'user_input_news': user_query_hausa,        # Hausa
            'tavily_answer': cleaned_summary_hausa,     # Hausa
            'sources': sources,
            'hidden_input_value': hidden_input_value
        })

    return render(request, 'hausa_factcheck.html')




def result_detail_hausa(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'hausa_preview.html', {
        'factcheck': factcheck,
        'user_input_news': factcheck.user_input_news,
        'tavily_answer': factcheck.fresult,
        'sources': factcheck.genuine_urls,
    })



def yoruba_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_yoruba = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Yoruba -> English
        query_en = llm_translate(user_query_yoruba, target_lang="English")

        # Run search + summary in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> Yoruba
        cleaned_summary_yoruba = llm_translate(cleaned_summary_en, target_lang="Yoruba")

        # Save (store original Yoruba input + Yoruba result)
        Factcheck.objects.create(
            user_input_news=user_query_yoruba,
            fresult=cleaned_summary_yoruba,
            genuine_urls=sources,
        )

        return render(request, 'yoruba_preview.html', {
            'user_input_news': user_query_yoruba,         # Yoruba
            'tavily_answer': cleaned_summary_yoruba,      # Yoruba
            'sources': sources,
            'hidden_input_value': hidden_input_value
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



def igbo_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_igbo = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Igbo -> English
        query_en = llm_translate(user_query_igbo, target_lang="English")

        # Search + summarize in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> Igbo
        cleaned_summary_igbo = llm_translate(cleaned_summary_en, target_lang="Igbo")

        # Save original Igbo + Igbo output
        Factcheck.objects.create(
            user_input_news=user_query_igbo,
            fresult=cleaned_summary_igbo,
            genuine_urls=sources,
        )

        return render(request, 'igbo_preview.html', {
            'user_input_news': user_query_igbo,
            'tavily_answer': cleaned_summary_igbo,
            'sources': sources,
            'hidden_input_value': hidden_input_value
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



def swahili_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_swahili = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Swahili -> English
        query_en = llm_translate(user_query_swahili, target_lang="English")

        # Search + summarize in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> Swahili
        cleaned_summary_swahili = llm_translate(cleaned_summary_en, target_lang="Swahili")

        Factcheck.objects.create(
            user_input_news=user_query_swahili,
            fresult=cleaned_summary_swahili,
            genuine_urls=sources,
        )

        return render(request, 'swahili_preview.html', {
            'user_input_news': user_query_swahili,
            'tavily_answer': cleaned_summary_swahili,
            'sources': sources,
            'hidden_input_value': hidden_input_value
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


def french_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_french = speech_input_result if speech_input_result else request.POST.get('query', '')

        # French -> English
        query_en = llm_translate(user_query_french, target_lang="English")

        # Search + summarize in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> French
        cleaned_summary_french = llm_translate(cleaned_summary_en, target_lang="French")

        Factcheck.objects.create(
            user_input_news=user_query_french,
            fresult=cleaned_summary_french,
            genuine_urls=sources,
        )

        return render(request, 'french_preview.html', {
            'user_input_news': user_query_french,
            'tavily_answer': cleaned_summary_french,
            'sources': sources,
            'hidden_input_value': hidden_input_value
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


def arabic_result(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        user_query_arabic = speech_input_result if speech_input_result else request.POST.get('query', '')

        # Arabic -> English
        query_en = llm_translate(user_query_arabic, target_lang="English")

        # Search + summarize in English
        cleaned_summary_en, sources = get_external_api_answer(query_en)

        # English -> Arabic
        cleaned_summary_arabic = llm_translate(cleaned_summary_en, target_lang="Arabic")

        Factcheck.objects.create(
            user_input_news=user_query_arabic,
            fresult=cleaned_summary_arabic,
            genuine_urls=sources,
        )

        return render(request, 'arabic_preview.html', {
            'user_input_news': user_query_arabic,
            'tavily_answer': cleaned_summary_arabic,
            'sources': sources,
            'hidden_input_value': hidden_input_value
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








def reliable_sources(request, user_query):
    authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources = check_news_authenticity(user_query)

    # Extract keywords for each URL
    url_keywords_mapping = {}
    keyword_sentiments = {}  # Store sentiment scores for each keyword
    for url in genuine_urls:
        parsed_url = urlparse(url)
        path = parsed_url.path
        keywords = path.split('/')

        # Filter out empty keywords and extract the last part of the path
        non_empty_keywords = [keyword.replace("-", " ") for keyword in keywords if keyword]

        if non_empty_keywords:
            # Extract the last part of the path as the keyword
            keyword = non_empty_keywords[-1]
            url_keywords_mapping[url] = keyword

            # Perform sentiment analysis on the keyword
            sentiment_scores = analyzer.polarity_scores(keyword)
            keyword_sentiments[keyword] = sentiment_scores
        else:
            # If no non-empty keywords found, set an empty string
            url_keywords_mapping[url] = ""

    # Perform sentiment analysis on the user's input
    sentiment_scores = analyzer.polarity_scores(user_query)

    # Determine sentiment label
    sentiment_label = "Neutral"
    if sentiment_scores['compound'] > 0.05:
        sentiment_label = "Positive"
    elif sentiment_scores['compound'] < -0.05:
        sentiment_label = "Negative"

    return render(request, 'reliable.html', {'genuine_sources': genuine_sources,
                                              'result': authenticity_result,
                                              'genuine_urls': genuine_urls,
                                              'num_genuine_sources': num_genuine_sources,
                                              'non_authentic_urls': non_authentic_urls,
                                              'n': non_authentic_sources,
                                              'user_input_news': user_query,
                                              'url_keywords_mapping': url_keywords_mapping,
                                              'keyword_sentiments': keyword_sentiments,
                                              'sentiment_label': sentiment_label})




def internet_safety(request):
    return render(request, 'internet_safety.html')


def fact1(request):
    events=Factcheck.objects.all()
    print(events)
    context={'events':events}
    return render(request,'fact.html',context)

# Tavily and LLM setup
def get_tavily_answer_with_retry(query):
    # Initialize Tavily search and agent
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    # Create the agent
    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Run the agent with the user query
    output = agent_chain.run(query)
    return output

class FactcheckAPI(APIView):
    def post(self, request):
        user_query = request.data.get('user_input_news', '')

        # Use Tavily to get the answer
        tavily_answer = get_tavily_answer_with_retry(user_query)

        # Check news authenticity
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources, non_legit_length = check_news_authenticity(user_query)

        # Extract dates for genuine URLs
        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None

        # Perform sentiment analysis
        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        # Save the factcheck result to the database
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

        serializer = FactcheckSerializer(factcheck)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    





# from django.shortcuts import render
# from django.http import JsonResponse
# from crewai import Agent, Task, Crew
# from crewai_tools import SerperDevTool

# # Define the intelligent assistant
# intelligent_agent = Agent(
#     role='Conversational and Fact-Checking Assistant',
#     goal="""Act as both a friendly conversational assistant and a fact-checking expert. 
#     - If the user input is a greeting or casual conversation, respond conversationally and naturally.
#     - If the user input involves a claim or request for fact-checking, analyze the claim, verify its authenticity, and provide a detailed analysis, including source links.""",
#     backstory="""You're a dual-purpose AI assistant with expertise in engaging conversations and fact-checking. 
#     Your primary goals are:
#     - Responding to greetings or casual inquiries in a friendly and informative manner.
#     - Analyzing and verifying claims to detect misinformation using credible sources, and providing clear references with source links.""",
#     verbose=True
# )

# # Initialize the semantic search tool for fact-checking
# search_tool = SerperDevTool()

# def process_user_input(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')

#         # Define the task dynamically based on user input
#         task = Task(
#             description=f"Process the user input: '{user_input}'",
#             expected_output="""Provide a response based on the type of input:
#             - For greetings or casual conversation, generate a natural, friendly response.
#             - For claims or fact-checking requests, provide:
#               - Whether the claim is True, False, or Unverified.
#               - A summary of evidence supporting the conclusion.
#               - References to credible sources, including their URLs, used for verification.""",
#             agent=intelligent_agent,
#             tools=[search_tool]
#         )

#         # Create a crew with the intelligent agent and task
#         crew = Crew(
#             agents=[intelligent_agent],
#             tasks=[task],
#             verbose=True
#         )

#         # Execute the task and get the result
#         result = crew.kickoff()

#         # Safely handle CrewOutput object
#         try:
#             # Assuming result has an attribute 'output' or similar
#             output_data = str(result)  # Convert the CrewOutput to a string if it cannot be serialized directly
#             return JsonResponse({'result': output_data})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     # Render the input page
#     return render(request, 'input_page.html')


# class FactCheckAPIView(APIView):
#     def post(self, request):
#         user_input = request.data.get('user_input', '')

#         if not user_input:
#             return Response(
#                 {"error": "User input is required."},
#                 status=status.HTTP_400_BAD_REQUEST
#             )

#         # Define the task dynamically based on user input
#         task = Task(
#             description=f"Process the user input: '{user_input}'",
#             expected_output="""Provide a response based on the type of input:
#             - For greetings or casual conversation, generate a natural, friendly response.
#             - For claims or fact-checking requests, provide:
#               - Whether the claim is True, False, or Unverified.
#               - A summary of evidence supporting the conclusion.
#               - References to credible sources, including their URLs, used for verification.""",
#             agent=intelligent_agent,
#             tools=[search_tool]
#         )

#         # Create a crew with the intelligent agent and task
#         crew = Crew(
#             agents=[intelligent_agent],
#             tasks=[task],
#             verbose=True
#         )

#         # Execute the task and handle the result
#         try:
#             result = crew.kickoff()
#             output_data = str(result)  # Convert CrewOutput to string if not serializable
#             return Response({'result': output_data}, status=status.HTTP_200_OK)
#         except Exception as e:
#             return Response(
#                 {"error": f"An error occurred while processing the request: {str(e)}"},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )



# from django.shortcuts import render
# from crewai import Agent, Task, Crew
# from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# def factcheck_webpage(request):
#     if request.method == 'POST':
#         # Get the URL from the user input
#         webpage_url = request.POST.get('webpage_url', '').strip()
        
#         if not webpage_url:
#             return render(request, 'factcheck_webpage.html', {'error': 'Please provide a valid webpage URL.'})
        
#         # Initialize the website scraping tool
#         scrape_tool = ScrapeWebsiteTool(website_url=webpage_url)

#         # Initialize the fact-checking tool
#         fact_check_tool = SerperDevTool()

#         # Define the fact-checking agent
#         fact_checker_agent = Agent(
#             role="Fact-Checking Expert",
#             goal="Extract claims from a webpage, verify them for authenticity, and detect misinformation.",
#             backstory="""You are an expert in detecting misinformation and fact-checking claims.
#             Your task is to analyze text content, extract notable claims, and verify their authenticity.""",
#             verbose=True
#         )

#         # Define the claim extraction and fact-checking task
#         extraction_task = Task(
#             description="Extract notable claims from the webpage content and verify them for authenticity.",
#             expected_output="""A report that includes:
#             - Extracted claims (if any) from the content
#             - Verification results for each claim
#             - References to credible sources for verification
#             - If no claims are found, a message stating 'No claims to fact-check'.""",
#             agent=fact_checker_agent,
#             tools=[scrape_tool, fact_check_tool]
#         )

#         # Define the crew with the agent and task
#         crew = Crew(
#             agents=[fact_checker_agent],
#             tasks=[extraction_task],
#             verbose=True
#         )

#         try:
#             # Run the task
#             result = crew.kickoff()

#             # Process and display results
#             if result:
#                 return render(request, 'factcheck_webpage.html', {'result': result})
#             else:
#                 return render(request, 'factcheck_webpage.html', {'error': 'No claims to fact-check.'})
#         except Exception as e:
#             return render(request, 'factcheck_webpage.html', {'error': f'Error processing the webpage: {str(e)}'})

#     # Render the input page
#     return render(request, 'factcheck_webpage.html')


# from django.shortcuts import render
# from django.http import JsonResponse
# from crewai import Agent, Task, Crew
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# # Define the intelligent assistant
# intelligent_agent = Agent(
#     role='Conversational and Fact-Checking Assistant',
#     goal="""Act as both a friendly conversational assistant and a fact-checking expert. 
#     - If the user input is a greeting or casual conversation, respond conversationally and naturally.
#     - If the user input involves a claim or request for fact-checking, analyze the claim, verify its authenticity, and provide a detailed analysis, including source links.""",
#     backstory="""You're a dual-purpose AI assistant with expertise in engaging conversations and fact-checking. 
#     Your primary goals are:
#     - Responding to greetings or casual inquiries in a friendly and informative manner.
#     - Analyzing and verifying claims to detect misinformation using credible sources, and providing clear references with source links.""",
#     verbose=True
# )

# # Initialize the semantic search tool for fact-checking
# search_tool = SerperDevTool()

# def combined_view(request):
#     context = {}

#     if request.method == 'POST':
#         # Handle user input form
#         if 'user_input' in request.POST:
#             user_input = request.POST.get('user_input')

#             # Define the task dynamically based on user input
#             task = Task(
#                 description=f"Process the user input: '{user_input}'",
#                 expected_output="""Provide a response based on the type of input:
#                 - For greetings or casual conversation, generate a natural, friendly response.
#                 - For claims or fact-checking requests, provide:
#                   - Whether the claim is True, False, or Unverified.
#                   - A summary of evidence supporting the conclusion.
#                   - References to credible sources, including their URLs, used for verification.""",
#                 agent=intelligent_agent,
#                 tools=[search_tool]
#             )

#             # Create a crew with the intelligent agent and task
#             crew = Crew(
#                 agents=[intelligent_agent],
#                 tasks=[task],
#                 verbose=True
#             )

#             try:
#                 # Execute the task
#                 result = crew.kickoff()
#                 context['user_input_result'] = str(result)  # Convert CrewOutput to string
#             except Exception as e:
#                 context['user_input_error'] = f"Error processing input: {str(e)}"

#         # Handle webpage URL form
#         if 'webpage_url' in request.POST:
#             webpage_url = request.POST.get('webpage_url', '').strip()

#             if not webpage_url:
#                 context['webpage_error'] = 'Please provide a valid webpage URL.'
#             else:
#                 # Initialize the website scraping tool
#                 scrape_tool = ScrapeWebsiteTool(website_url=webpage_url)

#                 # Define the fact-checking agent
#                 fact_checker_agent = Agent(
#                     role="Fact-Checking Expert",
#                     goal="Extract claims from a webpage, verify them for authenticity, and detect misinformation.",
#                     backstory="""You are an expert in detecting misinformation and fact-checking claims.
#                     Your task is to analyze text content, extract notable claims, and verify their authenticity.""",
#                     verbose=True
#                 )

#                 # Define the claim extraction and fact-checking task
#                 extraction_task = Task(
#                     description="Extract notable claims from the webpage content and verify them for authenticity.",
#                     expected_output="""A report that includes:
#                     - Extracted claims (if any) from the content
#                     - Verification results for each claim
#                     - References to credible sources for verification
#                     - If no claims are found, a message stating 'No claims to fact-check'.""",
#                     agent=fact_checker_agent,
#                     tools=[scrape_tool, search_tool]
#                 )

#                 # Create the crew with the agent and task
#                 crew = Crew(
#                     agents=[fact_checker_agent],
#                     tasks=[extraction_task],
#                     verbose=True
#                 )

#                 try:
#                     # Run the task
#                     result = crew.kickoff()
#                     context['webpage_result'] = str(result)  # Convert CrewOutput to string
#                 except Exception as e:
#                     context['webpage_error'] = f"Error processing webpage: {str(e)}"

#     return render(request, 'combined_view.html', context)

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from langchain_community.retrievers import TavilySearchAPIRetriever
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
# import os

# # Set up API keys


# # Set up the TavilySearchAPIRetriever for retrieving fact-based information
# retriever = TavilySearchAPIRetriever(k=5)  # Retrieve top 5 relevant documents

# # Custom prompt tailored for direct fact-checking with greetings handling
# prompt = ChatPromptTemplate.from_template(
#     """
#     You are a Fact-Checking AI assistant. Your primary role is to verify claims or statements and provide accurate, fact-based responses. However, you should respond warmly and engagingly to greetings.

#     Response Guidelines:
#     - If the user greets (e.g., says "hello," "hi," or similar), respond in a friendly and engaging manner.
#     - For all other inputs, check if the input is a claim or question requiring fact-checking:
#       - If yes, provide a concise and factual response based on the given context.
#       - If no, politely inform the user that you are a fact-checking assistant and can only verify claims or statements requiring verification.

#     Context: {context}
    
#     Question: {question}
    
#     Response Format:
#     - For greetings: Provide a friendly and engaging response.
#     - For fact-checking: Provide a direct, fact-based answer, avoiding unnecessary qualifiers.
#     - For unrelated queries: Politely explain your purpose and limitations.
#     """
# )

# # Define the language model with minimal configuration
# llm = ChatOpenAI(
#     model_name="gpt-4o",
#     temperature=0,  # Ensure deterministic responses
# )

# # Format retrieved documents into a readable string
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Define the processing chain
# chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# class FactCheckWithTavilyAPIView(APIView):
#     def post(self, request):
#         user_input = request.data.get('user_input', '')

#         if not user_input:
#             return Response(
#                 {"error": "User input is required."},
#                 status=status.HTTP_400_BAD_REQUEST
#             )
        
#         # Process the user input through the chain
#         try:
#             response = chain.invoke(user_input)
#             return Response({"response": response}, status=status.HTTP_200_OK)
#         except Exception as e:
#             return Response(
#                 {"error": f"An error occurred while processing the request: {str(e)}"},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )



# from django.shortcuts import render
# from django import forms

# class UserInputForm(forms.Form):
#     user_input = forms.CharField(widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Type your input here..."}), required=True)

# class WebpageForm(forms.Form):
#     webpage_url = forms.URLField(widget=forms.URLInput(attrs={"placeholder": "https://example.com"}), required=True)

# def fact_check_view(request):
#     user_input_result, user_input_error = None, None
#     webpage_result, webpage_error = None, None

#     if request.method == "POST":
#         if "submit_user_input" in request.POST:
#             user_form = UserInputForm(request.POST)
#             webpage_form = WebpageForm()  # Empty form for webpage fact-check

#             if user_form.is_valid():
#                 user_input = user_form.cleaned_data["user_input"]
#                 # Process the user input (e.g., pass to AI model)
#                 user_input_result = f"Fact-check result for: {user_input}"  # Replace with actual processing
#             else:
#                 user_input_error = "Invalid input!"

#         elif "submit_webpage" in request.POST:
#             webpage_form = WebpageForm(request.POST)
#             user_form = UserInputForm()  # Empty form for user input fact-check

#             if webpage_form.is_valid():
#                 webpage_url = webpage_form.cleaned_data["webpage_url"]
#                 # Process the webpage URL (e.g., analyze content)
#                 webpage_result = f"Fact-check result for: {webpage_url}"  # Replace with actual processing
#             else:
#                 webpage_error = "Invalid URL!"

#     else:
#         user_form = UserInputForm()
#         webpage_form = WebpageForm()

#     return render(request, "fact_check.html", {
#         "user_form": user_form,
#         "webpage_form": webpage_form,
#         "user_input_result": user_input_result,
#         "user_input_error": user_input_error,
#         "webpage_result": webpage_result,
#         "webpage_error": webpage_error,
#     })


