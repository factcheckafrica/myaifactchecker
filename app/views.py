from rest_framework import generics, viewsets, status, permissions
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import time
import re
import requests
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from langchain.schema import SystemMessage, HumanMessage
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Initialize components
analyzer = SentimentIntensityAnalyzer()

# Set environment variables (move these to settings.py or environment file)
os.environ["OPENAI_API_KEY"] = 'your-openai-key-here'
os.environ["TAVILY_API_KEY"] = 'your-tavily-key-here'
os.environ['GROQ_API_KEY'] = "your-groq-key-here"
os.environ["SERPER_API_KEY"] = "your-serper-key-here"


class FactcheckViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing fact-check operations
    """
    queryset = Factcheck.objects.all().order_by('-created_at')
    serializer_class = FactcheckSerializer
    permission_classes = [permissions.AllowAny]
    lookup_field = 'slug'
    
    def get_serializer_class(self):
        if self.action == 'create':
            return FactcheckCreateSerializer
        return FactcheckSerializer
    
    @action(detail=False, methods=['get'])
    def export_csv(self, request):
        """Export all factchecks as CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="factchecks.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'ID', 'User Input News', 'Fact Check Result', 'Sentiment Label', 
            'Genuine URLs', 'Non-Authentic URLs', 'Number of Genuine Sources', 
            'Genuine URLs and Dates', 'Non-Authentic Sources', 'Created At'
        ])
        
        factchecks = self.get_queryset()
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
                factcheck.non_authentic_sources,
                factcheck.created_at
            ])
        
        return response


class UserReportViewSet(viewsets.ModelViewSet):
    """
    ViewSet for user reports
    """
    queryset = UserReport.objects.all().order_by('-timestamp')
    serializer_class = UserReportSerializer
    permission_classes = [permissions.AllowAny]


class ActiveUserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for active users (read-only)
    """
    queryset = ActiveUser.objects.all()
    serializer_class = ActiveUserSerializer
    permission_classes = [permissions.IsAuthenticated]


# Translation utility
def llm_translate(text: str, target_lang: str) -> str:
    """Translate text using LLM"""
    if not text:
        return text
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        sys_msg = SystemMessage(content=(
            "You are a professional translator. Translate the user text faithfully into the target language. "
            "Do not add, remove, or embellish meaning. Keep names/URLs as-is. "
            "If the input is already in the target language, return it unchanged. "
            "Return only the translation—no explanations."
        ))
        usr_msg = HumanMessage(content=f"Target language: {target_lang}\n\nText:\n{text}")
        out = llm.invoke([sys_msg, usr_msg])
        return out.content.strip()
    except Exception as e:
        print(f"[LLM translate error] {e}")
        return text


def get_external_api_answer(query):
    """Get fact-check answer using external APIs"""
    # Initialize LLMs
    llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize search tools
    serper_tool = GoogleSerperAPIWrapper()
    tavily_tool = TavilySearch()

    # Run searches
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

    # Combine search content
    serper_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('snippet', '')}"
        for item in serper_data.get("organic", [])
    )

    tavily_snippets = "\n\n".join(
        f"- {item.get('title', '')}\n  {item.get('content', '')}"
        for item in tavily_data.get("results", [])
    )

    combined_context = f"{serper_snippets}\n\n{tavily_snippets}"

    # System prompt for fact-checking
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


class FactCheckAPIView(APIView):
    """
    Main API endpoint for fact-checking
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = FactcheckCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user_query = serializer.validated_data['user_input_news']
        
        try:
            # Get fact-check result
            fact_check_result, sources = get_external_api_answer(user_query)
            
            # Perform sentiment analysis
            sentiment_scores = analyzer.polarity_scores(user_query)
            sentiment_label = "Neutral"
            if sentiment_scores['compound'] > 0.05:
                sentiment_label = "Positive"
            elif sentiment_scores['compound'] < -0.05:
                sentiment_label = "Negative"
            
            # Create factcheck record
            factcheck = Factcheck.objects.create(
                user_input_news=user_query,
                fresult=fact_check_result,
                sentiment_label=sentiment_label,
                genuine_urls=sources,
                num_genuine_sources=len(sources)
            )
            
            response_serializer = FactcheckSerializer(factcheck)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": f"An error occurred while processing the request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MultiLanguageFactCheckAPIView(APIView):
    """
    API endpoint for multi-language fact-checking
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        user_query = request.data.get('user_input_news', '')
        target_language = request.data.get('language', 'english').lower()
        
        if not user_query:
            return Response(
                {"error": "user_input_news is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # If not English, translate to English first
            if target_language != 'english':
                query_en = llm_translate(user_query, target_lang="English")
            else:
                query_en = user_query
            
            # Get fact-check result in English
            fact_check_result_en, sources = get_external_api_answer(query_en)
            
            # Translate result back to target language if needed
            if target_language != 'english':
                fact_check_result = llm_translate(fact_check_result_en, target_lang=target_language.title())
            else:
                fact_check_result = fact_check_result_en
            
            # Perform sentiment analysis
            sentiment_scores = analyzer.polarity_scores(user_query)
            sentiment_label = "Neutral"
            if sentiment_scores['compound'] > 0.05:
                sentiment_label = "Positive"
            elif sentiment_scores['compound'] < -0.05:
                sentiment_label = "Negative"
            
            # Create factcheck record
            factcheck = Factcheck.objects.create(
                user_input_news=user_query,
                fresult=fact_check_result,
                sentiment_label=sentiment_label,
                genuine_urls=sources,
                num_genuine_sources=len(sources)
            )
            
            response_serializer = FactcheckSerializer(factcheck)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": f"An error occurred while processing the request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

