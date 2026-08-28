"""
Test script for Serper and Tavily search APIs
Run: python test_search_apis.py
"""
from dotenv import load_dotenv
import os

load_dotenv()

# Get API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

print("\n" + "="*60)
print("SEARCH API TEST")
print("="*60)

# Test query
test_query = "Nigeria president Tinubu 2024"

# ===== Test Serper =====
print("\n--- TESTING SERPER API ---")
if SERPER_API_KEY:
    print(f"API Key: {SERPER_API_KEY[:10]}...{SERPER_API_KEY[-4:]}")
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        serper = GoogleSerperAPIWrapper()
        results = serper.results(test_query)

        print(f"Response type: {type(results)}")
        if isinstance(results, dict):
            print(f"Keys: {results.keys()}")
            organic = results.get("organic", [])
            print(f"Organic results: {len(organic)}")
            for i, item in enumerate(organic[:3]):
                print(f"  [{i+1}] {item.get('title', 'No title')[:60]}")
                print(f"      URL: {item.get('link', 'No link')[:80]}")
        else:
            print(f"Unexpected response: {str(results)[:200]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
else:
    print("SKIPPED - No API key")

# ===== Test Tavily =====
print("\n--- TESTING TAVILY API ---")
if TAVILY_API_KEY:
    print(f"API Key: {TAVILY_API_KEY[:10]}...{TAVILY_API_KEY[-4:]}")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    try:
        from langchain_tavily import TavilySearch
        tavily = TavilySearch(max_results=5, topic="general")
        results = tavily.invoke({"query": test_query})

        print(f"Response type: {type(results)}")
        if isinstance(results, dict):
            print(f"Keys: {results.keys()}")
            items = results.get("results", [])
            print(f"Results count: {len(items)}")
            for i, item in enumerate(items[:3]):
                print(f"  [{i+1}] {item.get('title', 'No title')[:60]}")
                print(f"      URL: {item.get('url', 'No url')[:80]}")
        elif isinstance(results, list):
            print(f"List with {len(results)} items")
            for i, item in enumerate(results[:3]):
                if isinstance(item, dict):
                    print(f"  [{i+1}] {item.get('title', 'No title')[:60]}")
                else:
                    print(f"  [{i+1}] {str(item)[:80]}")
        else:
            print(f"Response: {str(results)[:300]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
else:
    print("SKIPPED - No API key")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60 + "\n")
