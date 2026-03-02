from googlesearch import search
import os
import requests
import json
import nltk
import openai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
api_key = os.getenv("OPENAI_API_KEY", "")
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

analyzer = SentimentIntensityAnalyzer()

# Example list of domains
domains = [
    "punchng.com",
    "www.vanguardngr.com",
    "www.thisdaylive.com",
    "www.guardian.ng",
    "dailytrust.com",
    "www.premiumtimesng.com",
    "www.leadership.ng",
    "www.sunnewsonline.com",
    "dailypost.ng",
    'www.arise.tv',
    'www.thecable.ng',
    'thenationonlineng.net',
    "www.channelstv.com",
    "ask.un.org",
    "www.africa.com",

    "lenqueteur-niger.com",
    "www.lesahel.org",
    "nigerexpress.info",
    "www.journalduniger.com",
    "www.actuniger.com",
    "www.nigerdiaspora.net",
    "sahelien.com",

    "www.cameroon-tribune.cm",
    "camerounexpress.net",
    "www.lemessager.fr",
    "www.africanews.com",
    "www.crtv.cm",
    "www.jeuneafrique.com"

    "ortb.bj",
    "www.lematin.ch",
    "www.lautrequotidien.fr",
    "www.24haubenin.info",
    "www.fratmat.info",
    "www.letelegramme.fr",
    "lanouvelletribune.info",
]

def check_news_authenticity(query):
    genuine_urls = []
    non_authentic_urls = []
    d = []
    n = []

    for i in search(query, tld="co.in", num=10, stop=20, pause=2):
        news_url = i
        keywords = i.split('/')
        domain_name = keywords[2]
        d.append(domain_name)

        if domain_name in domains:
            genuine_urls.append(news_url)
        else:
            non_authentic_urls.append(news_url)

    legit = set(domains) & set(d)
    legit = list(legit)
    legit_length = len(legit)
    non_legit_length = len(d) - legit_length

    for item in d:
        if item not in legit:
            n.append(item)

    if legit_length >= 4:
        authenticity_result = "Credible"
    else:
        authenticity_result = "Unreliable"

    genuine_sources = legit
    num_genuine_sources = legit_length
  
   

    return authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, n,non_legit_length

def FactCheck(query):
    try:
        payload = {
            'key': 'AIzaSyCg0hJErtkTLTWn90iw5GB49NTzkeCp-qk',
            'query': query
        }
        url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
        response = requests.get(url, params=payload)

        if response.status_code == 200:
            result = json.loads(response.text)
            if "claims" in result and result["claims"]:
                topRating = result["claims"][0]
                if "claimReview" in topRating and topRating["claimReview"]:
                    claimReview = topRating["claimReview"][0]["textualRating"]
                    claimVal = "According to " + str(topRating["claimReview"][0]['publisher']['name']) + " that claim is " + str(claimReview)
                    claimVal1 = str(claimReview)
                    return claimVal, claimVal1
                else:
                    return "No claim review found for the top claim."
            else:
                return "No claims found for the query."
        else:
            return "Failed to retrieve data from the Fact Check API."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def fact_check_statement(statement_to_check):
    try:
        openai.api_key = api_key
        # Define a list of messages for the chat-based model
        messages = [
            {"role": "system", "content": "You are a fact-checking assistant."},
            {"role": "user", "content": statement_to_check}
        ]

        # Make a request to the GPT-3 API using the chat-based endpoint
        response = openai.ChatCompletion.create(
            model="GPT-4o mini",
            messages=messages,
        )

        # Extract the model's response
        fact_check_result = response['choices'][0]['message']['content'].strip()

        return fact_check_result if fact_check_result else "No response"

    except Exception as e:
        return f"An error occurred: {str(e)}"
    

