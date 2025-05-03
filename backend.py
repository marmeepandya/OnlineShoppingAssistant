import os
import re
import ollama
from tavily import TavilyClient
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json

# Load environment variables
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

def translate_to_english(query):
    prompt = f"""You are a translation assistant. Detect the language of this query and translate it to English.

Query: {query}

Only return the translated English text, nothing else."""
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a multilingual translation assistant."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()

def extract_filters_from_query(query):
    prompt = f"""
Extract brand names, colors, and price constraints from this product search query.
Return output in JSON format like:
{{"brands": ["brand1", "brand2"], "colors": ["color1", "color2"], "budget": 100}}

Query: "{query}"
"""
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are an intelligent query parser for product filters."},
        {"role": "user", "content": prompt}
    ])
    try:
        json_part = response["message"]["content"]
        return json.loads(json_part)
    except:
        return {"brands": [], "colors": [], "budget": None}

def is_recommendation_query(query):
    prompt = f"""Does the following user query sound like a product recommendation request? 
Respond only with Yes or No.

Query: {query}"""
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a smart classifier."},
        {"role": "user", "content": prompt}
    ])
    return "yes" in response['message']['content'].strip().lower()

def generate_recommendations(products, user_query):
    prompt = f"""You are a product recommendation assistant.
From the list below, select 1-2 top recommended products for the query: "{user_query}"

Explain briefly why each was chosen.

Products:
{json.dumps(products[:5], indent=2)}

Format your response like:
1. Product Title — Reason
2. Product Title — Reason"""

    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a helpful recommendation engine."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()

def is_comparison_query(query):
    prompt = f"""
You are an intelligent assistant that checks if a query implies product comparison.

Query: "{query}"

Answer "yes" if the user is asking to compare products (e.g., using words like compare, vs, difference, better than, etc.), otherwise answer "no".
Only respond with 'yes' or 'no'."""
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a shopping query classifier."},
        {"role": "user", "content": prompt}
    ])
    return "yes" in response["message"]["content"].lower()

def generate_comparison_table(products):
    comparison = {
        "Title": [],
        "Price": [],
        "Rating": [],
        "Delivery": [],
        "Key Features": [],
        "Pros/Cons": [],
    }

    for p in products:
        comparison["Title"].append(p.get("title", "N/A"))
        comparison["Price"].append(p.get("price", "N/A"))
        comparison["Rating"].append(str(p.get("rating", "N/A")))
        comparison["Delivery"].append("N/A")  # SERP API doesn't return delivery details by default

        # Generate key features and pros/cons using LLaMA
        content = p.get("content", "")
        if content:
            prompt = (
                f"Summarize key features and pros/cons of this product for comparison:\n\n{content}\n\n"
                "Format:\nKey Features: • ...\nPros: • ...\nCons: • ..."
            )
            response = ollama.chat(model="llama3", messages=[
                {"role": "system", "content": "You are a product comparison assistant."},
                {"role": "user", "content": prompt}
            ])
            summary = response['message']['content']
            features = summary.split("Pros:")[0].strip() if "Pros:" in summary else summary
            pros_cons = "Pros:" + summary.split("Pros:")[1].strip() if "Pros:" in summary else "N/A"
        else:
            features, pros_cons = "N/A", "N/A"

        comparison["Key Features"].append(features)
        comparison["Pros/Cons"].append(pros_cons)

    return comparison

# --- LLM-Powered Query Restructuring ---
def restructure_query_with_llama(user_query):
    prompt = f"""Restructure the following query to be concise and suitable for product search. 
Return only the restructured query as output (no explanation, no formatting):

{user_query}"""
    
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a helpful assistant that rewrites shopping queries for search."},
        {"role": "user", "content": prompt}
    ])
    
    restructured = response['message']['content'].strip()

    # Return only the first line (in case model still includes explanation)
    return restructured.splitlines()[0]

# --- Core Backend Functions ---
def extract_budget(query):
    match = re.search(r"under\s+(\d+)", query)
    return int(match.group(1)) if match else None

def search_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "tbm": "shop",
        "location": "Germany",
        "gl": "de",
        "api_key": serpapi_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    product_results = results.get("shopping_results", [])

    products = []
    for item in product_results:
        title = item.get("title")
        link = item.get("link")
        price = item.get("price")
        rating = item.get("rating")
        reviews = item.get("reviews")
        thumbnail = item.get("thumbnail")
        
        if title and link:
            products.append({
                "title": title,
                "link": link,
                "price": price,
                "rating": rating,
                "reviews": reviews,
                "image": thumbnail
            })

    return products

def enrich_with_tavily(products):
    enriched = []
    specifications = {}
    for p in products:
        url = p.get("link")
        if not url:
            continue
        try:
            result = tavily_client.search(query=url, search_depth="basic", max_results=1)
            content = result['results'][0].get("content", "")
        except:
            content = ""
        
        # 🧠 Extract availability and delivery using LLaMA
        availability, delivery = "Unknown", "Unknown"
        if content:
            prompt = (
                f"From the following product page content, extract:\n"
                f"1. Is the product available (Yes/No)?\n"
                f"2. Is there any delivery info (e.g., free shipping, 2-day delivery)?\n\n"
                f"Content:\n{content}\n\n"
                f"Respond in this format:\n"
                f"Availability: Yes/No\nDelivery: ...\n"
            )
            try:
                response = ollama.chat(model="llama3", messages=[
                    {"role": "system", "content": "You are an expert product info extractor."},
                    {"role": "user", "content": prompt}
                ])
                lines = response['message']['content'].splitlines()
                for line in lines:
                    if line.lower().startswith("availability"):
                        availability = line.split(":")[1].strip()
                    if line.lower().startswith("delivery"):
                        delivery = line.split(":")[1].strip()
            except:
                pass

            spec_prompt = (
                f"Extract structured specifications from the following product info:\n\n{content}\n\n"
                f"Return output as JSON with keys like display, battery, camera, storage, material, etc."
            )
            try:
                spec_response = ollama.chat(model="llama3", messages=[
                    {"role": "system", "content": "You are a spec extraction expert."},
                    {"role": "user", "content": spec_prompt}
                ])
                specs_raw = spec_response['message']['content']
                specifications = json.loads(specs_raw)
            except:
                specifications = {}

        enriched.append({
            "title": p.get("title"),
            "url": url,
            "price": p.get("price"),
            "rating": p.get("rating"),
            "reviews": p.get("reviews"),
            "image": p.get("image"),
            "content": content,
            "availability": availability,
            "delivery": delivery,
            "specifications": specifications
        })
    return enriched

def validate_results(results):
    keywords = ["buy", "price", "review", "deal", "discount", "shop"]
    banned_domains = ["reddit.com", "quora.com", "youtube.com"]
    filtered = []
    for r in results:
        url = r["url"].lower()
        content = r.get("content", "").lower()
        if any(b in url for b in banned_domains):
            continue
        if any(k in content for k in keywords):
            filtered.append(r)
    return filtered

def summarize_with_llama(content):
    prompt = f"Summarize this product for a shopping assistant:\n\n{content}"
    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a helpful shopping assistant."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def rank_results(results, user_query, selected_site=None, max_price=None):
    keywords = ["best", "cheap", "value", "recommended", "sale"]
    trusted = ["zalando", "amazon", "c-and-a", "otto", "patpat", "kleinanzeigen"]

    budget = extract_budget(user_query)
    ranked = []
    for r in results:
        score = 0
        content = r.get("content", "").lower()
        url = r["url"].lower()
        title = r["title"].lower()
        score += sum(1 for k in keywords if k in content or k in title)
        score += 3 if any(t in url for t in trusted) else 0

        if selected_site and selected_site.lower() not in url:
            continue

        if max_price and r.get("price"):
            price_str = re.sub(r"[^\d.]", "", r["price"])
            try:
                price = float(price_str)
                if price > max_price:
                    continue
            except:
                pass

        ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]

def extract_price(price_str):
    if not price_str:
        return float('inf')
    try:
        return float(re.sub(r"[^\d.]", "", price_str))
    except:
        return float('inf')

def sort_results(results, sort_by):
    if sort_by == "Price: Low to High":
        return sorted(results, key=lambda r: extract_price(r.get("price")))
    elif sort_by == "Price: High to Low":
        return sorted(results, key=lambda r: extract_price(r.get("price")), reverse=True)
    return results
