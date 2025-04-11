import os
import re
import ollama
from tavily import TavilyClient
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

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
        rating = item.get("rating")        # ⭐️ Extract rating
        reviews = item.get("reviews")      # 📝 Extract number of reviews

        if title and link:
            products.append({
                "title": title,
                "link": link,
                "price": price,
                "rating": rating,
                "reviews": reviews
            })

    return products

def enrich_with_tavily(products):
    enriched = []
    for p in products:
        url = p.get("link")
        if not url:
            continue
        try:
            result = tavily_client.search(query=url, search_depth="basic", max_results=1)
            content = result['results'][0].get("content", "")
        except:
            content = ""
        enriched.append({
            "title": p.get("title"),
            "url": url,
            "price": p.get("price"),
            "rating": p.get("rating"),      # ✅ Pass through
            "reviews": p.get("reviews"),    # ✅ Pass through
            "content": content
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
        return float('inf')  # unknown prices go last
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
