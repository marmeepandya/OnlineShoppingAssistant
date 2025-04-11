import os
import re
import ollama
import streamlit as st
from tavily import TavilyClient
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load API keys from .env
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

# --- Helper Functions ---
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
    return results.get("shopping_results", [])

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
    trusted = ["zalando", "amazon", "c-and-a", "otto", "patpat"]
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

# --- Streamlit UI ---
st.set_page_config(page_title="Online Shopping Assistant", layout="wide")
st.title("🛍️ Online Shopping Assistant")

query = st.text_input("Enter your product search query:", placeholder="e.g. dress for girls under 50 euros")

col1, col2 = st.columns(2)
with col1:
    selected_site = st.selectbox("Filter by website (optional):", ["", "zalando", "amazon", "c-and-a", "otto", "patpat"])
with col2:
    max_price = st.slider("Maximum Price (€):", 5, 200, 50)

if st.button("Search") and query:
    with st.spinner("Searching for products..."):
        serp_results = search_serpapi(query)
        enriched = enrich_with_tavily(serp_results)
        validated = validate_results(enriched)
        ranked = rank_results(validated, query, selected_site, max_price)

    if not ranked:
        st.warning("No relevant products found.")
    else:
        st.success(f"Showing top {min(10, len(ranked))} results")
        for r in ranked[:10]:
            with st.container():
                st.subheader(r["title"])
                st.write(f"**Price**: {r.get('price', 'N/A')}")
                st.write(f"[View Product]({r['url']})")
                summary = summarize_with_llama(r["content"]) if r["content"] else "No summary available."
                st.caption(summary)

