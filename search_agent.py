import os
import re
from tavily import TavilyClient
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load .env and API key
load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=api_key)

# 🔍 Search Agent
def search_products(query):
    print(f"Searching for: {query}")
    results = client.search(
        query=query,
        search_depth="advanced",
        include_images=True,
        max_results=10
    )
    return results['results']

# 🧹 Validator Agent
def validate_results(results):
    validated = []
    keywords = [
        "buy", "shop", "price", "sale", "discount", "deal", 
        "review", "specs", "features", "top pick", "best", 
        "offer", "affordable", "cost", "where to buy"
    ]
    banned_domains = ["reddit.com", "quora.com", "youtube.com"]

    for result in results:
        title = result['title'].lower()
        url = result['url'].lower()
        content = result.get('content', '').lower()

        # Check for useful keywords
        is_relevant = any(kw in title or kw in content for kw in keywords)

        # Check if domain is not blacklisted
        is_not_banned = not any(domain in url for domain in banned_domains)

        if is_relevant and is_not_banned:
            validated.append(result)

    return validated

# 📋 Display function
def display_results(results):
    if not results:
        print("No valid product results found.")
        return

    for idx, result in enumerate(results, 1):
        print(f"\nResult #{idx}")
        print("Title:", result['title'])
        print("URL:", result['url'])
        print("Snippet:", result.get('content', 'No description'))

def rank_results(results, user_query):
    ranked = []

    # Keywords to boost relevance
    good_keywords = [
        "best", "top", "recommended", "value", "cheap", 
        "durable", "popular", "new", "trending", "comfortable", 
        "eco-friendly", "stylish", "lightweight", "sale", "affordable"
    ]

    trusted_sources = [
        "cnet.com", "whathifi.com", "techradar.com", "pcmag.com", 
        "rtings.com", "soundguys.com", "nytimes.com", "goodhousekeeping.com", 
        "bestproducts.com", "thewirecutter.com", "amazon.com", "etsy.com",
        "wayfair.com", "zara.com", "h&m.com", "nike.com"
    ]


    for result in results:
        score = 0
        content = result.get('content', '').lower()
        title = result['title'].lower()
        url = result['url'].lower()

        # Keyword match score
        score += sum(1 for kw in good_keywords if kw in content or kw in title)

        # Trusted domain bonus
        if any(domain in url for domain in trusted_sources):
            score += 3

        # Budget match (checks if content contains a number close to user budget)
        if "under" in user_query or "less than" in user_query:
            if "150" in content or "100" in content or "€" in content or "$" in content:
                score += 2

        ranked.append((score, result))

    # Sort by score, highest first
    ranked.sort(key=lambda x: x[0], reverse=True)

    # Return just the sorted result list
    return [r[1] for r in ranked]

def extract_budget(query):
    match = re.search(r"under\s+(\d+)", query)
    if match:
        return int(match.group(1))
    return None


def search_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "tbm": "shop",
        "location": "Germany",
        "gl": "de",  # Google Germany
        "api_key": os.getenv("SERPAPI_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    product_results = results.get("shopping_results", [])

    links = []
    for item in product_results:
        title = item.get("title")
        link = item.get("link")
        price = item.get("price")
        if title and link:
            links.append({
                "title": title,
                "link": link,
                "price": price
            })

    return links

def enrich_with_tavily(serp_links):
    enriched = []

    for item in serp_links:
        url = item["link"]
        title = item["title"]
        price = item.get("price")

        try:
            result = client.search(query=url, search_depth="basic", max_results=1)
            content = result['results'][0].get("content", "")
        except Exception as e:
            content = ""

        enriched.append({
            "title": title,
            "url": url,
            "price": price,
            "content": content
        })

    return enriched

def summarize_results(results):
    print("\n📊 Top Product Summary:")
    print(f"{'Title':<50} | {'Price':<10} | Link")
    print("-" * 100)
    for r in results[:5]:
        print(f"{r['title'][:50]:<50} | {r.get('price', 'N/A'):<10} | {r['url']}")

# 🧪 Main Execution
if __name__ == "__main__":
    # user_query = input("Enter your product search query: ")
    # raw_results = search_products(user_query)
    # filtered_results = validate_results(raw_results)
    # ranked_results = rank_results(filtered_results, user_query)
    # display_results(ranked_results)
    user_query = input("Enter your product search query: ")
    
    print("\n🔎 Fetching products via SERPApi...")
    serp_results = search_serpapi(user_query)
    
    print(f"Found {len(serp_results)} products. Enriching with Tavily...")
    enriched = enrich_with_tavily(serp_results)

    # for idx, product in enumerate(enriched, 1):
    #     print(f"\nProduct #{idx}")
    #     print("Title:", product["title"])
    #     print("Price:", product.get("price", "N/A"))
    #     print("URL:", product["url"])
    #     print("Summary Snippet:", product["content"][:300] + "...")

    # 🔍 Run Validator + Ranker on enriched results
    print("\n🧹 Validating results...")
    filtered_results = validate_results(enriched)

    print(f"✅ {len(filtered_results)} results passed validation. Now ranking...")

    ranked_results = rank_results(filtered_results, user_query)

    # Display final results
    display_results(ranked_results)

    summarize_results(ranked_results)