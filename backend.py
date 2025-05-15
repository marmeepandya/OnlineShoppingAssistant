import os
import re
import json
import ollama
import time
import logging
import functools
import concurrent.futures
from tavily import TavilyClient
from dotenv import load_dotenv
import serpapi
from cachetools import cached, TTLCache
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

# Consolidated query processing function that replaces 4 separate LLM calls
def process_query(query):
    """Process a user query in a single LLM call instead of multiple sequential calls"""
    prompt = f"""Analyze this shopping query thoroughly: "{query}"

    1. If not in English, translate to English (if already English, just repeat it).
    2. Restructure for product search (make it concise, clear, specific).
    3. Is this a comparison query? Answer only yes/no (looking for terms like "vs", "compare", "better", "difference").
    4. Is this a recommendation query? Answer only yes/no (looking for terms like "suggest", "recommend", "best for me").

    Format your response exactly as follows:
    Translated: [translated text]
    Restructured: [restructured query]
    Comparison: [yes/no]
    Recommendation: [yes/no]
    """

    try:
        response = ollama.chat(model="llama3.1", messages=[
            {"role": "system", "content": "You are a precise query analysis assistant."},
            {"role": "user", "content": prompt}
        ])

        result = response['message']['content']

        # Parse the response
        translated = re.search(r"^Translated: (.+)$", result, re.MULTILINE)
        restructured = re.search(r"^Restructured: (.+)$", result, re.MULTILINE)
        comparison = re.search(r"^Comparison: (.+)$", result, re.MULTILINE)
        recommendation = re.search(r"^Recommendation: (.+)$", result, re.MULTILINE)

        processed_query = {
            "translated": translated.group(1) if translated else query,
            "restructured": restructured.group(1) if restructured else query,
            "is_comparison": "yes" in comparison.group(1).lower() if comparison else False,
            "is_recommendation": "yes" in recommendation.group(1).lower() if recommendation else False
        }

        return processed_query

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Fallback response
        return {
            "translated": query,
            "restructured": query,
            "is_comparison": False,
            "is_recommendation": False
        }

# Modified search_serpapi function that includes validation
def search_serpapi(query):
    """
    Search for products with caching, timeout, and validation.
    Limits the number of products returned to improve performance.
    
    Args:
        query (str): The search query
        limit (int): Maximum number of products to return (default: 5)
        
    Returns:
        list: Limited list of product dictionaries
    """
    try:
        params = {
            "api_key": serpapi_key,
            "engine": "google_shopping",
            "q": query,
            #"num": 20, 
            "gl": "de",
            "tbm": "shop",
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: serpapi.search(params))
            results = future.result().as_dict()

        product_results = results.get("shopping_results", [])
        
        # Ensure we don't exceed the requested limit
        # product_results = product_results[:50]

        products = []
        for r in product_results:
            products.append({
                "product_id": r.get('product_id', ''),
                "title": r.get('title', ''),
                "url": r.get('product_link', ''),
                "source": r.get('source', ''),
                "price": r.get('extracted_price', ''),
                "old_price": r.get('extracted_old_price', ''),
                "rating": r.get('rating', ''),
                "reviews": r.get('reviews', ''),
                "extensions": r.get('extensions', []),
                "image": r.get('thumbnail', ''),
            })
        
        logger.info(f"Successfully retrieved {len(products)} products from SerpAPI")
        return products

    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def extract_specifications(products):
    detailed_products = []
    tool = TavilySearchResults(
            max_results=2,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
    )
    for product in products:
        try:
            search_query = f"{product['title']} product technical description details"
            details = tool.invoke(search_query)
            
            # The error is here - details appears to be a list of strings, not dictionaries
            # Let's properly handle the different possible response formats
            content = ""
            if details and isinstance(details, list):
                for item in details:
                    if isinstance(item, dict) and 'content' in item:
                        content += item['content'] + '\n'
                    elif isinstance(item, str):
                        content += item + '\n'
            
            # If we couldn't extract any content, use a placeholder
            if not content:
                content = "No details found."
                
            detailed_products.append({
                **product,
                "details": content
            })
        except Exception as e:
            logger.error(f"Error extracting specifications for product {product.get('title')}: {e}")
            detailed_products.append({
                **product,
                "details": "No details found."
            })
    
    return detailed_products

def rank_products(products, user_query, max_price=None):
    """
    Rank products using a point-based system and LLaMA 3 for analysis.
    More flexible than the previous implementation, keeping more products in results.
    
    Args:
        products (list): List of product dicts with fields like title, price, rating, content
        user_query (str): The original user search query
        max_price (int, optional): Maximum price constraint set by the user
        
    Returns:
        tuple: (ranked_products, top_recommendations)
            - ranked_products: List of ranked product dicts
            - top_recommendations: Dict with information about the top 1-3 recommended products
    """
    import re
    import logging
    import ollama
    
    logger = logging.getLogger(__name__)
    
    if not products:
        return [], {"text": "No products found.", "products": []}
    
    # Initial score assignment based on available data
    for product in products:
        # Start with base score
        product['score'] = 50
        
        # Convert price to float for comparison and scoring
        try:
            if product.get('price'):
                price_value = float(re.sub(r'[^\d.]', '', str(product['price'])))
                product['price_numeric'] = price_value
                
                # Price-based scoring (lower price = higher score, max 20 points)
                # We'll normalize this later when we know the price range
                product['price_score'] = price_value
            else:
                product['price_numeric'] = None
                product['price_score'] = 0
        except (ValueError, TypeError):
            product['price_numeric'] = None
            product['price_score'] = 0
        
        # Rating-based scoring (up to 20 points)
        if product.get('rating'):
            try:
                rating_value = float(str(product['rating']).split()[0])
                # Convert 5-star rating to points (5 stars = 20 points)
                product['score'] += min(rating_value * 4, 20)
            except (ValueError, IndexError):
                pass
        
        # Review count scoring (up to 10 points)
        if product.get('reviews'):
            try:
                # Extract number from formats like "1,234 reviews"
                reviews_str = str(product['reviews']).replace(',', '')
                review_count = int(re.search(r'\d+', reviews_str).group())
                
                # Log scale for reviews: 0 reviews = 0 points, 10 reviews = 3 points, 
                # 100 reviews = 6 points, 1000+ reviews = 10 points
                if review_count > 0:
                    import math
                    review_points = min(3 * math.log10(review_count) + 3, 10)
                    product['score'] += review_points
            except (ValueError, AttributeError):
                pass
        
        # Detail availability scoring (up to 10 points)
        if product.get('details') and product['details'] != "No details found.":
            # More detailed descriptions get more points
            detail_length = len(str(product.get('details', '')))
            detail_points = min(detail_length / 100, 10)  # 1 point per 100 chars, max 10
            product['score'] += detail_points
    
    # If we have price data, normalize the price scores across products
    products_with_price = [p for p in products if p.get('price_numeric') is not None]
    if products_with_price:
        min_price = min(p['price_numeric'] for p in products_with_price)
        max_price = max(p['price_numeric'] for p in products_with_price)
        
        if max_price > min_price:
            price_range = max_price - min_price
            for product in products_with_price:
                # Invert and normalize: lower prices get higher scores
                # 20 points for lowest price, scaled down for higher prices
                normalized_price_score = 20 * (1 - (product['price_numeric'] - min_price) / price_range)
                product['score'] += normalized_price_score
    
    # Prepare subset of products for LLM analysis - select all products but limit text length
    filtered_products = []
    for product in products:
        # Only apply price filter if explicitly specified
        if max_price is not None and product.get('price_numeric') is not None:
            if product['price_numeric'] > max_price:
                # Mark as filtered but keep in the results
                product['filtered_by_price'] = True
                product['score'] -= 30  # Penalty for exceeding price limit
                
        # Make sure we have basic info needed for display
        if product.get('title'):
            filtered_products.append(product)
    
    # If no products pass filtering, try again with no filtering
    if not filtered_products:
        filtered_products = [p for p in products if p.get('title')]
    
    # If still no products, return empty results
    if not filtered_products:
        return [], {"text": "No products matched your criteria.", "products": []}
    
    # Prepare text for LLM - take top 10 by initial score for detailed analysis
    filtered_products.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_products_for_analysis = filtered_products[:10]
    
    products_details = []
    for idx, p in enumerate(top_products_for_analysis, 1):
        product_info = f"Product {idx}:\n"
        product_info += f"Title: {p['title']}\n"
        
        if p.get('price'):
            product_info += f"Price: {p['price']}\n"
        
        # Add rating and reviews if available
        if p.get('rating'):
            product_info += f"Rating: {p['rating']}\n"
        if p.get('reviews'):
            product_info += f"Reviews: {p['reviews']}\n"
            
        # Truncate details to reduce token usage
        details = p.get('details', '')
        if details and details != "No details found.":
            details_snippet = details[:500] + "..." if len(details) > 500 else details
            product_info += f"Details: {details_snippet}\n\n"
        
        products_details.append(product_info)
    
    products_text = "\n".join(products_details)
    
    prompt = f"""Original query: "{user_query}"

I need you to analyze these products and rank them based on how well they match the query.
Consider:
1. Relevance to the query "{user_query}"
2. Quality (ratings and reviews)
3. Value for money
4. Features and specifications

Here are the products to analyze:

{products_text}

For each product, assign:
1. A relevance score from 1-10 where 10 is perfect match to query
2. A brief justification for this score (1-2 sentences)

Format your response with the product number followed by the score and justification:
Product 1: Score X - [justification]
Product 2: Score Y - [justification]
...

Do NOT include any introductory text or conclusion, just the product scores and justifications.
"""

    try:
        response = ollama.chat(model="llama3.1", messages=[
            {"role": "system", "content": "You are a product ranking assistant focused on accurately matching products to user needs."},
            {"role": "user", "content": prompt}
        ])

        ranking_output = response['message']['content'].strip()
        
        # Parse the LLM response
        product_rankings = {}
        for line in ranking_output.split('\n'):
            if not line.strip():
                continue
                
            # Extract product number, score and justification
            match = re.match(r"Product (\d+):.*Score (\d+).*-\s*(.*)", line)
            if match:
                product_num = int(match.group(1))
                score = int(match.group(2))
                justification = match.group(3).strip()
                
                if 1 <= product_num <= len(top_products_for_analysis):
                    product_rankings[product_num] = {
                        "llm_score": score,
                        "rank_reason": justification
                    }
        
        # Add LLM scores to products
        for idx, product in enumerate(top_products_for_analysis, 1):
            if idx in product_rankings:
                # LLM score is on a scale of 1-10, multiply by 10 to match other scores
                product["llm_score"] = product_rankings[idx]["llm_score"] * 10
                product["score"] += product["llm_score"]  # Add to total score
                product["rank_reason"] = product_rankings[idx]["rank_reason"]
            else:
                # Default values if parsing failed
                product["llm_score"] = 0
                product["rank_reason"] = "No specific ranking information available."
        
        # Now generate detailed descriptions for top products
        top_products = top_products_for_analysis[:5]  # Take top 5 for detailed descriptions
        
        detailed_prompt = f"""The user searched for: "{user_query}"

For each of these top products, write a helpful explanation (2-3 sentences) about why it would be a good match for the user's query.
Focus on specific features, value proposition, and how well it addresses the user's needs.

{products_text[:5000]}  # Limit text to avoid token limits

Format your response as:
PRODUCT 1 DESCRIPTION:
[Your explanation here]

PRODUCT 2 DESCRIPTION:
[Your explanation here]
...

After all product descriptions, add a section titled "TOP RECOMMENDATIONS:" where you highlight the 1-3 best products and explain why they stand out as the top choices (3-4 sentences).
"""
        
        try:
            detailed_response = ollama.chat(model="llama3.1", messages=[
                {"role": "system", "content": "You are a product recommendation expert who provides concise, helpful product analyses."},
                {"role": "user", "content": detailed_prompt}
            ])
            
            detailed_output = detailed_response['message']['content'].strip()
            
            # Parse detailed descriptions for each product
            descriptions = {}
            top_recommendations_text = ""
            
            # Split by "TOP RECOMMENDATIONS:" to separate product descriptions from recommendations
            parts = detailed_output.split("TOP RECOMMENDATIONS:")
            product_descriptions_part = parts[0].strip()
            
            if len(parts) > 1:
                top_recommendations_text = parts[1].strip()
            
            # Extract individual product descriptions
            products_desc_blocks = re.split(r'PRODUCT \d+ DESCRIPTION:', product_descriptions_part)
            for i, desc_block in enumerate(products_desc_blocks[1:], 1):  # Skip the first empty split
                descriptions[i] = desc_block.strip()
            
            # Add descriptions to products
            for idx, product in enumerate(top_products, 1):
                if idx in descriptions:
                    product["detailed_description"] = descriptions[idx]
                else:
                    product["detailed_description"] = f"This {product['title']} appears to be a good match for your search."
            
            # Spread descriptions to other products with similar titles
            for product in filtered_products:
                if not product.get("detailed_description"):
                    # Try to find a similar product with a description
                    for described_product in top_products:
                        if described_product.get("detailed_description") and similar_titles(product['title'], described_product['title']):
                            product["detailed_description"] = described_product["detailed_description"]
                            break
                    
                    # If still no description, create a generic one
                    if not product.get("detailed_description"):
                        product["detailed_description"] = f"This product appears to match your search for '{user_query}'."
            
            # Create top recommendations object
            # Sort by score for final ranking
            filtered_products.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            top_recommendations = {
                "text": top_recommendations_text if top_recommendations_text else "Based on your search, these are the best options available.",
                "products": filtered_products[:3]  # Get top 3 ranked products
            }
                
        except Exception as e:
            logger.error(f"Error generating detailed descriptions: {e}")
            # Add default descriptions if detailed generation fails
            for product in filtered_products:
                if not product.get("detailed_description"):
                    product["detailed_description"] = f"This product matches your search for '{user_query}'."
            
            top_recommendations = {
                "text": "Based on your search, these are the best options available.",
                "products": filtered_products[:3]
            }
        
        # Calculate final ranks based on score
        filtered_products.sort(key=lambda x: x.get('score', 0), reverse=True)
        for rank, product in enumerate(filtered_products, 1):
            product["rank"] = rank
        
        # Clean up temporary fields
        for product in filtered_products:
            # Remove scoring fields that shouldn't be in the final output
            fields_to_remove = ['score', 'price_score', 'price_numeric', 'llm_score', 'filtered_by_price']
            for field in fields_to_remove:
                if field in product:
                    del product[field]
                
        return filtered_products, top_recommendations

    except Exception as e:
        logger.error(f"Error ranking products: {e}")
        # Fallback: just sort by rating and return all products
        for product in filtered_products:
            # Clean up temporary fields
            for field in ['score', 'price_score', 'price_numeric', 'llm_score', 'filtered_by_price']:
                if field in product:
                    del product[field]
            
            # Add missing fields
            if not product.get("rank_reason"):
                product["rank_reason"] = "Product appears to match your search criteria."
            if not product.get("detailed_description"):
                product["detailed_description"] = f"This product matches your search for '{user_query}'."
            
        # Simple sorting by rating as fallback
        try:
            filtered_products.sort(key=lambda x: float(str(x.get('rating', '0')).split()[0]), reverse=True)
        except:
            pass  # If rating sorting fails, keep original order
            
        # Assign ranks
        for rank, product in enumerate(filtered_products, 1):
            product["rank"] = rank
            
        default_top_recommendations = {
            "text": "Based on your search, these products might be a good match.",
            "products": filtered_products[:3] if len(filtered_products) >= 3 else filtered_products
        }
        
        return filtered_products, default_top_recommendations

def similar_titles(title1, title2):
    """Check if two product titles are similar enough to share descriptions"""
    # Convert to lowercase and remove common punctuation
    title1 = title1.lower().replace(',', '').replace('.', '')
    title2 = title2.lower().replace(',', '').replace('.', '')
    
    # If one title is completely contained in the other, they're similar
    if title1 in title2 or title2 in title1:
        return True
    
    # Count matching words
    words1 = set(title1.split())
    words2 = set(title2.split())
    common_words = words1.intersection(words2)
    
    # If they share at least 3 significant words, consider them similar
    return len(common_words) >= 3

def extract_price(price_str):
    """Extract numeric price from string"""
    if not price_str:
        return float('inf')
    try:
        return float(re.sub(r"[^\d.]", "", price_str))
    except:
        return float('inf')

def sort_results(results, sort_by):
    """Sort results by specified criteria"""
    if sort_by == "Price: Low to High":
        return sorted(results, key=lambda r: extract_price(r.get("price", "inf")))
    elif sort_by == "Price: High to Low":
        return sorted(results, key=lambda r: extract_price(r.get("price", "0")), reverse=True)
    return results

def extract_budget(query):
    """Extract budget from query"""
    patterns = [
        r"under\s+(\d+)",
        r"less than\s+(\d+)",
        r"below\s+(\d+)",
        r"max\s+(\d+)",
        r"maximum\s+(\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def summarize_with_llama(content):
    """Summarize product content with caching"""
    if not content or len(content) < 100:
        return "No detailed description available."

    # Truncate content to reduce token usage
    content_sample = content[:3000] if len(content) > 3000 else content

    prompt = f"""Summarize this product description concisely (max 2-3 sentences):

{content_sample}

Focus on key features, benefits, and unique selling points."""

    try:
        response = ollama.chat(model="llama3.1", messages=[
            {"role": "system", "content": "You are a concise product summarizer."},
            {"role": "user", "content": prompt}
        ])

        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return content[:200] + "..." if len(content) > 200 else content

def generate_comparison_table(products):
    """Generate comparison table for products"""
    if len(products) < 2:
        return {"Title": ["N/A"], "Price": ["N/A"], "Rating": ["N/A"],
                "Key Features": ["N/A"], "Pros/Cons": ["N/A"]}

    comparison = {
        "Title": [],
        "Price": [],
        "Rating": [],
        "Key Features": [],
        "Pros/Cons": [],
    }

    # Add basic information first
    for p in products[:3]:  # Limit to top 3
        comparison["Title"].append(p.get("title", "N/A"))
        comparison["Price"].append(p.get("price", "N/A"))
        comparison["Rating"].append(str(p.get("rating", "N/A")))

        # Pre-fill with empty values
        comparison["Key Features"].append("Loading...")
        comparison["Pros/Cons"].append("Loading...")

    # Function to process a single product
    def process_product_comparison(idx, product):
        content = product.get("content", "")
        if not content or len(content) < 100:
            return idx, "No detailed information available.", "N/A"

        cache_key = f"comparison_{hash(content[:500])}"
        if cache_key in llm_cache:
            result = llm_cache[cache_key]
            return idx, result["features"], result["pros_cons"]

        prompt = f"""Extract from this product description:
1. Key Features (bullet list of 3-4 main features)
2. Pros and Cons (2-3 of each)

Product: {product.get('title')}
Description: {content[:2000] if len(content) > 2000 else content}

Format your response exactly as:
Key Features:
• Feature 1
• Feature 2
• Feature 3

Pros:
• Pro 1
• Pro 2

Cons:
• Con 1
• Con 2"""

        try:
            response = ollama.chat(model="llama3.1", messages=[
                {"role": "system", "content": "You extract product comparison data."},
                {"role": "user", "content": prompt}
            ])

            summary = response['message']['content']

            # Split the response
            if "Pros:" in summary:
                features = summary.split("Pros:")[0].strip()
                pros_cons = "Pros:" + summary.split("Pros:")[1].strip()
            else:
                features = summary
                pros_cons = "No pros/cons information available."
            return idx, features, pros_cons

        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return idx, "Error extracting features.", "Error extracting pros/cons."

    # Process products in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_product_comparison, i, p) for i, p in enumerate(products[:3])]

        for future in concurrent.futures.as_completed(futures):
            idx, features, pros_cons = future.result()
            comparison["Key Features"][idx] = features
            comparison["Pros/Cons"][idx] = pros_cons

    return comparison

def generate_recommendations(products, user_query):
    """Generate personalized product recommendations"""
    if not products or len(products) < 2:
        return "Not enough products to generate recommendations."

    # Prepare a simplified product list to reduce token usage
    simplified_products = []
    for p in products[:5]:  # Limit to top 5 for recommendations
        simplified = {
            "title": p.get("title", ""),
            "price": p.get("price", ""),
            "rating": p.get("rating", ""),
            "reviews": p.get("reviews", ""),
            "features": summarize_with_llama(p.get("content", "")) if p.get("content") else ""
        }
        simplified_products.append(simplified)

    prompt = f"""As a product recommendation expert, analyze these products for the query: "{user_query}"

Select the top 1-2 recommended products and explain why they're best for this specific query.

Products:
{json.dumps(simplified_products, indent=2)}

Format your response as a recommendation to the user:
1. Clear explanation of what makes a good choice for this query
2. Your top picks with brief reasoning for each
3. Any additional advice for this purchase"""

    try:
        response = ollama.chat(model="llama3.1", messages=[
            {"role": "system", "content": "You are a helpful recommendation engine that gives clear, decisive advice."},
            {"role": "user", "content": prompt}
        ])

        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return "Unable to generate personalized recommendations at this time."
