import os
import re
import json
import ollama
import time
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from tavily import TavilyClient
from dotenv import load_dotenv
from cachetools import cached, TTLCache
from langchain_community.tools.tavily_search import TavilySearchResults
from serpapi import GoogleSearch
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

class ProductState(TypedDict):
    """State for the shopping assistant workflow"""
    query: str
    max_price: Optional[float]
    additional_requirements: str  
    products: List[Dict[str, Any]]
    processed_query: Dict[str, str]
    detailed_products: List[Dict[str, Any]]
    ranked_products: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    recommendations_analysis: str
    status: Dict[str, str]

class ShoppingGraph:
    def __init__(self):
        self.graph = self._build_graph()
        self.product_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
        
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow"""
        # Define the nodes
        workflow = Graph()
        
        # Add nodes for each step in the workflow
        workflow.add_node("process_query", self._process_query_node)
        workflow.add_node("search_products", self._search_products_node)
        workflow.add_node("extract_specifications", self._extract_specifications_node)
        workflow.add_node("rank_products", self._rank_products_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
                
        # Define the edges
        workflow.add_edge("process_query", "search_products")
        workflow.add_edge("search_products", "extract_specifications")
        workflow.add_edge("extract_specifications", "rank_products")
        workflow.add_edge("rank_products", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)  # Add direct edge to end
        
        # Set the entry point
        workflow.set_entry_point("process_query")
        
        return workflow.compile()
    
    async def _process_query_node(self, state: ProductState) -> ProductState:
        """Process and restructure the user query with additional requirements"""
        try:
            # Create the prompt
            prompt = f"""You are an AI assistant that restructures user queries for product searches. 
            Your task is to create a detailed search query that includes ALL requirements and price limit.
            
            Basic Query: {state['query']}
            Max Price: {state['max_price']} euros
            Additional Requirements: {state['additional_requirements']}
            
            Please provide your response in the following format:
            Translated: [translated query if needed]
            Restructured: [restructured query incorporating additional requirements and price limit]
            
            CRITICAL RULES:
            1. ALWAYS include the price limit in the restructured query
            2. ALWAYS include ALL additional requirements in the restructured query
            3. Make the query specific and search-friendly
            4. Keep the query concise but informative
            5. For technical specifications (RAM, storage, etc.), add them AFTER the main product
            6. For product type modifiers (Gaming, Professional, etc.), add them BEFORE the main product
            7. Use natural language that search engines understand
            8. Make sure that there are no repetitive phrases
            
            Examples:
            Query: Laptop
            Max Price: 1000 euros
            Additional Requirements: 16GB RAM, Gaming
            Restructured: Gaming Laptop with 16GB RAM under 1000 euros
            
            Query: Laptop
            Max Price: 4000 euros
            Additional Requirements: Gaming
            Restructured: Gaming Laptop under 4000 euros
            
            Query: Washing Machine
            Max Price: 800 euros
            Additional Requirements: 
            Restructured: Washing Machine under 800 euros
            
            Query: Smartphone
            Max Price: 500 euros
            Additional Requirements: 5G, 128GB
            Restructured: 5G Smartphone with 128GB storage under 500 euros
            
            Query: Monitor
            Max Price: 300 euros
            Additional Requirements: 27 inch, 4K
            Restructured: 27 inch 4K Monitor under 300 euros
            
            Query: Headphones
            Max Price: 200 euros
            Additional Requirements: Wireless, Noise Cancelling
            Restructured: Wireless Noise Cancelling Headphones under 200 euros
            
            Query: Camera
            Max Price: 1000 euros
            Additional Requirements: DSLR, 24MP
            Restructured: DSLR Camera with 24MP under 1000 euros
            
            Query: Printer
            Max Price: 150 euros
            Additional Requirements: Wireless, Color
            Restructured: Wireless Color Printer under 150 euros"""
            
            # Call Ollama directly
            response = ollama.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            result = response['message']['content']
            
            # Extract translated and restructured queries
            translated = re.search(r"^Translated: (.+)$", result, re.MULTILINE)
            restructured = re.search(r"^Restructured: (.+)$", result, re.MULTILINE)
            
            # Ensure price is included in restructured query
            final_restructured = restructured.group(1) if restructured else state["query"]
            
            # Ensure additional requirements are included
            if state["additional_requirements"] and state["additional_requirements"].strip():
                requirements = state["additional_requirements"].strip()
                if requirements.lower() not in final_restructured.lower():
                    # Add requirements at the beginning of the query
                    final_restructured = f"{requirements} {final_restructured}"
            
            # Ensure price is included
            if state["max_price"] and f"{state['max_price']} euros" not in final_restructured.lower():
                final_restructured = f"{final_restructured} under {state['max_price']} euros"
            
            state["processed_query"] = {
                "translated": translated.group(1) if translated else state["query"],
                "restructured": final_restructured,
                "original_requirements": state["additional_requirements"]
            }
            
            state["status"] = {
                "process_query": "Completed",
                "search_products": "Pending",
                "extract_specifications": "Pending",
                "rank_products": "Pending",
                "generate_recommendations": "Pending"
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error in process_query_node: {e}")
            state["processed_query"] = {
                "translated": state["query"],
                "restructured": state["query"],
                "original_requirements": state["additional_requirements"]
            }
            state["status"] = {
                "process_query": f"Failed: {str(e)}",
                "search_products": "Pending",
                "extract_specifications": "Pending",
                "rank_products": "Pending",
                "generate_recommendations": "Pending"
            }
            return state
    
    async def _search_products_node(self, state: ProductState) -> ProductState:
        """Search for products using SerpAPI"""
        try:
            params = {
                "api_key": serpapi_key,
                "engine": "google_shopping",
                "q": state["processed_query"]["restructured"],
                "num": 20,
                "location": "Germany",
                "gl": "de",
                "hl": "en",
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            product_results = results.get("shopping_results", [])[:20]

            products = []
            for r in product_results:
                # Get price directly without conversion
                price = r.get('extracted_price', '')
                if price and isinstance(price, (int, float)):
                    price = f"€{price:.2f}"
                else:
                    price = f"€{price}" if price else 'N/A'

                # Format reviews to preserve exact number
                reviews = r.get('reviews', '')
                if reviews and isinstance(reviews, (int, float)):
                    reviews = str(int(reviews))  # Convert to integer and then string to remove decimal places
                elif not reviews:
                    reviews = 'N/A'

                product = {
                    "product_id": r.get('product_id', ''),
                    "title": r.get('title', ''),
                    "url": r.get('product_link', ''),
                    "source": r.get('source', ''),
                    "price": price,
                    "old_price": r.get('extracted_old_price', ''),
                    "rating": r.get('rating', ''),
                    "reviews": reviews,
                    "extensions": r.get('extensions', []),
                    "image": r.get('thumbnail', ''),
                    "processed_query": state["processed_query"]  # Add the processed query to each product
                }
                products.append(product)
            
            state["products"] = products
            state["status"]["search_products"] = f"Completed: Found {len(products)} products"
            state["status"]["extract_specifications"] = "Pending"
            
            return state

        except Exception as e:
            logger.error(f"Error in search: {e}")
            state["products"] = []
            state["status"]["search_products"] = f"Failed: {str(e)}"
            state["status"]["extract_specifications"] = "Pending"
            return state
    
    async def _extract_specifications_node(self, state: ProductState) -> ProductState:
        """Extract and structure product specifications using Tavily and LLM"""
        tool = TavilySearchResults(
            max_results=2,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
        )
        
        detailed_products = []
        for product in state["products"]:
            try:
                # First search for general product information
                search_query = f"{product['title']} product technical description details specifications features pros cons"
                details = tool.invoke(search_query)
                
                content = ""
                if details and isinstance(details, list):
                    for item in details:
                        if isinstance(item, dict) and 'content' in item:
                            content += item['content'] + '\n'
                        elif isinstance(item, str):
                            content += item + '\n'
                
                if not content:
                    content = "No details found."
                
                # Use LLM to structure and summarize the details
                prompt = f"""You are a product analysis expert. Analyze and structure the following product details into a clear, organized format.
                Focus on key specifications, features, and important information.
                
                Product: {product['title']}
                Price: {product.get('price', 'N/A')}
                Rating: {product.get('rating', 'N/A')}
                Reviews: {product.get('reviews', 'N/A')}
                
                Raw Details: {content}
                
                You MUST respond with a valid JSON object in this exact format:
                {{
                    "key_features": [
                        "feature 1",
                        "feature 2",
                        "feature 3"
                    ],
                    "pros": [
                        "pro 1",
                        "pro 2",
                        "pro 3"
                    ],
                    "cons": [
                        "con 1",
                        "con 2",
                        "con 3"
                    ],
                    "summary": "Brief overall summary of the product's value proposition"
                }}
                
                CRITICAL RULES:
                1. Your response MUST be a valid JSON object
                2. Do not include any text before or after the JSON object
                3. Use double quotes for all strings
                4. Provide at least 3 items in each list
                5. Use specific, detailed information
                6. Focus on concrete features and specifications
                7. Do not include any markdown formatting
                8. Do not include any explanatory text
                """
                
                response = ollama.chat(model='llama3.1', messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ])
                
                try:
                    # Clean the response to ensure it's valid JSON
                    content = response['message']['content'].strip()
                    # Remove any markdown code block markers
                    content = content.replace('```json', '').replace('```', '').strip()
                    
                    # Try to fix common JSON formatting issues
                    content = content.replace("'", '"')  # Replace single quotes with double quotes
                    content = re.sub(r'(\w+):', r'"\1":', content)  # Add quotes to keys
                    
                    # Parse the JSON response
                    try:
                        structured_details = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Initial JSON parsing failed for product {product.get('title')}, attempting to fix format")
                        # Try to extract JSON-like structure using regex
                        key_features_match = re.search(r'"key_features"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                        pros_match = re.search(r'"pros"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                        cons_match = re.search(r'"cons"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                        summary_match = re.search(r'"summary"\s*:\s*"(.*?)"', content, re.DOTALL)
                        
                        # Create structured details from matches
                        structured_details = {
                            'key_features': [f.strip().strip('"\'') for f in key_features_match.group(1).split(',')] if key_features_match else ['No key features found'],
                            'pros': [p.strip().strip('"\'') for p in pros_match.group(1).split(',')] if pros_match else ['No pros found'],
                            'cons': [c.strip().strip('"\'') for c in cons_match.group(1).split(',')] if cons_match else ['No cons found'],
                            'summary': summary_match.group(1) if summary_match else 'No summary available'
                        }
                    
                    # Validate the structure
                    required_fields = ['key_features', 'pros', 'cons', 'summary']
                    for field in required_fields:
                        if field not in structured_details:
                            structured_details[field] = [] if field != 'summary' else 'No summary available'
                        elif field != 'summary' and not isinstance(structured_details[field], list):
                            structured_details[field] = [str(structured_details[field])]
                    
                    # Format the sections for display
                    formatted_details = {
                        'key_features': '\n'.join(f"- {f}" for f in structured_details.get('key_features', [])) or "No key features found",
                        'pros': '\n'.join(f"- {p}" for p in structured_details.get('pros', [])) or "No pros found",
                        'cons': '\n'.join(f"- {c}" for c in structured_details.get('cons', [])) or "No cons found",
                        'summary': structured_details.get('summary', 'No summary available')
                    }
                except Exception as e:
                    logger.error(f"Error parsing response for product {product.get('title')}: {e}")
                    # Create a default structured response
                    structured_details = {
                        'key_features': ['No key features found'],
                        'pros': ['No pros found'],
                        'cons': ['No cons found'],
                        'summary': 'No summary available'
                    }
                    formatted_details = {
                        'key_features': "No key features found",
                        'pros': "No pros found",
                        'cons': "No cons found",
                        'summary': "No summary available"
                    }
                    
                detailed_product = {
                    **product,
                    "raw_details": content,
                    "structured_details": structured_details,
                    "formatted_details": formatted_details
                }
                detailed_products.append(detailed_product)
                    
            except Exception as e:
                logger.error(f"Error extracting specifications for product {product.get('title')}: {e}")
                detailed_products.append({
                    **product,
                    "raw_details": "No details found.",
                    "structured_details": "No structured details available.",
                    "formatted_details": {
                        'key_features': "No key features found",
                        'pros': "No pros found",
                        'cons': "No cons found",
                        'summary': "No summary available"
                    }
                })
        
        state["detailed_products"] = detailed_products
        state["status"]["extract_specifications"] = f"Completed: Extracted and structured details for {len(detailed_products)} products"
        state["status"]["rank_products"] = "Pending"
        return state
    
    async def _rank_products_node(self, state: ProductState) -> ProductState:
        """Rank products based on LLM analysis of their details and user requirements"""
        try:
            # Process products in batches of 5
            batch_size = 5
            all_ranked_products = []
            
            for i in range(0, len(state["detailed_products"]), batch_size):
                batch_products = state["detailed_products"][i:i + batch_size]
                
                # Create a prompt for analyzing the batch of products
                prompt = f"""You are a product analysis expert. Analyze and rank these products based on multiple criteria.
                Consider the user's requirements and provide a comprehensive analysis with detailed scoring.
                
                User Requirements:
                - Basic Query: {state['query']}
                - Max Price: {state['max_price']} euros
                - Additional Requirements: {state['additional_requirements']}
                
                Products to Analyze:
                {json.dumps([{
                    'title': p['title'],
                    'price': p.get('price', 'N/A'),
                    'rating': p.get('rating', 'N/A'),
                    'reviews': p.get('reviews', 'N/A'),
                    'structured_details': p.get('structured_details', '')
                } for p in batch_products], indent=2)}
                
                You MUST respond with a valid JSON object in this exact format:
                {{
                    "products": [
                        {{
                            "title": "exact product title",
                            "price": "price",
                            "scores": {{
                                "performance": 1-10,
                                "value_for_money": 1-10,
                                "matching_requirements": 1-10,
                                "overall_score": 1-10
                            }},
                            "analysis": {{
                                "performance_analysis": "Detailed analysis of product performance based on specs and features",
                                "value_analysis": "Analysis of price vs features and quality",
                                "requirements_match": "How well it matches user requirements",
                                "why_recommended": "Overall recommendation reason"
                            }}
                        }},
                        ...
                    ],
                    "overall_analysis": "Brief analysis comparing the products and explaining the rankings"
                }}
                
                CRITICAL RULES:
                1. Your response MUST be a valid JSON object
                2. Do not include any text before or after the JSON object
                3. Use double quotes for all strings
                4. Include all products in the analysis
                5. Provide specific, detailed explanations for each score
                6. Consider the following for scoring:
                   - Performance: Based on specifications, features, and capabilities
                   - Value for Money: Price vs features, quality, and market comparison
                   - Matching Requirements: How well it meets user's specific needs
                   - Overall Score: Weighted combination of all factors
                7. Scores must be between 1-10 (whole numbers)
                8. Provide detailed analysis for each scoring category
                9. Do not include any markdown formatting
                10. Do not include any explanatory text
                11. IMPORTANT: When evaluating ratings:
                    - If a product has less than 10 reviews, give minimal weight to its rating
                    - If a product has 10-50 reviews, give moderate weight to its rating
                    - If a product has more than 50 reviews, give full weight to its rating
                    - Products with no reviews should be evaluated based on their specifications and features only
                12. Always mention the number of reviews in your analysis when discussing ratings
                """
                
                # Get LLM's analysis for this batch
                response = ollama.chat(model='llama3.1', messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ])
                
                try:
                    # Clean the response to ensure it's valid JSON
                    content = response['message']['content'].strip()
                    # Remove any markdown code block markers
                    content = content.replace('```json', '').replace('```', '').strip()
                    
                    # Parse the JSON response
                    analysis_data = json.loads(content)
                    
                    # Process the analysis to extract product rankings
                    for product_analysis in analysis_data.get('products', []):
                        # Validate required fields
                        if 'title' not in product_analysis:
                            continue
                        
                        # Ensure all required fields exist with defaults
                        scores = product_analysis.get('scores', {})
                        analysis = product_analysis.get('analysis', {})
                        
                        # Set default scores if missing
                        default_scores = {
                            'performance': 5,
                            'value_for_money': 5,
                            'matching_requirements': 5,
                            'overall_score': 5
                        }
                        for key in default_scores:
                            if key not in scores:
                                scores[key] = default_scores[key]
                        
                        # Set default analysis if missing
                        default_analysis = {
                            'performance_analysis': 'No performance analysis available',
                            'value_analysis': 'No value analysis available',
                            'requirements_match': 'No requirements match analysis available',
                            'why_recommended': 'No recommendation reason provided'
                        }
                        for key in default_analysis:
                            if key not in analysis:
                                analysis[key] = default_analysis[key]
                        
                        # Find the matching product
                        matching_product = next(
                            (p for p in batch_products 
                             if p['title'].lower() in product_analysis['title'].lower() or 
                             product_analysis['title'].lower() in p['title'].lower()),
                            None
                        )
                        
                        if matching_product:
                            # Get the formatted details from extract_specifications_node
                            formatted_details = matching_product.get('formatted_details', {})
                            
                            # Create the analysis object
                            product_analysis = {
                                'key_features': formatted_details.get('key_features', 'No key features found'),
                                'pros': formatted_details.get('pros', 'No pros found'),
                                'cons': formatted_details.get('cons', 'No cons found'),
                                'scores': scores,
                                'analysis': analysis,
                                'price': matching_product.get('price', 'N/A')
                            }
                            
                            # Add the analysis to the product
                            all_ranked_products.append({
                                **matching_product,
                                "analysis": product_analysis
                            })
                
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response in rank_products_node for batch {i//batch_size + 1}: {e}")
                    # Create a basic analysis for products in this batch
                    for product in batch_products:
                        formatted_details = product.get('formatted_details', {})
                        basic_analysis = {
                            'key_features': formatted_details.get('key_features', 'No key features found'),
                            'pros': formatted_details.get('pros', 'No pros found'),
                            'cons': formatted_details.get('cons', 'No cons found'),
                            'scores': {
                                'performance': 5,
                                'value_for_money': 5,
                                'matching_requirements': 5,
                                'overall_score': 5
                            },
                            'analysis': {
                                'performance_analysis': 'No performance analysis available',
                                'value_analysis': 'No value analysis available',
                                'requirements_match': 'No requirements match analysis available',
                                'why_recommended': 'No recommendation reason provided'
                            },
                            'price': product.get('price', 'N/A')
                        }
                        all_ranked_products.append({
                            **product,
                            "analysis": basic_analysis
                        })
            
            # Add any remaining products that weren't analyzed
            analyzed_titles = {p['title'].lower() for p in all_ranked_products}
            for product in state["detailed_products"]:
                if product['title'].lower() not in analyzed_titles:
                    # Create a basic analysis for unanalyzed products
                    formatted_details = product.get('formatted_details', {})
                    basic_analysis = {
                        'key_features': formatted_details.get('key_features', 'No key features found'),
                        'pros': formatted_details.get('pros', 'No pros found'),
                        'cons': formatted_details.get('cons', 'No cons found'),
                        'scores': {
                            'performance': 5,
                            'value_for_money': 5,
                            'matching_requirements': 5,
                            'overall_score': 5
                        },
                        'analysis': {
                            'performance_analysis': 'No performance analysis available',
                            'value_analysis': 'No value analysis available',
                            'requirements_match': 'No requirements match analysis available',
                            'why_recommended': 'No recommendation reason provided'
                        },
                        'price': product.get('price', 'N/A')
                    }
                    all_ranked_products.append({
                        **product,
                        "analysis": basic_analysis
                    })
            
            # Sort products by overall score
            all_ranked_products.sort(key=lambda x: x.get('analysis', {}).get('scores', {}).get('overall_score', 0), reverse=True)
            
            # Store all ranked products but only return top 10
            state["ranked_products"] = all_ranked_products[:10]
            state["status"]["rank_products"] = f"Completed: Ranked {len(all_ranked_products)} products, displaying top 10"
            state["status"]["generate_recommendations"] = "Pending"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in rank_products_node: {e}")
            state["ranked_products"] = state["detailed_products"][:10]  # Limit to top 10 even in case of error
            state["status"]["rank_products"] = f"Failed: {str(e)}"
            state["status"]["generate_recommendations"] = "Pending"
            return state
    
    async def _generate_recommendations_node(self, state: ProductState) -> ProductState:
        """Generate personalized product recommendations using LLM"""
        try:
            recommendations = []
            products = state["ranked_products"]  # Already limited to top 10
            
            # Create a prompt for personalized recommendations
            prompt = f"""You are a product analysis expert. Analyze these products and give a detailed explanation on why this product is recommended.
            Consider the user's requirements and provide a comprehensive analysis.
            Consider the user's query: {state['query']}
            Max Price: {state['max_price']} euros
            
            Ranked Products:
            {json.dumps([{
                'title': p['title'],
                'price': p.get('price', 'N/A'),
                'rating': p.get('rating', 'N/A'),
                'reviews': p.get('reviews', 'N/A'),
                'analysis': p.get('analysis', '')
            } for p in products[:3]], indent=2)}
            
            Please provide recommendations in the following format:
            
            Top Recommendations:
            
            [product name]
            Why Recommended: [detailed explanation of why this product is recommended]
            
            [product name]
            Why Recommended: [detailed explanation of why this product is recommended]
            
            [product name]
            Why Recommended: [detailed explanation of why this product is recommended]
            
            Overall Analysis:
            [Brief analysis regarding why these 3 products are recommended as top products]
            
            
            CRITICAL RULES:
            1. Don't include the ratings given during the ranking process.
            2. Don't give responses such as "Same as the above", or something similar. Make sure that you provide explanation to each product, individually.
            3. You must provide a detailed explaination based on the information you have regarding the product. """
            
            response = ollama.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            recommendations_text = response['message']['content']
            
            # Extract recommended products
            recommended_products = []
            for product in products[:3]:  # Only take top 3
                recommendation_reason = self._extract_recommendation_reason(recommendations_text, product['title'])
                recommended_products.append({
                    **product,
                    "recommendation_reason": recommendation_reason if recommendation_reason else "No specific reasoning found"
                })
            
            state["recommendations"] = recommended_products
            state["recommendations_analysis"] = recommendations_text
            state["status"]["generate_recommendations"] = f"Completed: Generated {len(recommended_products)} personalized recommendations"
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_recommendations_node: {e}")
            state["recommendations"] = []
            state["recommendations_analysis"] = "Failed to generate recommendations"
            state["status"]["generate_recommendations"] = f"Failed: {str(e)}"
            return state
    
    def _extract_recommendation_reason(self, recommendations_text: str, product_title: str) -> str:
        """Extract the reasoning for a specific product recommendation"""
        try:
            # Find the section containing the product title
            lines = recommendations_text.split('\n')
            for i, line in enumerate(lines):
                if product_title.lower() in line.lower():
                    # Look for the next line starting with "Why Recommended:"
                    for j in range(i, min(i + 5, len(lines))):
                        if lines[j].strip().startswith("Why Recommended:"):
                            return lines[j].replace("Why Recommended:", "").strip()
            return "No specific reasoning found"
        except Exception:
            return "No specific reasoning found"

    def _should_end(self, state: ProductState) -> bool:
        """Determine if the workflow should end"""
        return True  # Always end after generating recommendations

class ShoppingAssistant:
    def __init__(self):
        self.graph = ShoppingGraph()
    
    async def process_shopping_query(self, query: str, max_price: Optional[float] = None, additional_requirements: str = "") -> Dict[str, Any]:
        """Process a shopping query through the entire workflow"""
        initial_state = ProductState(
            query=query,
            max_price=max_price,
            additional_requirements=additional_requirements,
            products=[],
            processed_query={},
            detailed_products=[],
            ranked_products=[],
            recommendations=[],
            recommendations_analysis="",
            status={}
        )
        
        try:
            # Execute the workflow
            final_state = await self.graph.graph.ainvoke(initial_state)
            
            if final_state is None:
                logger.error("Graph execution returned None")
                return {
                    "processed_query": {
                        "translated": query,
                        "restructured": query,
                        "original_requirements": additional_requirements
                    },
                    "products": [],
                    "detailed_products": [],
                    "ranked_products": [],
                    "recommendations": [],
                    "recommendations_analysis": "Failed to generate recommendations",
                    "status": {
                        "process_query": "Failed: Graph execution returned None",
                        "search_products": "Not started",
                        "extract_specifications": "Not started",
                        "rank_products": "Not started",
                        "generate_recommendations": "Not started"
                    }
                }
            
            # Ensure we have a valid state
            if not isinstance(final_state, dict):
                logger.error(f"Invalid state type: {type(final_state)}")
                return initial_state
            
            # Save results to CSV
            save_to_csv(
                query=query,
                max_price=max_price,
                additional_requirements=additional_requirements,
                raw_products=final_state.get("products", []),
                ranked_products=final_state.get("ranked_products", []),
                recommendations=final_state.get("recommendations", [])
            )
            
            return {
                "processed_query": final_state.get("processed_query", {
                    "translated": query,
                    "restructured": query,
                    "original_requirements": additional_requirements
                }),
                "products": final_state.get("products", []),
                "detailed_products": final_state.get("detailed_products", []),
                "ranked_products": final_state.get("ranked_products", []),
                "recommendations": final_state.get("recommendations", []),
                "recommendations_analysis": final_state.get("recommendations_analysis", ""),
                "status": final_state.get("status", {})
            }
        except Exception as e:
            logger.error(f"Error in process_shopping_query: {e}")
            return {
                "processed_query": {
                    "translated": query,
                    "restructured": query,
                    "original_requirements": additional_requirements
                },
                "products": [],
                "detailed_products": [],
                "ranked_products": [],
                "recommendations": [],
                "recommendations_analysis": "Failed to generate recommendations",
                "status": {
                    "process_query": f"Failed: {str(e)}",
                    "search_products": "Not started",
                    "extract_specifications": "Not started",
                    "rank_products": "Not started",
                    "generate_recommendations": "Not started"
                }
            }

def save_to_csv(query: str, max_price: float, additional_requirements: str, 
                raw_products: List[Dict], ranked_products: List[Dict], 
                recommendations: List[Dict]) -> None:
    """Save search results and rankings to a CSV file"""
    try:
        # Create a filename from the query
        # Replace spaces and special characters with underscores
        safe_query = re.sub(r'[^a-zA-Z0-9\s-]', '', query)
        safe_query = safe_query.replace(' ', '_').lower()
        filename = f"shopping_results_{safe_query}.csv"
        
        # Get the restructured query from the first product if available
        restructured_query = query
        if raw_products and 'processed_query' in raw_products[0]:
            restructured_query = raw_products[0]['processed_query'].get('restructured', query)
        
        # Prepare data for raw products (all SerpAPI products)
        raw_data = []
        for product in raw_products:
            raw_data.append({
                'query': restructured_query,
                'max_price': max_price,
                'additional_requirements': additional_requirements,
                'product_type': 'raw',
                'title': product.get('title', ''),
                'url': product.get('url', '')
            })
        
        # Prepare data for top 10 ranked products
        ranked_data = []
        # Ensure we're using the sorted ranked products (they should already be sorted by overall score)
        for i, product in enumerate(ranked_products[:10], 1):  # Only take top 10 ranked products
            ranked_data.append({
                'query': restructured_query,
                'max_price': max_price,
                'additional_requirements': additional_requirements,
                'product_type': f'ranked_{i}',  # Add rank number to product_type
                'title': product.get('title', ''),
                'url': product.get('url', '')
            })
        
        # Combine data in the desired order: raw products first, then ranked products
        all_data = raw_data + ranked_data
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

# Export the ShoppingAssistant class
__all__ = ['ShoppingAssistant'] 