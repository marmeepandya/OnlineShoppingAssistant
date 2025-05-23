import streamlit as st
import asyncio
from backend import ShoppingAssistant
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
from typing import Dict, Any, List
import json

# Set page config
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    
    /* Apply to all relevant containers */
    .block-container, .main .block-container, [data-testid="stVerticalBlock"] {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
    }

    .main {
        padding: 2rem;
    }
    
    /* Set a greyish-black gradient background for the entire app */
    .main, body {
        background: linear-gradient(135deg, grey, black);
        color: #ffffff;
    }

    /* Force Streamlit's default body background to match */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, grey, black);
        color: #ffffff !important;
    }
    
    .stButton>button {
        width: 100%;
        font-size: 2.5rem;
        background-color: white;
        color: black;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, black, grey);
        color: white !important;
    }
    
    /* Increase font size of input labels */
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stNumberInput"] label,
    .stTextInput label p,
    .stTextArea label p,
    .stNumberInput label p {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: white !important;
    }


    /* Custom title styling */
    .custom-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-align: center;
        padding: 1rem 2rem;
        background: linear-gradient(135deg, black, grey);
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        display: block;
        width: fit-content;
        margin: 0 auto 2rem auto;
    }

    /* Product card styling */
    .product-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Top recommendation card styling */
    .top-recommendation {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Override Streamlit's default styles */
    .main {
        padding: 2rem;
    }

    /* Center all headings and titles */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown,
    .stAlert,
    .stSpinner {
        text-align: center !important;
    }
    
    /* Style the spinner message */
    .stSpinner > div {
        text-align: center !important;
        font-size: 1.5rem !important;
        color: white !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    /* Fix search button focus state */
    .stButton>button:focus {
        background-color: white !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Center Streamlit's default elements */
    [data-testid="stMarkdown"] {
        text-align: center !important;
    }
    
    /* Center the processed query */
    .processed-query {
        text-align: center !important;
        margin: 20px auto !important;
        max-width: 80% !important;
    }
    
    /* Center the recommendations header */
    .recommendations-header {
        text-align: center !important;
        margin: 20px auto !important;
    }
    
    /* Center the overall analysis */
    .overall-analysis {
        text-align: center !important;
        margin: 20px auto !important;
    }
    
    .product-title {
        color: white !important;
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: white !important;
        color: black !important;
        caret-color: black !important;
    }
    /* Textarea styling */
    .stTextArea>div>div>textarea,
    div[data-baseweb="textarea"] textarea {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        caret-color: black !important;
    }
    /* Number input styling */
    .stNumberInput>div>div>input,
    div[data-baseweb="input"] input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        caret-color: black !important;
    }
    /* Maintain text color on focus */
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stNumberInput>div>div>input:focus {
        color: black !important;
        border-color: #ccc !important;
        box-shadow: none !important;
    }

    /* Placeholder styling */
    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }
    .product-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    .price {
        color: #4CAF50;
        font-weight: bold;
    }
    .rating {
        color: #FFA500;
    }
    /* Center all headings and titles */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown,
    .stAlert,
    .stSpinner {
        text-align: center !important;
    }
    
    /* Style the spinner message */
    .stSpinner > div {
        text-align: center !important;
        font-size: 1.5rem !important;
        color: white !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    /* Fix search button focus state */
    .stButton>button:focus {
        background-color: white !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Center Streamlit's default elements */
    [data-testid="stMarkdown"] {
        text-align: center !important;
    }
    
    /* Center the processed query */
    .processed-query {
        text-align: center !important;
        margin: 20px auto !important;
        max-width: 80% !important;
    }
    
    /* Center the recommendations header */
    .recommendations-header {
        text-align: center !important;
        margin: 20px auto !important;
    }
    
    /* Center the overall analysis */
    .overall-analysis {
        text-align: center !important;
        margin: 20px auto !important;
    }
    
    /* Top recommendation styling */
    .top-recommendation {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    
    .product-title {
        color: white !important;
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    
    /* Ensure the container doesn't restrict width */
    [data-testid="stVerticalBlock"] {
        width: 100% !important;
        max-width: none !important;
    }

    /* Ensure the main container doesn't restrict width */
    .main .block-container {
        max-width: none !important;
        padding: 2rem 5rem !important;
    }
    </style>
    <div class="custom-title">🛍️ AI Shopping Assistant</div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = ShoppingAssistant()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def load_image(url):
    """Load image from URL with error handling"""
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        return None

def display_rating(rating: str, reviews: str) -> None:
    """Display rating and reviews in a formatted way"""
    if rating != 'N/A':
        st.markdown(f'<div class="product-rating" style="color: white;">⭐️ {rating} ({reviews} reviews)</div>', unsafe_allow_html=True)

def display_product_card(product: Dict[str, Any], compact: bool = False) -> None:
    """Display a product card with all available information"""
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .product-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .product-title {
            color: white !important;
            font-size: 1.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        .product-price {
            color: white;
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
        }
        .product-rating {
            color: white;
            margin: 5px 0;
            text-align: center;
        }
        .product-features {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .pros-cons {
            display: flex;
            gap: 20px;
            margin: 10px 0;
        }
        .pros {
            color: #2e7d32;
        }
        .cons {
            color: #c62828;
        }
        .recommendation {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .product-link {
            display: inline-block;
            background-color: white;
            color: black !important;
            padding: 8px 16px;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 10px;
            font-weight: bold;
            border: 1px solid #e0e0e0;
            text-align: center;
        }
        .product-link:hover {
            background-color: #f5f5f5;
            border-color: #bdbdbd;
        }
        .score-container {
            display: flex;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .score-item {
            background-color: #f8f9fa;
            padding: 5px 10px;
            border-radius: 5px;
            min-width: 100px;
            text-align: center;
        }
        .score-label {
            font-size: 0.8em;
            color: #666;
        }
        .score-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #1a237e;
        }
        .overall-score {
            background-color: white;
            color: black;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            margin-top: 10px;
            width: fit-content;
        }
        .overall-score-label {
            font-size: 0.8em;
            opacity: 0.9;
            color: black;
        }
        .overall-score-value {
            font-size: 1.2em;
            font-weight: bold;
            color: black;
        }
        .analysis-section {
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 5px;
            margin: 8px 0;
        }
        .analysis-title {
            font-size: 0.9em;
            font-weight: bold;
            color: #1a237e;
            margin-bottom: 5px;
        }
        .analysis-content {
            font-size: 0.9em;
            color: #333;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)

    if compact:
        col1, col2 = st.columns([1, 3])
        with col1:
            if product.get('image'):
                st.image(product['image'], width=150)
            price = product.get('price', 'N/A')
            if price != 'N/A' and not price.startswith('€'):
                price = f"€{price}"
            st.markdown(f'<div class="product-price">{price}</div>', unsafe_allow_html=True)
            display_rating(product.get('rating', 'N/A'), product.get('reviews', 'N/A'))
            if product.get('url'):
                st.markdown(f'<a href="{product["url"]}" class="product-link" target="_blank">View Product</a>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="product-title">{product["title"]}</div>', unsafe_allow_html=True)
            if 'recommendation_reason' in product:
                st.markdown(f'<div class="recommendation">{product["recommendation_reason"]}</div>', unsafe_allow_html=True)
            # Add overall score for compact view
            if 'analysis' in product and 'scores' in product['analysis']:
                overall_score = product['analysis']['scores'].get('overall_score', 5)
                st.markdown(f'''
                    <div class="overall-score">
                        <div class="overall-score-label">Overall Score</div>
                        <div class="overall-score-value">{overall_score}/10</div>
                    </div>
                ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="product-title">{product["title"]}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Left column container
            st.markdown('<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">', unsafe_allow_html=True)
            
            # Image section
            st.markdown('<div style="width: 100%; display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
            if product.get('image'):
                st.image(product['image'], width=200, use_column_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rating and reviews section
            st.markdown('<div style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-bottom: 1rem;">', unsafe_allow_html=True)
            price = product.get('price', 'N/A')
            if price != 'N/A' and not price.startswith('€'):
                price = f"€{price}"
            st.markdown(f'<div style="font-size: 1.2rem; color: white; font-weight: bold; margin-bottom: 0.5rem;">{price}</div>', unsafe_allow_html=True)
            display_rating(product.get('rating', 'N/A'), product.get('reviews', 'N/A'))
            if product.get('url'):
                st.markdown(f'<a href="{product["url"]}" class="product-link" target="_blank" style="margin-top: 0.5rem;">View Product</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Overall score section
            if 'analysis' in product and 'scores' in product['analysis']:
                overall_score = product['analysis']['scores'].get('overall_score', 5)
                st.markdown(f'''
                    <div style="display: flex; justify-content: center; width: 100%;">
                        <div style="
                            background-color: white;
                            color: black;
                            padding: 8px;
                            border-radius: 5px;
                            text-align: center;
                            margin-top: 10px;
                            width: fit-content;
                        ">
                            <div style="font-size: 0.8em; opacity: 0.9; color: black;">Overall Score</div>
                            <div style="font-size: 1.2em; font-weight: bold; color: black;">{overall_score}/10</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # End left column container
        
        with col2:
            if 'analysis' in product:
                analysis = product['analysis']
                if isinstance(analysis, dict):
                    if 'key_features' in analysis:
                        st.markdown('<div class="product-features">', unsafe_allow_html=True)
                        st.markdown("**Key Features:**")
                        st.markdown(analysis['key_features'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="pros-cons">', unsafe_allow_html=True)
                    if 'pros' in analysis:
                        st.markdown('<div class="pros">', unsafe_allow_html=True)
                        st.markdown("**Pros:**")
                        st.markdown(analysis['pros'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    if 'cons' in analysis:
                        st.markdown('<div class="cons">', unsafe_allow_html=True)
                        st.markdown("**Cons:**")
                        st.markdown(analysis['cons'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display all scores in a compact format after pros/cons
                    if 'scores' in analysis:
                        scores = analysis['scores']
                        st.markdown('<div class="score-container">', unsafe_allow_html=True)
                        for score_type, score_value in scores.items():
                            if score_type != 'overall_score':  # Skip overall score here
                                score_label = score_type.replace('_', ' ').title()
                                st.markdown(f'''
                                    <div class="score-item">
                                        <div class="score-label">{score_label}</div>
                                        <div class="score-value">{score_value}/10</div>
                                    </div>
                                ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display detailed analyses
                    if 'analysis' in analysis:
                        analyses = analysis['analysis']
                        # Performance Analysis
                        if 'performance_analysis' in analyses:
                            st.markdown(f'''
                                <div class="analysis-section">
                                    <div class="analysis-title">Performance Analysis</div>
                                    <div class="analysis-content">{analyses['performance_analysis']}</div>
                                </div>
                            ''', unsafe_allow_html=True)
                        
                        # Value Analysis
                        if 'value_analysis' in analyses:
                            st.markdown(f'''
                                <div class="analysis-section">
                                    <div class="analysis-title">Value Analysis</div>
                                    <div class="analysis-content">{analyses['value_analysis']}</div>
                                </div>
                            ''', unsafe_allow_html=True)
                        
                        # Requirements Match
                        if 'requirements_match' in analyses:
                            st.markdown(f'''
                                <div class="analysis-section">
                                    <div class="analysis-title">Requirements Match</div>
                                    <div class="analysis-content">{analyses['requirements_match']}</div>
                                </div>
                            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_recommendations(recommendations: List[Dict[str, Any]], ranked_products: List[Dict[str, Any]], recommendations_analysis: str) -> None:
    """Display recommendations with analysis"""
    # Display the recommendations header with increased size
    st.markdown('<div class="recommendations-header" style="font-size: 2.5rem; font-weight: bold;">🌟 Top Recommendations</div>', unsafe_allow_html=True)
    
    # Display each recommendation
    for product in recommendations:
        st.markdown('<div class="top-recommendation">', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-title" style="font-size: 1.8rem; margin-bottom: 1rem; text-align: center; width: 100%;">{product["title"]}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Left column container
            st.markdown('<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">', unsafe_allow_html=True)
            
            # Image section
            st.markdown('<div style="width: 100%; display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
            if product.get('image'):
                st.image(product['image'], width=200, use_column_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rating and reviews section
            st.markdown('<div style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-bottom: 1rem;">', unsafe_allow_html=True)
            price = product.get('price', 'N/A')
            if price != 'N/A' and not price.startswith('€'):
                price = f"€{price}"
            st.markdown(f'<div style="font-size: 1.2rem; color: white; font-weight: bold; margin-bottom: 0.5rem;">{price}</div>', unsafe_allow_html=True)
            display_rating(product.get('rating', 'N/A'), product.get('reviews', 'N/A'))
            if product.get('url'):
                st.markdown(f'''
                    <a href="{product["url"]}" class="product-link" target="_blank" style="
                        display: inline-block;
                        background-color: white;
                        color: black !important;
                        padding: 8px 16px;
                        border-radius: 5px;
                        text-decoration: none;
                        margin-top: 0.5rem;
                        font-weight: bold;
                        border: 1px solid #e0e0e0;
                        text-align: center;
                    ">View Product</a>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Overall score section
            if 'analysis' in product and 'scores' in product['analysis']:
                overall_score = product['analysis']['scores'].get('overall_score', 5)
                st.markdown(f'''
                    <div style="display: flex; justify-content: center; width: 100%;">
                        <div style="
                            background-color: white;
                            color: black;
                            padding: 8px;
                            border-radius: 5px;
                            text-align: center;
                            margin-top: 10px;
                            width: fit-content;
                        ">
                            <div style="font-size: 0.8em; opacity: 0.9; color: black;">Overall Score</div>
                            <div style="font-size: 1.2em; font-weight: bold; color: black;">{overall_score}/10</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # End left column container
        
        with col2:
            # Container for recommendation reason with vertical centering
            st.markdown('<div style="display: flex; align-items: center; height: 100%; min-height: 200px;">', unsafe_allow_html=True)
            if 'recommendation_reason' in product:
                st.markdown(f'''
                    <div class="recommendation-reason" style="
                        font-size: 1.1rem;
                        line-height: 1.6;
                        color: white;
                        padding: 1rem;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 8px;
                        width: 100%;
                    ">{product["recommendation_reason"]}</div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display overall analysis
    if recommendations_analysis:
        st.markdown('<div class="overall-analysis">', unsafe_allow_html=True)
        st.markdown("### Our Analysis")
        # Extract and clean up the analysis text
        if "Overall Analysis:" in recommendations_analysis:
            analysis_text = recommendations_analysis.split("Overall Analysis:")[1].strip()
        else:
            analysis_text = recommendations_analysis
            
        # Clean up the text
        analysis_text = analysis_text.replace("within the specified maximum budget", "within your budget")
        analysis_text = analysis_text.replace("with a maximum budget of ", "")
        analysis_text = analysis_text.replace("euros.", "")
        analysis_text = analysis_text.replace("**", "")
        
        # Remove any lines that start with "Note"
        analysis_text = '\n'.join(line for line in analysis_text.split('\n') if not line.strip().startswith('Note'))
        
        st.markdown(f'<div class="analysis-text">{analysis_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add section for all products with increased size
    st.markdown("---")
    st.markdown('<div class="recommendations-header" style="font-size: 2.5rem; font-weight: bold;">📋 All Products</div>', unsafe_allow_html=True)
    
    # Display all products
    for product in ranked_products:
        display_product_card(product)
        st.markdown("---")

def main():
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        query = st.text_input("🔍 What are you looking for?", placeholder="e.g., Laptop, Smartphone, etc.")
        max_price = st.number_input("💰 Maximum Price (€)", min_value=0, value=0, step=100)
    
    with col2:
        additional_requirements = st.text_area(
            "📌 Additional Requirements",
            placeholder="Enter any specific features or requirements you're looking for (e.g., '16GB RAM, NVIDIA graphics')",
            help="Specify any particular features, specifications, or requirements you want in your product.",
            height=121  
        )
    
    # Center the search button using CSS
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100px;
            margin: 0 auto;
            display: block;
            padding: 0.3rem 1rem;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("Search"):
        if query:
            with st.spinner("✨ Finding the best products for you..."):
                # Initialize the shopping assistant
                assistant = ShoppingAssistant()
                
                # Process the query
                results = asyncio.run(assistant.process_shopping_query(
                    query=query,
                    max_price=max_price,
                    additional_requirements=additional_requirements
                ))
                
                # Store results in session state
                st.session_state.results = results
                
                # Display recommendations
                if results['recommendations']:
                    display_recommendations(
                        recommendations=results['recommendations'],
                        ranked_products=results['ranked_products'],
                        recommendations_analysis=results['recommendations_analysis']
                    )
                else:
                    st.warning("No recommendations found. Try adjusting your search criteria.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main() 