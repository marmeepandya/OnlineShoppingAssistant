import streamlit as st
from backend import (
    process_query,
    search_serpapi,
    extract_specifications,
    summarize_with_llama,
    sort_results,
    generate_comparison_table,
    generate_recommendations,
    rank_products
)
import time
import json

# --- Theme Configuration ---
st.set_page_config(
    page_title="Online Shopping Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom theme
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #0EA5E9;
            --secondary-color: #06B6D4;
            --background-color: #F0F9FF;
            --text-color: #0F172A;
            --text-light: #334155;
            --placeholder-color: #64748B;
            --button-text: #FFFFFF;
            --input-bg: #FFFFFF;
            --card-bg: rgba(255, 255, 255, 0.9);
        }
        
        /* Global styles */
        .stApp {
            background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 50%, #7DD3FC 100%);
            background-attachment: fixed;
            color: var(--text-color);
        }
        
        /* Header styling */
        header[data-testid="stHeader"] {
            background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 50%, #7DD3FC 100%);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Header buttons */
        header[data-testid="stHeader"] button {
            background: transparent !important;
            border: none !important;
            color: var(--text-color) !important;
            backdrop-filter: blur(5px);
        }
        
        header[data-testid="stHeader"] button:hover {
            background: rgba(255, 255, 255, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
        }
        
        /* Main content area */
        .main .block-container {
            background: var(--card-bg);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin: 1rem;
            backdrop-filter: blur(10px);
            color: var(--text-color);
        }
        
        /* Text colors for all elements */
        .stText, .stMarkdown, .stProgress > div > div > div, 
        div[data-testid="stText"], 
        div[data-testid="stMarkdown"],
        .stMarkdown p {
            color: black !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--input-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: black;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #333333;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: black !important;
            border-radius: 0.375rem;
        }
        
        .stProgress > div > div > div {
            background: black !important;
        }
        
        /* All borders */
        .stExpander, 
        .stExpander > div,
        .stExpander > div > div,
        .stExpander > div > div > div,
        hr,
        .stMarkdown hr,
        .stDivider,
        .streamlit-expanderHeader,
        .streamlit-expanderContent {
            border-color: black !important;
            border-top-color: black !important;
            border-bottom-color: black !important;
            border-left-color: black !important;
            border-right-color: black !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            border: 1px solid black !important;
            border-radius: 0.375rem !important;
        }
        
        .streamlit-expanderContent {
            border: 1px solid black !important;
            border-top: none !important;
            border-radius: 0 0 0.375rem 0.375rem !important;
        }
        
        /* Divider lines */
        .stMarkdown hr, hr {
            border: none !important;
            border-top: 1px solid black !important;
            margin: 1rem 0 !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-color) !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
            text-align: center !important;
        }
        
        h1 {
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 2rem !important;
            letter-spacing: -0.5px !important;
        }
        
        /* Input container */
        .input-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            gap: 2rem;
            padding: 0 1rem;
        }
        
        .left-inputs {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .right-input {
            flex: 1;
        }
        
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, black 0%, blue 100%);
            color: var(--button-text) !important;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            font-weight: 500;
        }
        
        /* Button Text */
        .stButton > button > div > p {
            color: white !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            background: linear-gradient(135deg, blue 0%, black 100%);
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            border: 1px solid #E2E8F0;
            border-radius: 0.375rem;
            background: var(--input-bg);
            backdrop-filter: blur(5px);
            color: var(--text-color);
            caret-color: var(--primary-color);
        }
        
        /* Text area styling */
        .stTextArea>div>div>textarea {
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.375rem !important;
            background: var(--input-bg) !important;
            backdrop-filter: blur(5px) !important;
            color: var(--text-color) !important;
            caret-color: var(--primary-color) !important;
            font-family: inherit !important;
            padding: 0.5rem !important;
        }
        
        /* Input labels */
        .stTextInput > label, .stTextArea > label {
            color: var(--text-color) !important;
        }
        
        /* Placeholder styles */
        .stTextInput>div>div>input::placeholder,
        .stTextArea>div>div>textarea::placeholder {
            color: var(--placeholder-color);
            opacity: 0.8;
        }
        
        /* Selectbox styling */
        .stSelectbox {
            border: none !important;
            background: transparent !important;
        }
        
        /* Selectbox label */
        .stSelectbox > label {
            color: var(--text-color) !important;
        }
        
        .stSelectbox > div > div {
            border: 1px solid darkblue !important;
            border-radius: 0.375rem !important;
            background: var(--input-bg) !important;
        }
        
        .stSelectbox > div > div > div {
            background: var(--input-bg) !important;
        }
        
        .stSelectbox > div > div > div > div {
            background: var(--input-bg) !important;
            color: var(--text-color) !important;
        }
        
        /* Selectbox dropdown */
        div[data-baseweb="popover"] {
            background: transparent !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.375rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
        
        div[data-baseweb="popover"] > div {
            background: var(--input-bg) !important;
        }
        
        div[data-baseweb="popover"] > div > div {
            background: var(--input-bg) !important;
        }
        
        div[data-baseweb="popover"] > div > div > div {
            background: var(--input-bg) !important;
            padding: 8px 12px !important;
            color: var(--text-color) !important;
        }
        
        div[data-baseweb="popover"] > div > div > div:hover {
            background-color: rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Style the dropdown arrow */
        div[data-baseweb="select"] svg {
            fill: var(--text-color) !important;
        }
        
        /* Product cards */
        .product-card {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
            border: 1px solid black;
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease;
        }
        
        .product-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.1), 0 4px 6px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid black;
        }
        
        /* Metric cards */
        .metric-card {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 0.5rem 0;
            border: 1px solid black;
        }
        
        /* Links */
        a {
            color: var(--primary-color) !important;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        
        a:hover {
            color: var(--secondary-color) !important;
            text-decoration: underline;
        }
        
        /* Success messages */
        .stSuccess {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white !important;
            padding: 1rem;
            border-radius: 0.375rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Warning messages */
        .stWarning {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: white !important;
            padding: 1rem;
            border-radius: 0.375rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Error messages */
        .stError {
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
            color: white !important;
            padding: 1rem;
            border-radius: 0.375rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Info messages */
        .stInfo {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: white !important;
            padding: 1rem;
            border-radius: 0.375rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Success and other message boxes */
        .stSuccess, .stWarning, .stError, .stInfo {
            border: 1px solid black !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Setup ---
st.title("🛍️ Online Shopping Assistant")

# --- User Inputs ---
col1, col2 = st.columns([2, 2])

with col1:
    query = st.text_input("Enter your product search query:", placeholder="e.g. black wireless headphones")
    max_price_input = st.text_input("Enter your maximum budget (€):", placeholder="e.g. 100")
    sort_by = st.selectbox("Sort by (optional):", ["None", "Price: Low to High", "Price: High to Low"], index=0)

with col2:
    description_input = st.text_area("Additional Description (optional):", placeholder="e.g. must have noise cancellation", height=206)

# Center the search button using columns
_, center_col, _ = st.columns([5, 1, 5])
with center_col:
    search_clicked = st.button("Search", use_container_width=True)

# --- Main Logic ---
if search_clicked and query:
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Query Processing
    status_text.text("Understanding your query...")
    start_time = time.time()
    
    # Combined query processing instead of multiple LLM calls
    query_analysis = process_query(query, max_price_input)
    structured_query = query_analysis["restructured"]
    
    progress_bar.progress(20)
    status_text.text(f"Searching for products: '{structured_query}'")
    
    # Stage 2: Basic Search
    serp_results = search_serpapi(structured_query)
    progress_bar.progress(40)
    
    if serp_results:
        status_text.text("Analyzing product details...")
    
        try:
            max_price = int(max_price_input) if max_price_input else None
        except ValueError:
            max_price = 10000
            st.warning("Invalid price input. Defaulting to €10,000.")
        
        # Debug: Extract specifications separately for better visualization
        product_details = extract_specifications(serp_results)
        progress_bar.progress(60)
        
        # Stage 3: Ranking and recommendations
        status_text.text("Ranking products based on your search...")
        result_data = rank_products(product_details, structured_query, max_price)
        if isinstance(result_data, tuple) and len(result_data) == 2:
            results, top_recommendations = result_data
        else:
            results = result_data
            top_recommendations = {"text": "", "products": []}
        progress_bar.progress(80)
        
        # Optional additional sorting if user specified
        if sort_by:
            results = sort_results(results, sort_by)

        progress_bar.progress(100)
        status_text.text(f"Search completed in {time.time() - start_time:.2f} seconds")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not results:
            st.warning("No relevant products found matching your criteria.")
        else:
            st.success(f"Found {len(results)} results for: {structured_query}")
            
            # Display top recommendations first
            if top_recommendations and top_recommendations["text"]:
                with st.expander("💫 Top Recommendations", expanded=True):
                    st.markdown(top_recommendations["text"])
                    
                    # Display recommended products in a grid
                    if top_recommendations["products"]:
                        cols = st.columns(min(len(top_recommendations["products"]), 2))
                        for i, product in enumerate(top_recommendations["products"]):
                            with cols[i % 2]:
                                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                                st.subheader(product["title"])
                                if product.get("image"):
                                    st.image(product["image"], width=150)
                                st.write(f"**Price**:  {product.get('price', 'N/A')}")
                                if product.get("rating"):
                                    st.write(f"**Rating**: {product['rating']} ⭐️")
                                st.markdown(f"**Why we recommend it**: {product.get('rank_reason', 'This product matches your search criteria.')}")
                                
                                # Show pros and cons if available
                                if product.get("pros") or product.get("cons"):
                                    st.write("**Pros & Cons**:")
                                    for pro in product.get("pros", []):
                                        st.markdown(f"✅ {pro}")
                                    for con in product.get("cons", []):
                                        st.markdown(f"⚠️ {con}")
                                        
                                st.write(f"[View Product]({product['url']})")
                                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display all results with detailed descriptions
            st.subheader("🔍 All Matching Products")
            for i, r in enumerate(results):
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                cols = st.columns([1, 3])
                with cols[0]:
                    if r.get("image"):
                        st.image(r["image"], width=100)
                with cols[1]:
                    st.subheader(r["title"])
                    st.write(f"**Price**: {r.get('price', 'N/A')}")
                    if r.get("rating"):
                        st.write(f"**Rating**: {r['rating']} ⭐️")
                    if r.get("reviews"):
                        st.write(f"**Reviews**: {r['reviews']}")
                    st.write(f"[View Product]({r['url']})")
                
                # Show product analysis below image and basic info
                st.markdown(f"**Why this product made the list**: {r.get('rank_reason', 'This product matches your search criteria.')}")
                
                # Enhanced specifications display with fallback options
                if r.get("specifications") and isinstance(r.get("specifications"), dict) and r.get("specifications"):
                    with st.expander("Show Specifications"):
                        for key, value in r["specifications"].items():
                            if isinstance(value, str):
                                if value.startswith("+") or value.startswith("-"):
                                    # This is likely a pros/cons list, format accordingly
                                    st.markdown(f"**{key}**:")
                                    for line in value.split("\n"):
                                        if line.strip():
                                            if line.strip().startswith("+"):
                                                st.markdown(f"✅ {line.strip()[1:].strip()}")
                                            elif line.strip().startswith("-"):
                                                st.markdown(f"⚠️ {line.strip()[1:].strip()}")
                                            else:
                                                st.markdown(f"• {line.strip()}")
                                else:
                                    st.markdown(f"**{key.capitalize()}**: {value}")
                            else:
                                st.markdown(f"**{key.capitalize()}**: {value}")
                elif r.get("pros") or r.get("cons"):
                    # Fallback to showing pros and cons directly
                    with st.expander("Show Specifications"):
                        if r.get("pros"):
                            st.markdown("**Pros:**")
                            for pro in r["pros"]:
                                st.markdown(f"✅ {pro}")
                        if r.get("cons"):
                            st.markdown("**Cons:**")
                            for con in r["cons"]:
                                st.markdown(f"⚠️ {con}")
                        # Add basic details
                        st.markdown(f"**Price**: {r.get('price', 'N/A')}")
                        if r.get("rating"):
                            st.markdown(f"**Rating**: {r['rating']}")
                        if r.get("detailed_description"):
                            st.markdown(f"**Details**: {r['detailed_description']}")
                elif r.get("details"):
                    # Last resort fallback
                    with st.expander("Show Specifications"):
                        st.markdown(r["details"][:500] + "..." if len(r["details"]) > 500 else r["details"])
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

            

    else:
        st.warning("No products found. Please try a different search query.")

# --- Footer ---
st.markdown("---")
st.caption("Online Shopping Assistant ❤️")