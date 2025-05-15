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

# --- UI Setup ---
st.set_page_config(page_title="Online Shopping Assistant", layout="wide")
st.title("🛍️ Online Shopping Assistant")

# --- User Inputs ---
query = st.text_input("Enter your product search query:", placeholder="e.g. black wireless headphones")

col1, col2, col3 = st.columns(3)
with col1:
    max_price_input = st.text_input("Enter your maximum budget (€):", placeholder="e.g. 100")
with col2:
    description_input = st.text_input("Additional Description (optional):", placeholder="e.g. must have noise cancellation")
with col3:
    sort_by = st.selectbox("Sort by (optional):", ["", "Price: Low to High", "Price: High to Low"])

# --- Main Logic ---
if st.button("Search") and query:
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Query Processing
    status_text.text("Understanding your query...")
    start_time = time.time()
    
    # Combined query processing instead of multiple LLM calls
    query_analysis = process_query(query)
    structured_query = query_analysis["restructured"]
    comparison_mode = query_analysis["is_comparison"]
    recommendation_mode = query_analysis["is_recommendation"]
    
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
        
        product_details = extract_specifications(serp_results)
        st.write(json.dumps(product_details, indent=2, ensure_ascii=False))
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
                with st.expander("💫 **Top Recommendations**", expanded=True):
                    st.markdown(top_recommendations["text"])
                    
                    # Display recommended products in a grid
                    if top_recommendations["products"]:
                        cols = st.columns(min(len(top_recommendations["products"]), 2))
                        for i, product in enumerate(top_recommendations["products"]):
                            with cols[i % 2]:
                                st.subheader(product["title"])
                                if product.get("image"):
                                    st.image(product["image"], width=150)
                                st.write(f"**Price**: {product.get('price', 'N/A')}")
                                if product.get("rating"):
                                    st.write(f"**Rating**: {product['rating']} ⭐️")
                                st.markdown(f"**Why we recommend it**: {product.get('rank_reason', 'No reason provided.')}")
                                st.write(f"[View Product]({product['url']})")
            
            # Display all results with detailed descriptions
            st.subheader("🔍 All Matching Products")
            for i, r in enumerate(results):
                with st.container():
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
                st.markdown(f"**Why this product made the list**: {r.get('detailed_description', 'No detailed description available.')}")
                
                # Show specifications in an expander
                if r.get("specifications") or r.get("details"):
                    with st.expander("Show Specifications"):
                        if r.get("specifications"):
                            for key, value in r["specifications"].items():
                                st.markdown(f"- **{key.capitalize()}**: {value}")
                        elif r.get("details"):
                            st.markdown(r["details"])
                
                st.markdown("---")

            if comparison_mode and len(results) >= 2:
                st.markdown("---")
                st.subheader("📊 Side-by-Side Product Comparison (Top 3)")
                with st.spinner("Generating comparison..."):
                    comparison = generate_comparison_table(results[:3])
                
                col_titles = st.columns(len(comparison["Title"]))
                for i, title in enumerate(comparison["Title"]):
                    col_titles[i].markdown(f"### {title[:40]}")
                for row_label in ["Price", "Rating", "Key Features", "Pros/Cons"]:
                    cols = st.columns(len(comparison["Title"]))
                    for i, col in enumerate(cols):
                        col.markdown(f"**{row_label}:**")
                        col.markdown(comparison[row_label][i] if comparison[row_label][i] else "N/A")

    else:
        st.warning("No products found. Please try a different search query.")

# --- Footer ---
st.markdown("---")
st.caption("Online Shopping Assistant ❤️")