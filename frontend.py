import streamlit as st
from backend import (
    restructure_query_with_llama,
    search_serpapi,
    enrich_with_tavily,
    validate_results,
    rank_results,
    summarize_with_llama,
    sort_results,
    generate_comparison_table,
    is_comparison_query,
    is_recommendation_query,
    generate_recommendations,
    extract_filters_from_query,
    translate_to_english
)

# --- UI Setup ---
st.set_page_config(page_title="Online Shopping Assistant", layout="wide")
st.title("🛍️ Online Shopping Assistant")

query = st.text_input("Enter your product search query:", placeholder="e.g. compare black wireless headphones under 100 euros")

col1, col2, col3 = st.columns(3)
with col1:
    selected_site = st.selectbox("Filter by website (optional):", ["", "amazon", "ebay", "otto", "zalando", "kleinanzeigen"])
with col2:
    max_price = st.slider("Maximum Price (€):", 5, 10000, 100)
with col3:
    sort_by = st.selectbox("Sort by (optional):", ["", "Price: Low to High", "Price: High to Low"])

# --- Main Logic ---
if st.button("Search") and query:
    with st.spinner("Understanding and restructuring your query..."):
        translated_query = translate_to_english(query)
        structured_query = restructure_query_with_llama(translated_query)
        comparison_mode = is_comparison_query(query)
        recommendation_mode = is_recommendation_query(query)

        st.session_state["restructured_query"] = structured_query
        st.session_state["is_comparison"] = comparison_mode
        st.session_state["is_reco"] = recommendation_mode

    with st.spinner(f"Searching for: {structured_query}"):
        serp_results = search_serpapi(structured_query)[:20]
        enriched = enrich_with_tavily(serp_results)
        validated = validate_results(enriched)
        ranked = rank_results(validated, structured_query, selected_site, max_price)

        # Extract adaptive filters
        filters = extract_filters_from_query(query)
        st.session_state["filters"] = filters

        st.session_state["results"] = ranked
        st.session_state["active_search"] = True

# --- Display Results ---
if "results" in st.session_state and st.session_state.get("active_search"):
    filters = st.session_state.get("filters", {})
    brand_options = list(set(filters.get("brands", [])))
    color_options = list(set(filters.get("colors", [])))

    st.markdown("### 🎛️ Adaptive Filters")

    if brand_options:
        selected_brand = st.selectbox("Filter by detected brand:", [""] + brand_options)
    else:
        selected_brand = ""

    if color_options:
        selected_color = st.selectbox("Filter by detected color:", [""] + color_options)
    else:
        selected_color = ""

    # Apply brand and color filters if selected
    filtered_results = st.session_state["results"]
    if selected_brand:
        filtered_results = [r for r in filtered_results if selected_brand.lower() in r["title"].lower()]
    if selected_color:
        filtered_results = [r for r in filtered_results if selected_color.lower() in r["title"].lower()]

    # Sort if selected
    results = sort_results(filtered_results, sort_by) if sort_by else filtered_results

    if not results:
        st.warning("No relevant products found.")
    else:
        st.success(f"Showing top {min(10, len(results))} results for: {st.session_state.get('restructured_query')}")
        for r in results[:10]:
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
                    if r.get("availability"):
                        st.write(f"**Availability**: {r['availability']}")
                    if r.get("delivery"):
                        st.write(f"**Delivery**: {r['delivery']}")
                    st.write(f"[View Product]({r['url']})")
                    summary = summarize_with_llama(r["content"]) if r["content"] else "No summary available."
                    st.caption(summary)
                    if r.get("specifications"):
                        st.markdown("**Specifications:**")
                        for key, value in r["specifications"].items():
                            st.markdown(f"- **{key.capitalize()}**: {value}")


        # Comparison view if applicable
        if st.session_state.get("is_comparison"):
            st.markdown("---")
            st.subheader("📊 Side-by-Side Product Comparison (Top 3)")
            top3 = results[:3]
            comparison = generate_comparison_table(top3)

            col_titles = st.columns(len(top3))
            for i, title in enumerate(comparison["Title"]):
                col_titles[i].markdown(f"### {title[:40]}")

            for row_label in ["Price", "Rating", "Delivery", "Key Features", "Pros/Cons"]:
                cols = st.columns(len(top3))
                for i, col in enumerate(cols):
                    col.markdown(f"**{row_label}:**")
                    col.markdown(comparison[row_label][i] if comparison[row_label][i] else "N/A")

        # Recommendation view if applicable
        if st.session_state.get("is_reco"):
            reco_text = generate_recommendations(results, query)
            st.markdown("### 🎯 Recommendations")
            st.markdown(reco_text)
