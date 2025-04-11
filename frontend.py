import streamlit as st
from backend import (
    search_serpapi,
    enrich_with_tavily,
    validate_results,
    rank_results,
    summarize_with_llama,
    sort_results
)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Online Shopping Assistant", layout="wide")
st.title("🛍️ Online Shopping Assistant")

query = st.text_input("Enter your product search query:", placeholder="e.g. wireless headphones under 100 euros")

col1, col2, col3 = st.columns(3)

with col1:
    selected_site = st.selectbox("Filter by website (optional):", ["", "amazon", "ebay", "otto", "zalando", "kleinanzeigen"])

with col2:
    max_price = st.slider("Maximum Price (€):", 5, 10000, 100)

with col3:
    sort_by = st.selectbox("Sort by(optional):", ["", "Price: Low to High", "Price: High to Low"])

# --- Fetch Results ---
if st.button("Search") and query:
    with st.spinner("Searching for products..."):
        # serp_results = search_serpapi(query)
        serp_results = search_serpapi(query)[:20]
        enriched = enrich_with_tavily(serp_results)
        validated = validate_results(enriched)
        ranked = rank_results(validated, query, selected_site, max_price)
        
        # Save to session_state
        st.session_state["results"] = ranked
        st.session_state["query"] = query  # Save original query for LLaMA
        st.session_state["active_search"] = True

# --- Display Results ---
if "results" in st.session_state and st.session_state.get("active_search"):
    # Only sort if user selected a sort option
    results = (
        sort_results(st.session_state["results"], sort_by)
        if sort_by else st.session_state["results"]
    )

    if not results:
        st.warning("No relevant products found.")
    else:
        st.success(f"Showing top {min(10, len(results))} results")
        for r in results[:10]:
            with st.container():
                st.subheader(r["title"])
                st.write(f"**Price**: {r.get('price', 'N/A')}")

                # ⭐️ Optional rating + reviews display
                rating = r.get("rating")
                reviews = r.get("reviews")
                if rating:
                    st.write(f"**Rating**: {rating} ⭐️")
                if reviews:
                    st.write(f"**Reviews**: {reviews}")

                st.write(f"[View Product]({r['url']})")
                summary = summarize_with_llama(r["content"]) if r["content"] else "No summary available."
                st.caption(summary)