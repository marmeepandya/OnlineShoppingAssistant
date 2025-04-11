from langgraph.graph import StateGraph
from langgraph.graph.graph import END

# Nodes (simple passthroughs — real logic is in backend)
def node_search(state): return {"query": state["query"], "products": state["search_fn"](state["query"])}
def node_enrich(state): return {"products": state["enrich_fn"](state["products"])}
def node_validate(state): return {"products": state["validate_fn"](state["products"])}
def node_rank(state): 
    ranked = state["rank_fn"](state["products"], state["query"], state.get("site"), state.get("max_price"))
    return {"results": ranked}
def node_done(state): return state

# Create LangGraph
def create_graph():
    graph = StateGraph(dict)
    graph.add_node("search", node_search)
    graph.add_node("enrich", node_enrich)
    graph.add_node("validate", node_validate)
    graph.add_node("rank", node_rank)
    graph.add_node("done", node_done)

    graph.set_entry_point("search")
    graph.add_edge("search", "enrich")
    graph.add_edge("enrich", "validate")
    graph.add_edge("validate", "rank")
    graph.add_edge("rank", "done")
    graph.set_finish_point("done")

    return graph.compile()
