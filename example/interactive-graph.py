import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Title
st.title("Interactive Graph with Streamlit & Pyvis")

# Create a NetworkX graph
G = nx.Graph()
G.add_nodes_from(["A", "B", "C", "D"])
G.add_edges_from([("A", "B"), ("A", "C"), ("C", "D")])

# Save and render the interactive graph
def draw_graph(graph):
    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(graph)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        temp_path = tmpfile.name
        net.save_graph(temp_path)

    # Show in Streamlit
    with open(temp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=550)
    os.remove(temp_path)  # Clean up

draw_graph(G)