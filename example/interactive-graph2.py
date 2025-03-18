import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Title
st.title("Interactive Graph with Node Attributes")

# Create Graph
G = nx.Graph()

# Node Attributes
node_info = {
    "A": "A is the first letter of the alphabet",
    "B": "B is the second letter of the alphabet",
    "C": "C is the third letter of the alphabet",
    "D": "D is the fourth letter of the alphabet"
}

# Add nodes with attributes
for node, info in node_info.items():
    G.add_node(node, title=info)  # "title" in Pyvis shows on hover

# Add Edges
G.add_edges_from([("A", "B"), ("A", "C"), ("C", "D")])

# Function to Render Graph
def draw_graph(graph):
    net = Network(height="500px", width="100%", notebook=False, cdn_resources="remote")
    
    # Add nodes with attributes as tooltips
    for node, info in node_info.items():
        net.add_node(node, label=node, title=info)

    # Add edges
    for edge in graph.edges():
        net.add_edge(edge[0], edge[1])

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        temp_path = tmpfile.name
        net.save_graph(temp_path)

    # Show in Streamlit
    with open(temp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=550)
    os.remove(temp_path)  # Clean up

# Render Graph
draw_graph(G)