import sys
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QDialog, QLabel, QLineEdit, QFormLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Sample spatial context data
spatial_context = {
    "room1": [
        {"entity": {"type": "avatar", "name": "player", "status": "stand"}, "relation": "has", "target": {"type": "item", "id": 1, "name": "mustard spinach", "status": "unwashed"}},
        {"entity": {"entity": {"type": "item", "id": 2, "name": "bowl", "status": "contain water"}, "relation": "has", "target": {"type": "item", "id": 3, "name": "sieve", "status": "in water"}}, "relation": "has", "target": {"type": "item", "id": 1, "name": "mustard spinach", "status": "unwashed"}},
        {"entity": {"type": "item", "id": 4, "name": "pot", "status": "heating"}, "relation": "has", "target": {"type": "item", "id": 5, "name": "water", "status": "boiling"}}
    ]
}

class EditNodeDialog(QDialog):
    def __init__(self, node_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Node")
        self.node_label = node_label
        
        self.layout = QFormLayout()
        self.name_input = QLineEdit()
        self.status_input = QLineEdit()
        self.id_input = QLineEdit()
        
        self.layout.addRow(QLabel("Name:"), self.name_input)
        self.layout.addRow(QLabel("Status:"), self.status_input)
        self.layout.addRow(QLabel("ID:"), self.id_input)
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        self.layout.addWidget(save_button)
        
        self.setLayout(self.layout)
    
    def get_data(self):
        return self.name_input.text(), self.status_input.text(), self.id_input.text()

class SceneGraphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Scene Graph")
        self.setGeometry(100, 100, 800, 600)
        
        self.graph = nx.DiGraph()
        self.create_graph()
        
        self.canvas = FigureCanvas(plt.figure())
        self.setCentralWidget(self.canvas)
        self.draw_graph()
        
        self.canvas.mpl_connect("button_press_event", self.on_click)
    
    def create_graph(self):
        for room, relations in spatial_context.items():
            for relation in relations:
                entity = relation["entity"]
                target = relation["target"]
                relation_type = relation["relation"]
                
                if isinstance(entity, dict) and "entity" in entity:
                    parent_entity = entity["entity"]
                    nested_relation = entity["relation"]
                    nested_target = entity["target"]
                    
                    self.add_nodes_edges(parent_entity, nested_target, nested_relation)
                    self.add_nodes_edges(nested_target, target, relation_type)
                else:
                    self.add_nodes_edges(entity, target, relation_type)
    
    def add_nodes_edges(self, entity, target, relation):
        entity_label = f"{entity['name']} ({entity['status']})"
        target_label = f"{target['name']} ({target['status']})"
        
        self.graph.add_node(entity_label)
        self.graph.add_node(target_label)
        self.graph.add_edge(entity_label, target_label, label=relation)
    
    def draw_graph(self):
        plt.clf()
        pos = nx.spring_layout(self.graph)
        edge_labels = {(u, v): d["label"] for u, v, d in self.graph.edges(data=True)}
        
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)
        
        self.canvas.draw()
    
    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            clicked_node = None
            for node in self.graph.nodes:
                if event.xdata in range(int(event.xdata - 1), int(event.xdata + 1)) and event.ydata in range(int(event.ydata - 1), int(event.ydata + 1)):
                    clicked_node = node
                    break
            
            if clicked_node:
                dialog = EditNodeDialog(clicked_node, self)
                if dialog.exec_():
                    name, status, node_id = dialog.get_data()
                    if name and status:
                        new_label = f"{name} ({status})"
                        self.graph = nx.relabel_nodes(self.graph, {clicked_node: new_label})
                        self.draw_graph()

if __name__ == "__main__":
    print("dd")
    app = QApplication(sys.argv)
    window = SceneGraphApp()
    window.show()
    sys.exit(app.exec_())
