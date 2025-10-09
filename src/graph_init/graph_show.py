import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G):
    """
    Visualise le graphe à l'aide de Matplotlib.
    """
    print("\n--- Visualisation du graphe ---")

    # Définir le layout de la visualisation
    pos = nx.spring_layout(G)

    # Options de style pour les nœuds et les arêtes
    node_sizes = [v * 5000 for v in nx.betweenness_centrality(G).values()]
    node_colors = ['skyblue' if G.nodes[n].get('role') != 'admin' else 'salmon' for n in G.nodes()]

    # Dessiner le graphe
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    plt.title("Visualisation du graphe simple")
    plt.axis("off")  # Cacher les axes
    plt.show()