# graph_template.py
import matplotlib
matplotlib.use('TkAgg') # Ou 'Qt5Agg' ou 'Agg'
import matplotlib.pyplot as plt
import networkx as nx



def create_and_manipulate_graph():
    """
    Crée un graphe simple, y ajoute des nœuds et des arêtes,
    et démontre l'ajout de propriétés.
    """
    print("--- Création et manipulation de graphes ---")

    # Création d'un graphe non orienté
    G = nx.Graph()
    print(f"Graphe non orienté créé : {type(G)}")

    # Ajout de nœuds
    G.add_node("A")
    G.add_nodes_from(["B", "C", "D"])
    print(f"Nœuds ajoutés : {G.nodes()}")

    # Ajout d'arêtes
    G.add_edge("A", "B")
    G.add_edges_from([("B", "C"), ("C", "D"), ("D", "A")])
    print(f"Arêtes ajoutées : {G.edges()}")

    # Ajout de propriétés aux nœuds et aux arêtes
    G.add_node("A", role="admin", last_login="2023-10-27")
    G.add_edge("A", "B", weight=5, connection_type="friendship")
    print(f"\nPropriétés du nœud 'A' : {G.nodes['A']}")
    print(f"Propriétés de l'arête ('A', 'B') : {G.get_edge_data('A', 'B')}")

    return G


def analyze_graph_structure(G):
    """
    Explore les propriétés et la structure d'un graphe.
    """
    print("\n--- Analyse de la structure du graphe ---")
    print(f"Nombre de nœuds : {G.number_of_nodes()}")
    print(f"Nombre d'arêtes : {G.number_of_edges()}")

    # Degré des nœuds (nombre de connexions)
    print(f"Degré de chaque nœud : {G.degree()}")
    print(f"Voisins du nœud 'A' : {list(G.neighbors('A'))}")

    # Vérification de la connexité
    if nx.is_connected(G):
        print("Le graphe est connexe.")
    else:
        print("Le graphe n'est pas connexe.")


def apply_algorithms(G):
    """
    Démontre l'utilisation de quelques algorithmes de graphes courants.
    """
    print("\n--- Utilisation des algorithmes ---")

    # Exemple de parcours en largeur (BFS)
    print("Parcours en largeur (BFS) depuis le nœud 'A' :")
    bfs_tree = nx.bfs_tree(G, source="A")
    print(f"  - Arbres de parcours : {bfs_tree.edges()}")

    # Recherche du chemin le plus court (non pondéré)
    try:
        shortest_path = nx.shortest_path(G, source="B", target="D")
        print(f"Chemin le plus court entre 'B' et 'D' : {shortest_path}")
    except nx.NetworkXNoPath:
        print("Aucun chemin trouvé entre 'B' et 'D'.")

    # Calcul de la centralité des nœuds
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    print(f"Centralité de degré : {degree_centrality}")
    print(f"Centralité d'intermédiarité : {betweenness_centrality}")


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


if __name__ == "__main__":
    # Étape 1 : Création du graphe
    graph = create_and_manipulate_graph()

    # Étape 2 : Analyse de la structure
    analyze_graph_structure(graph)

    # Étape 3 : Application d'algorithmes
    apply_algorithms(graph)

    # Étape 4 : Visualisation
    visualize_graph(graph)
