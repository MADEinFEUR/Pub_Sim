import networkx as nx
import math
import csv

def load_graph_from_csv(nodes_file='Data\small_data.csv', connections_file='connections.csv'):
    """
    Charge les données des fichiers CSV et construit un graphe NetworkX.
    """
    G = nx.Graph()
    node_coords = {}

    # Lecture des nœuds et de leurs coordonnées
    with open(nodes_file, 'r') as node_file:
        reader = csv.DictReader(node_file)
        for row in reader:
            node_id = row['node_id']
            coords = (int(row['x']), int(row['y']), int(row['z']))
            G.add_node(node_id, pos=coords)
            node_coords[node_id] = coords

    # Lecture des connexions et calcul des poids des arêtes
    with open(connections_file, 'r') as conn_file:
        reader = csv.DictReader(conn_file)
        for row in reader:
            source, target = row['source'], row['target']
            if source in node_coords and target in node_coords:
                dist = math.sqrt(
                    (node_coords[source][0] - node_coords[target][0]) ** 2 +
                    (node_coords[source][1] - node_coords[target][1]) ** 2 +
                    (node_coords[source][2] - node_coords[target][2]) ** 2
                )
                G.add_edge(source, target, weight=dist)

    return G