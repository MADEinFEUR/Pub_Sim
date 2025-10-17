import csv
import math
import time
import random
import networkx as nx

_single_source_dijkstra_cache = {}


def load_graph_from_csv(nodes_file='Data\small_data.csv', connections_file='Data\queries_structured.csv'):
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


def find_shortest_path(graph, start_node, end_node):
    """
    Trouve le chemin le plus court entre deux nœuds en utilisant l'algorithme de Dijkstra.
    Retourne la distance totale et la liste des nœuds du chemin.
    """
    try:
        path = nx.dijkstra_path(graph, source=start_node, target=end_node, weight='weight')
        distance = nx.dijkstra_path_length(graph, source=start_node, target=end_node, weight='weight')

        # Objet complexe : dictionnaire
        result = {
            "distance": distance,
            "path": path
        }
        return result

    except nx.NetworkXNoPath:
        return {"distance": float('inf'), "path": []}
    except nx.NodeNotFound:
        return {"distance": float('inf'), "path": []}



# Utiliser un dictionnaire global pour le cache
_shortest_path_cache = {}


def find_shortest_path_cached(graph, start_node, end_node):
    """
    Trouve le chemin le plus court avec mise en cache.
    """
    # Clé du cache : tuple (start, end)
    cache_key = (start_node, end_node)

    # 1. Vérifier si le résultat est déjà en cache
    if cache_key in _shortest_path_cache:
        # print(f"Cache hit pour le chemin de {start_node} à {end_node}")
        return _shortest_path_cache[cache_key]

    # 2. Si le résultat n'est pas en cache, le calculer
    try:
        path = nx.dijkstra_path(graph, source=start_node, target=end_node, weight='weight')
        distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

        result = {
            "distance": distance,
            "path": path
        }

        # 3. Mettre le résultat en cache avant de le retourner
        _shortest_path_cache[cache_key] = result
        return result

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Mettre en cache les résultats infinis
        _shortest_path_cache[cache_key] = {"distance": float('inf'), "path": []}
        return _shortest_path_cache[cache_key]


def find_shortest_path_cached_v2(graph, start_node, end_node):
    """
    Trouve le chemin le plus court en utilisant un cache par nœud source.
    """
    # 1. Vérifier si les résultats pour le nœud source sont déjà en cache
    if start_node not in _single_source_dijkstra_cache:
        # print(f"Cache miss pour le nœud source '{start_node}'. Calcul de tous les chemins...")

        # Exécuter Dijkstra une seule fois pour le nœud source et mettre les résultats en cache
        try:
            distances, paths = nx.single_source_dijkstra(graph, source=start_node, weight='weight')
            _single_source_dijkstra_cache[start_node] = (distances, paths)
        except nx.NodeNotFound:
            # Gérer le cas où le nœud source n'existe pas
            return {"distance": float('inf'), "path": []}

    # 2. Récupérer les résultats du cache
    distances, paths = _single_source_dijkstra_cache[start_node]

    # 3. Extraire le chemin et la distance vers le nœud cible
    if end_node in distances:
        result = {
            "distance": distances[end_node],
            "path": paths[end_node]
        }
        return result
    else:
        # Gérer le cas où le nœud cible n'est pas accessible
        return {"distance": float('inf'), "path": []}


def main():
    """
    Fonction principale pour charger le graphe et mesurer le temps d'exécution
    de trois implémentations différentes.
    """
    print("--- Préparation du graphe ---")
    G = load_graph_from_csv()
    print(f"Graphe chargé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.")

    nodes = list(G.nodes())
    if len(nodes) < 2:
        print("Erreur: Pas assez de nœuds pour calculer un chemin.")
        return

    num_runs = 5000

    # Liste des fonctions à tester avec un nom descriptif
    implementations = [
        ("Version 1 : Appel double Dijkstra", find_shortest_path),
        ("Version 2 : Simple cache (memoization)", find_shortest_path_cached),
        ("Version 3 : Cache par source (single_source_dijkstra)", find_shortest_path_cached_v2)
    ]

    for name, func in implementations:
        # Réinitialiser le cache pour chaque nouvelle exécution
        _shortest_path_cache.clear()
        _single_source_dijkstra_cache.clear()

        print(f"\n--- Exécution de : {name} ---")

        start_time = time.time()

        for i in range(num_runs):
            start_node, end_node = random.sample(nodes, 2)
            # Exécuter l'implémentation actuelle
            path_info = func(G, start_node, end_node)

            if i % 100 == 0:
                print(f"  > Itération {i + 1} : De {start_node} à {end_node}. Distance: {path_info['distance']:.2f}")

        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / num_runs

        print(f"\n--- Bilan pour {name} ---")
        print(f"Nombre de chemins calculés : {num_runs}")
        print(f"Temps total d'exécution : {total_time:.4f} secondes")
        print(f"Temps moyen par calcul : {average_time:.6f} secondes")
        print("--------------------------------------------------")


if __name__ == '__main__':
    main()