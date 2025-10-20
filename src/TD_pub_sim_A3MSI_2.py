import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

def build_graph(df_nodes, k=5):
    features = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
    node_ids = df_nodes['node_id'].values
    G = nx.Graph()
    for node in node_ids:
        G.add_node(node)
    dist_matrix = cdist(features, features, metric='euclidean')
    for i, node in enumerate(node_ids):
        nearest = np.argsort(dist_matrix[i])[1:k+1]
        for j in nearest:
            G.add_edge(node, node_ids[j], weight=dist_matrix[i, j])
    return G

def weighted_distance_matrix(A_vector, X, w_vector):
    diff = X - A_vector
    return np.sqrt(np.sum(w_vector * diff**2, axis=1))

def radius_search_clustered_50D_vectorized(df_nodes, df_queries, features, cluster_dict, cluster_centroids):
    results = []
    for _, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])
        dist_to_centroids = weighted_distance_matrix(A_vector, cluster_centroids, w_vector)
        close_clusters = np.where(dist_to_centroids <= radius + 1e-6)[0]
        found_nodes = []
        for cluster_id in close_clusters:
            idxs = cluster_dict[cluster_id]
            nodes_features = features[idxs]
            dists = weighted_distance_matrix(A_vector, nodes_features, w_vector)
            mask = dists <= radius
            for node_idx, d in zip(np.array(idxs)[mask], dists[mask]):
                found_nodes.append((df_nodes.iloc[node_idx]['node_id'], d))
        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    return results

def main():
    root = tk.Tk()
    root.withdraw()

    # Choix des fichiers
    nodes_path = filedialog.askopenfilename(title="Choisir le fichier des nœuds (CSV)", filetypes=[("CSV files", "*.csv")])
    if not nodes_path:
        messagebox.showerror("Erreur", "Fichier des nœuds non sélectionné.")
        return

    queries_path = filedialog.askopenfilename(title="Choisir le fichier des requêtes (CSV)", filetypes=[("CSV files", "*.csv")])
    if not queries_path:
        messagebox.showerror("Erreur", "Fichier des requêtes non sélectionné.")
        return

    # Chargement des données
    df_nodes = pd.read_csv(nodes_path)
    df_queries = pd.read_csv(queries_path)

    print("Données des nœuds :")
    print(df_nodes.head())
    print("\nRequêtes :")
    print(df_queries.head())

    # Construction graphe
    G = build_graph(df_nodes, k=5)

    features = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values

    N_nodes = len(df_nodes)
    n_clusters = max(10, min(int(np.sqrt(N_nodes)), 200))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    df_nodes['cluster_id'] = cluster_labels
    cluster_dict = {i: np.where(cluster_labels == i)[0] for i in range(n_clusters)}
    cluster_centroids = kmeans.cluster_centers_

    t0 = time.time()
    results = radius_search_clustered_50D_vectorized(df_nodes, df_queries, features, cluster_dict, cluster_centroids)
    print(f"Recherche terminée en {time.time() - t0:.3f} secondes")

    # Affichage résultats console
    counts = []
    for res in results:
        n_found = len(res['found_nodes'])
        if n_found > 0:
            print(f"\nRequête : {res['query']} (Rayon={res['radius']})")
            print(f"Nombre de nœuds trouvés : {n_found}")
            for node_id, dist in res['found_nodes']:
                print(f" - Noeud {node_id}, distance pondérée {dist:.4f}")
        counts.append(n_found)

    # Graphique
    plt.figure(figsize=(10,5))
    plt.bar(range(len(counts)), counts)
    plt.title("Nombre de nœuds trouvés par requête")
    plt.xlabel("Index de la requête")
    plt.ylabel("Nombre de nœuds trouvés")
    plt.show()

if __name__ == "__main__":
    main()
