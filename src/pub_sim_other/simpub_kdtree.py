import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# --- Chargement des données ---
print("=== Chargement des données ===")
t0 = time.time()
df_nodes = pd.read_csv('Data/adsSim_data_nodes.csv')
df_queries = pd.read_csv('Data/queries_structured.csv')
print(f"Temps de chargement : {time.time() - t0:.3f} s")

# --- Construction du graphe pondéré ---
def build_graph(df_nodes, k=5):
    features = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
    node_ids = df_nodes['node_id'].values
    G = nx.Graph()
    for node in node_ids:
        G.add_node(node)
    dist_matrix = cdist(features, features, metric='euclidean')
    for i, node in enumerate(node_ids):
        nearest = np.argsort(dist_matrix[i])[1:k+1]  # on ignore soi-même
        for j in nearest:
            G.add_edge(node, node_ids[j], weight=dist_matrix[i, j])
    return G

# --- KD-Tree pondéré ---
def kd_tree_weighted_radius_search(df_nodes, df_queries):
    """
    KD-Tree avec distance pondérée par Y_vector
    """
    results = []
    features_cols = [f'feature_{i}' for i in range(1, 51)]
    X = df_nodes[features_cols].values.astype(float)
    node_ids = df_nodes['node_id'].values

    print("\n=== Recherche KD-Tree pondérée ===")
    t0 = time.time()
    for _, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])

        # Transformation pour distance pondérée
        sqrt_w = np.sqrt(w_vector)
        X_scaled = X * sqrt_w
        A_scaled = A_vector * sqrt_w

        # Construction du KD-Tree pondéré pour cette requête
        tree_scaled = KDTree(X_scaled, leaf_size=40)

        # Recherche dans le rayon
        indices = tree_scaled.query_radius([A_scaled], r=radius)[0]

        # Calcul exact de la distance pondérée pour les résultats
        found_nodes = []
        for idx in indices:
            dist = np.sqrt(np.sum(w_vector * (X[idx] - A_vector) ** 2))
            found_nodes.append((node_ids[idx], dist))

        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    t1 = time.time()
    print(f"Recherche KD-Tree pondérée terminée en {t1 - t0:.3f} s")
    return results

# --- Comparaison Vectorisée vs KD-Tree pondéré ---
print("\n=== Comparaison Vectorisé vs KD-Tree pondéré ===")

def vectorized_radius_search(df_nodes, df_queries):
    results = []
    X = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values.astype(float)
    node_ids = df_nodes['node_id'].values
    for _, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])
        diff = X - A_vector
        dist = np.sqrt(np.sum(w_vector * diff ** 2, axis=1))
        found_mask = dist <= radius
        found_nodes = list(zip(node_ids[found_mask], dist[found_mask]))
        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    return results

# --- Mesure de performance ---
t0 = time.time()
res_vectorized = vectorized_radius_search(df_nodes, df_queries)
tv = time.time() - t0
print(f"→ Méthode vectorisée : {tv:.3f} s")

t0 = time.time()
res_kdtree_weighted = kd_tree_weighted_radius_search(df_nodes, df_queries)
tk = time.time() - t0
print(f"→ Méthode KD-Tree pondérée : {tk:.3f} s")

# --- Comparaison des résultats ---
vector_counts = [len(r['found_nodes']) for r in res_vectorized]
kdtree_counts = [len(r['found_nodes']) for r in res_kdtree_weighted]

plt.figure(figsize=(10, 5))
plt.plot(vector_counts, label='Vectorisée (NumPy)')
plt.plot(kdtree_counts, label='KD-Tree pondérée')
plt.xlabel('Index de la requête')
plt.ylabel('Nombre de nœuds trouvés')
plt.title('Comparaison Vectorisée vs KD-Tree pondérée')
plt.legend()
plt.show()
