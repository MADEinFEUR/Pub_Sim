import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
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

print("Aperçu de small_data.csv :")
print(df_nodes.head())
print("\nAperçu de queries_structured.csv :")
print(df_queries.head())


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


print("\n=== Construction du graphe ===")
t0 = time.time()
G = build_graph(df_nodes, k=5)
print(f"Graphe construit avec {len(G.nodes)} nœuds et {len(G.edges)} arêtes.")
print(f"Temps de construction : {time.time() - t0:.3f} s")


# --- Distance pondérée vectorisée ---
def weighted_distance_matrix(A_vector, X_matrix, w_vector):
    """
    Calcule toutes les distances pondérées entre un vecteur A et chaque ligne de X.
    Vectorisé avec NumPy pour éviter les boucles.
    """
    diff = X_matrix - A_vector
    dist = np.sqrt(np.sum(w_vector * diff ** 2, axis=1))
    return dist


# --- Nouvelle version vectorisée de la recherche ---
def vectorized_radius_search(df_nodes, df_queries):
    results = []
    features_cols = [f'feature_{i}' for i in range(1, 51)]
    X = df_nodes[features_cols].values.astype(float)
    node_ids = df_nodes['node_id'].values

    for _, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])

        # Calcul vectorisé sur tout X
        dist = weighted_distance_matrix(A_vector, X, w_vector)

        # Sélection des nœuds dans le rayon
        found_mask = dist <= radius
        found_nodes = list(zip(node_ids[found_mask], dist[found_mask]))

        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })

    return results


print("\n=== Recherche naïve vectorisée (optimisée NumPy) ===")
t0 = time.time()
all_results = vectorized_radius_search(df_nodes, df_queries)
t1 = time.time()
print(f"Recherche vectorisée terminée en {t1 - t0:.3f} s")

# Affichage des résultats
for result in all_results:
    print(f"Requête {result['query']} (rayon {result['radius']}): {len(result['found_nodes'])} nœuds trouvés")


# --- Construction des arêtes (visualisation) ---
def get_knn_edges(df_nodes, k=5):
    X = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # on saute le premier (soi-même)
            edges.append((df_nodes.iloc[i]['node_id'], df_nodes.iloc[j]['node_id']))
    return edges


print("\n=== Calcul des arêtes KNN ===")
t0 = time.time()
edges = get_knn_edges(df_nodes, k=5)
print(f"{len(edges)} arêtes générées.")
print(f"Temps de calcul : {time.time() - t0:.3f} s")


# --- Réduction de dimension ---
def reduce_dimensions(features, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca


print("\n=== Réduction de dimension (PCA) ===")
t0 = time.time()
features = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
reduced_features, pca = reduce_dimensions(features, n_components=10)
print(f"Réduction terminée : 50D → 10D")
print(f"Temps PCA : {time.time() - t0:.3f} s")


# --- Recherche sur features réduites ---
def naive_radius_search_reduced(df_nodes, df_queries, reduced_features, pca):
    results = []
    node_ids = df_nodes['node_id'].values
    for idx, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        radius = float(query['D'])
        A_vector_reduced = pca.transform([A_vector])[0]

        # Vectorisation aussi sur les features réduites
        dist = np.linalg.norm(reduced_features - A_vector_reduced, axis=1)
        found_mask = dist <= radius
        found_nodes = list(zip(node_ids[found_mask], dist[found_mask]))

        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    return results


print("\n=== Recherche PCA (10D) ===")
t0 = time.time()
all_results_reduced = naive_radius_search_reduced(df_nodes, df_queries, reduced_features, pca)
t1 = time.time()
print(f"Recherche PCA terminée en {t1 - t0:.3f} s")

# --- Comparaison des performances ---
naive_counts = [len(r['found_nodes']) for r in all_results]
pca_counts = [len(r['found_nodes']) for r in all_results_reduced]

plt.figure(figsize=(10, 5))
plt.plot(naive_counts, label='Recherche vectorisée (50D)')
plt.plot(pca_counts, label='Recherche PCA (10D)')
plt.xlabel('Index de la requête')
plt.ylabel('Nombre de nœuds trouvés')
plt.title('Comparaison du nombre de nœuds trouvés par requête')
plt.legend()
plt.show()
