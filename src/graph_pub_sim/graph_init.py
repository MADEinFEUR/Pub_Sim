import pandas as pd

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import importlib.util
import os

module_name = "graph_template"
module_path = os.path.join(os.path.dirname(__file__), "graph_template.py")
spec = importlib.util.spec_from_file_location(module_name, module_path)
graph_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_template)






# --- Chargement des données ---
df_nodes = pd.read_csv('Data/adsSim_data_nodes.csv')
df_queries = pd.read_csv('Data/queries_structured.csv')

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

G = build_graph(df_nodes, k=5)

# --- Fonction de distance pondérée ---
def weighted_distance(x, y, w):
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)
    return np.sqrt(np.sum(w * (x - y) ** 2))

# --- Recherche naïve des nœuds dans le rayon pondéré ---
def naive_radius_search(df_nodes, df_queries):
    results = []
    features_cols = [f'feature_{i}' for i in range(1, 51)]
    for _, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])
        found_nodes = []
        for _, row in df_nodes.iterrows():
            node_id = row['node_id']
            x_B = row[features_cols].values.astype(float)
            dist = weighted_distance(A_vector, x_B, w_vector)
            if dist <= radius:
                found_nodes.append((node_id, dist))
        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    return results

all_results = naive_radius_search(df_nodes, df_queries)

# --- Affichage des résultats ---
for result in all_results:
    print(f"Requête {result['query']} (rayon {result['radius']}): {result['found_nodes']}")

# --- Construction des arêtes pour visualisation ou analyse ---
def get_knn_edges(df_nodes, k=5):
    X = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # on saute le premier (soi-même)
            edges.append((df_nodes.iloc[i]['node_id'], df_nodes.iloc[j]['node_id']))
    return edges

edges = get_knn_edges(df_nodes, k=5)
print("Exemple d'arêtes :", edges[:10])

from sklearn.decomposition import PCA

# --- Réduction de dimension ---

def reduce_dimensions(features, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

features = df_nodes[[f'feature_{i}' for i in range(1, 51)]].values
reduced_features, pca = reduce_dimensions(features, n_components=10)

# --- Nouvelle recherche naïve sur features réduites ---
def naive_radius_search_reduced(df_nodes, df_queries, reduced_features, pca):
    results = []
    for idx, query in df_queries.iterrows():
        A_vector = np.array([float(v) for v in query['A_vector'].split(';')])
        w_vector = np.array([float(v) for v in query['Y_vector'].split(';')])
        radius = float(query['D'])
        # Réduire la dimension du point de requête
        A_vector_reduced = pca.transform([A_vector])[0]
        found_nodes = []
        for i, row in df_nodes.iterrows():
            x_B_reduced = reduced_features[i]
            # On peut garder la pondération uniforme ou réduire w_vector aussi
            dist = np.linalg.norm(A_vector_reduced - x_B_reduced)
            if dist <= radius:
                found_nodes.append((row['node_id'], dist))
        results.append({
            'query': query['point_A'],
            'radius': radius,
            'found_nodes': found_nodes
        })
    return results

all_results_reduced = naive_radius_search_reduced(df_nodes, df_queries, reduced_features, pca)

import matplotlib.pyplot as plt

# Suppose que tu as déjà exécuté naive_radius_search et naive_radius_search_reduced
# et que tu as all_results (50D) et all_results_reduced (PCA 10D)

# Nombre de nœuds trouvés par requête
naive_counts = [len(r['found_nodes']) for r in all_results]
pca_counts = [len(r['found_nodes']) for r in all_results_reduced]

plt.figure(figsize=(10, 5))
plt.plot(naive_counts, label='Recherche naïve (50D)')
plt.plot(pca_counts, label='Recherche PCA (10D)')
plt.xlabel('Index de la requête')
plt.ylabel('Nombre de nœuds trouvés')
plt.title('Comparaison du nombre de nœuds trouvés par requête')
plt.legend()
plt.show()




# Distribution des distances pour la première requête (naïf)
distances_naive = [dist for _, dist in all_results[77]['found_nodes']]
plt.hist(distances_naive, bins=20, alpha=0.7, label='Naïf (50D)')

# Distribution des distances pour la première requête (PCA)
distances_pca = [dist for _, dist in all_results_reduced[77]['found_nodes']]
plt.hist(distances_pca, bins=20, alpha=0.7, label='PCA (10D)')

plt.xlabel('Distance pondérée')
plt.ylabel('Nombre de nœuds')
plt.title('Distribution des distances pour la première requête')
plt.legend()
plt.show()
# Utilisation de la fonction
graph_template.visualize_graph(G)

