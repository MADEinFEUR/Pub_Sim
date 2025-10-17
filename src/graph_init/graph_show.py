import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# === 1. Lecture des CSV ===
nodes_df = pd.read_csv("Data/adsSim_data_nodes.csv")  
queries_df = pd.read_csv("Data/queries_structured.csv")

# === 2. Nettoyage des colonnes ===
nodes_df.columns = nodes_df.columns.str.strip().str.lower()
queries_df.columns = queries_df.columns.str.strip().str.lower()
nodes_df["node_id"] = nodes_df["node_id"].str.strip().str.lower()
queries_df["point_a"] = queries_df["point_a"].str.strip().str.lower()

# === 3. Conversion de Y_vector ===
queries_df["y_vector"] = queries_df["y_vector"].apply(lambda s: np.array([float(v) for v in s.split(';')]))

# === 4. Mapping ads_X -> node_X pour fusion ===
queries_df["node_id_match"] = queries_df["point_a"].str.replace("ads", "node")

# === 5. Fusion ===
merged_df = queries_df.merge(nodes_df, left_on="node_id_match", right_on="node_id", how="left")

# === 6. Calcul de similarité cosinus ===
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

merged_df["similarity"] = merged_df.apply(
    lambda row: cosine_similarity(
        row["y_vector"],
        np.array([row[f"feature_{i}"] for i in range(1, 51)])
    ) if pd.notnull(row["node_id"]) else np.nan,
    axis=1
)

# === 7. PCA 2D pour positions du graphe 2D ===
features = nodes_df[[f"feature_{i}" for i in range(1, 51)]]
pca_2d = PCA(n_components=2)
coords_2d = pca_2d.fit_transform(features)
nodes_df["x2d"] = coords_2d[:, 0]
nodes_df["y2d"] = coords_2d[:, 1]

# === 8. PCA 3D pour positions du graphe 3D ===
pca_3d = PCA(n_components=3)
coords_3d = pca_3d.fit_transform(features)
nodes_df["x3d"], nodes_df["y3d"], nodes_df["z3d"] = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]

# === 9. Création du graphe NetworkX ===
G = nx.Graph()
positions_2d = {row["node_id"]: (row["x2d"], row["y2d"]) for _, row in nodes_df.iterrows()}

# Ajouter les noeuds
for _, row in nodes_df.iterrows():
    G.add_node(row["node_id"], cluster=row["cluster_id"])

# Optionnel : connecter seulement les k voisins les plus proches (pour grands datasets)
k_neighbors = 5
nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(features)
distances, indices = nbrs.kneighbors(features)
node_ids = nodes_df["node_id"].tolist()
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:  # ignorer le noeud lui-même
        G.add_edge(node_ids[i], node_ids[j])

# === 10. Visualisation 2D ===
plt.figure(figsize=(12, 10))
colors = [G.nodes[n]["cluster"] for n in G.nodes()]
nx.draw(G, pos=positions_2d, with_labels=False, node_size=50, node_color=colors, cmap=plt.cm.tab20, edge_color='gray')

# Ajouter les requêtes sur le graphe 2D
for _, row in merged_df.iterrows():
    if pd.notnull(row["node_id"]):
        plt.scatter(row["y_vector"][0], row["y_vector"][1], c='red', s=40)

plt.title("Graphe 2D des nœuds avec requêtes")
plt.show()

# === 11. Visualisation 3D ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Couleurs selon cluster
clusters = nodes_df["cluster_id"].astype(int)
colors_3d = plt.cm.tab20(clusters / clusters.max())
ax.scatter(nodes_df["x3d"], nodes_df["y3d"], nodes_df["z3d"], c=colors_3d, s=50, depthshade=True)

# Ajouter les requêtes dans le graphe 3D (les 3 premières dimensions de Y_vector)
for _, row in merged_df.iterrows():
    if pd.notnull(row["node_id"]):
        ax.scatter(row["y_vector"][0], row["y_vector"][1], row["y_vector"][2], c='red', s=40)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("Graphe 3D des nœuds avec requêtes")
plt.show()
