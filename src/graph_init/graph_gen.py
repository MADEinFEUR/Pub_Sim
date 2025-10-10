import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Nom du fichier CSV
fichier_csv = "Data/adsSim_data_nodes.csv"

# Lire le CSV avec pandas
df = pd.read_csv(fichier_csv)

# Sélectionner les colonnes de features
features = df[[f"feature_{i}" for i in range(1, 51)]]

# Réduire les dimensions à 2D avec PCA
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(features)

# Ajouter les coordonnées 2D au DataFrame
df['x'] = coords_2d[:, 0]
df['y'] = coords_2d[:, 1]

# Créer un graphe vide
G = nx.Graph()
positions = {}

# Ajouter les noeuds avec leurs positions 2Ds
for _, row in df.iterrows():
    node_id = row['node_id']
    G.add_node(node_id, cluster=row['cluster_id'])
    positions[node_id] = (row['x'], row['y'])

# Ajouter des arêtes entre les noeuds du même cluster
for cluster in df['cluster_id'].unique():
    cluster_nodes = df[df['cluster_id'] == cluster]['node_id'].tolist()
    for i in range(len(cluster_nodes)):
        for j in range(i + 1, len(cluster_nodes)):
            G.add_edge(cluster_nodes[i], cluster_nodes[j])

# Dessiner le graphe avec couleur selon le cluster
plt.figure(figsize=(10, 8))
colors = [G.nodes[node]['cluster'] for node in G.nodes()]
nx.draw(
    G, pos=positions, with_labels=True,
    node_size=300, node_color=colors, cmap=plt.cm.tab20, edge_color='gray'
)
plt.title("Graphe 2D depuis CSV avec PCA")
plt.show()
