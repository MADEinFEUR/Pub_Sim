# 🔍 Recherche de similarité publicitaire par clustering vectorisé (50D)

Ce projet implémente une approche de **recherche de similarité pondérée** dans un espace publicitaire à haute dimension (50 dimensions), optimisée par **clustering KMeans** et **vectorisation NumPy**.  
L’objectif est d’identifier rapidement les profils publicitaires similaires à un point de requête donné, tout en minimisant le temps de traitement.

---

## Installation

### 1️ Prérequis
- Python **3.9+**
- `pip` installé sur votre machine

### 2️ Cloner le dépôt
```bash
git clone https://github.com/ton-utilisateur/ton-projet.git
cd ton-projet
```

### 3️ Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Structure du projet

```
├── Data/
│   ├── adsSim_data_nodes.csv          # Données des nœuds publicitaires
│   ├── queries_structured.csv         # Données des requêtes
│
├── src/
│   ├── TD_pub_sim_A3MSI_2.py      # Script principal
│
├── requirements.txt
└── README.md
```

---

## Exécution du script

Depuis la racine du projet :
```bash
python src/TD_pub_sim_A3MSI_2.py
```

Le script :
1. Charge les données publicitaires et les requêtes ;
2. Construit un graphe pondéré entre les nœuds ;
3. Applique un **clustering KMeans** sur les 50 dimensions ;
4. Exécute une recherche **vectorisée et pondérée** dans les clusters pertinents ;
5. Affiche :
   - Le **temps d’exécution total** de la recherche ;
   - Un **graphique du nombre de nœuds trouvés** par requête.

---

##  Résultats attendus

- Un affichage console indiquant le temps de traitement :
  ```
  Temps recherche clustering vectorisée 50D : 1.532 s
  ```
- Un graphique Matplotlib présentant le nombre de nœuds trouvés pour chaque requête.

---

## Limites et améliorations possibles

- Le modèle repose sur une distance pondérée sensible au choix des coefficients `Y`.
- La dynamique temporelle des profils n’est pas prise en compte.
- La similarité est purement géométrique.

**Améliorations possibles :**
- Apprentissage automatique des poids `Y` à partir de données d’engagement (clics, conversions, etc.)
- Intégration de la dimension temporelle.
- Exploration de méthodes non linéaires (UMAP, autoencodeurs).

---

## Licence
Ce projet est distribué sous licence Apache.  
Vous êtes libre de le réutiliser, le modifier et le redistribuer sous réserve de mentionner l’auteur original.
