import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# 1. Load Data
data = pd.DataFrame({
    "Category Word (A)": [
        "cashier", "cashier", "cashier", "staff", "staff", "staff",
        "service", "service", "service", "service", "staff", "price", "price",
        "product", "product", "product", "selection", "selection",
        "shelves", "shelves", "shelves", "store", "store", "store", "store",
        "parking", "parking", "parking", "checkout", "checkout"
    ],
    "Sentiment Word (B)": [
        "rude", "disappoint", "friendly", "discrimination", "rude", "helpful",
        "poor", "excellent", "helpful", "good", "friendly", "expensive", "good",
        "rotten", "expired", "nice", "limited", "nice", "empty", "rotten", "clean",
        "dirty", "clean", "limited", "good", "limited", "excellent", "crowded",
        "crowded", "helpful"
    ],
    "Lift": [
        2.8125, 5.0625, 3.375, 0, 2.142857143, 3.673469388,
        0, 1.928571429, 5.510204082, 0.665024631, 5.142857143,
        1.35, 1.862068966, 0, 6.75, 2.25, 6.230769231, 3.461538462,
        27.0, 9.0, 0, 3.461538462, 0.923076923, 1.846153846, 1.312997347,
        3.0, 3.0, 3.0, 4.5, 6.428571429
    ]
})

# 2. Build Unique Word List and Dissimilarity Matrix
unique_words = sorted(list(set(data["Category Word (A)"]).union(set(data["Sentiment Word (B)"]))))
n_words = len(unique_words)
word_to_idx = {word: i for i, word in enumerate(unique_words)}

# Initialize distance matrix with a high value (representing unassociated pairs)
penalty_distance = 10.0
dissimilarity_matrix = np.full((n_words, n_words), penalty_distance)
np.fill_diagonal(dissimilarity_matrix, 0.0)

# Fill matrix using 1 / Lift for valid lift values > 0
for _, row in data.iterrows():
    wA, wB, lift = row["Category Word (A)"], row["Sentiment Word (B)"], row["Lift"]
    idxA, idxB = word_to_idx[wA], word_to_idx[wB]
    
    if lift > 0:
        dist = 1.0 / lift
        dissimilarity_matrix[idxA, idxB] = dist
        dissimilarity_matrix[idxB, idxA] = dist

# 3. Multidimensional Scaling (MDS)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10)
coords = mds.fit_transform(dissimilarity_matrix)

df_mds = pd.DataFrame(coords, columns=["Dim1", "Dim2"], index=unique_words)

# 4. Define Lexicon Polarity for Clustering Anchors
pos_words = {"friendly", "helpful", "excellent", "good", "nice", "clean"}
neg_words = {"rude", "disappoint", "discrimination", "poor", "expensive", "rotten", "expired", "limited", "empty", "dirty", "crowded"}

def get_word_type(word):
    if word in pos_words:
        return "Positive Sentiment"
    elif word in neg_words:
        return "Negative Sentiment"
    else:
        return "Category Aspect"

df_mds["Type"] = [get_word_type(w) for w in df_mds.index]

# Compute centroids for Positive and Negative sentiment anchors in 2D space
pos_centroid = df_mds[df_mds["Type"] == "Positive Sentiment"][["Dim1", "Dim2"]].mean().values
neg_centroid = df_mds[df_mds["Type"] == "Negative Sentiment"][["Dim1", "Dim2"]].mean().values

# Assign every point to the nearest Sentiment Pole
def assign_cluster(row):
    point = row[["Dim1", "Dim2"]].values
    dist_pos = np.linalg.norm(point - pos_centroid)
    dist_neg = np.linalg.norm(point - neg_centroid)
    return "Positive Cluster" if dist_pos < dist_neg else "Negative Cluster"

df_mds["Cluster"] = df_mds.apply(assign_cluster, axis=1)

# 5. Visualizing the MDS Plot
plt.figure(figsize=(12, 9))

# Plot cluster background decision regions / hulls
colors = {"Positive Sentiment": "#2ca02c", "Negative Sentiment": "#d62728", "Category Aspect": "#1f77b4"}
markers = {"Positive Sentiment": "^", "Negative Sentiment": "v", "Category Aspect": "o"}

for word_type in ["Category Aspect", "Positive Sentiment", "Negative Sentiment"]:
    subset = df_mds[df_mds["Type"] == word_type]
    plt.scatter(
        subset["Dim1"], subset["Dim2"],
        c=colors[word_type],
        marker=markers[word_type],
        s=120 if word_type != "Category Aspect" else 80,
        label=word_type,
        edgecolors='k',
        alpha=0.85
    )

# Draw lines connecting category words to their assigned sentiment centroid
for word, row in df_mds[df_mds["Type"] == "Category Aspect"].iterrows():
    target_centroid = pos_centroid if row["Cluster"] == "Positive Cluster" else neg_centroid
    line_color = "#2ca02c" if row["Cluster"] == "Positive Cluster" else "#d62728"
    plt.plot([row["Dim1"], target_centroid[0]], [row["Dim2"], target_centroid[1]], 
             linestyle="--", color=line_color, alpha=0.3)

# Annotate word labels
for word, row in df_mds.iterrows():
    plt.annotate(
        word,
        (row["Dim1"], row["Dim2"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=10,
        fontweight='bold' if row["Type"] != "Category Aspect" else 'normal'
    )

# Plot Centroids
plt.scatter(pos_centroid[0], pos_centroid[1], color='#2ca02c', s=300, marker='X', label='Pos Centroid', edgecolors='black')
plt.scatter(neg_centroid[0], neg_centroid[1], color='#d62728', s=300, marker='X', label='Neg Centroid', edgecolors='black')

plt.title("MDS Perceptual Map & Sentiment Pole Clustering (1 / Lift Distance)", fontsize=14, fontweight='bold')
plt.xlabel("MDS Dimension 1", fontsize=11)
plt.ylabel("MDS Dimension 2", fontsize=11)
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()