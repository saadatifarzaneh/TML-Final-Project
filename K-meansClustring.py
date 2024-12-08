import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# Load protein embeddings data
data = torch.load('protein_embeddings.pt')
ids_list = data['Protein_ID']
sequence_representations = data['Embedding']

# Convert sequence representations if they are a list of tensors
if isinstance(sequence_representations, list):
    sequence_representations = torch.stack(sequence_representations)

# Average each sequence representation across the sequence length dimension
sequence_representations = sequence_representations.mean(dim=1).cpu().numpy()

# Perform K-means clustering with 26 clusters
n_clusters = 26
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(sequence_representations)


# Apply t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(sequence_representations)

# Define a fixed color palette for 26 clusters
colors = cm.tab20.colors + cm.tab20b.colors[:6]  # Fixed palette for 26 colors

# Plotting the t-SNE result with K-means cluster colors
plt.figure(figsize=(12, 10))
for cluster_id in range(n_clusters):
    # Filter data points that belong to the current cluster
    cluster_points = embeddings_2d[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster_id], alpha=0.7, label=f"Cluster {cluster_id + 1}")

# Create legend with unique cluster colors, positioned beside the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="K-means Clusters", fontsize='small')

plt.title("t-SNE Visualization of Protein Embeddings by K-means Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
plt.show()
