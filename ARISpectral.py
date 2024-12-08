import torch
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

# Load protein embeddings data
data = torch.load('protein_embeddings.pt')
ids_list = data['Protein_ID']
sequence_representations = data['Embedding']

# Load protein families and subfamilies from the .fa file
family_map = {}
with open('pkfold_hs_curated.fa', 'r') as f:
    for line in f:
        if line.startswith('>'):
            parts = line[1:].strip().split()
            protein_id = parts[0]
            full_family = parts[2]
            subfamily_name = full_family.split('_')[1] if len(full_family.split('_')) > 1 else "Unknown"
            family_map[protein_id] = subfamily_name

# Generate true labels (subfamily labels)
true_labels = [family_map.get(protein_id, "Unknown") for protein_id in ids_list]

# Convert sequence representations if they are a list of tensors
if isinstance(sequence_representations, list):
    sequence_representations = torch.stack(sequence_representations)

# Average each sequence representation across the sequence length dimension
sequence_representations = sequence_representations.mean(dim=1).cpu().numpy()

# Perform Spectral Clustering with 26 clusters
n_clusters = 26
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels = spectral.fit_predict(sequence_representations)

# Compute the Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(true_labels, cluster_labels)
print(f"Adjusted Rand Index (ARI) between true subfamilies and Spectral clusters: {ari_score:.4f}")
