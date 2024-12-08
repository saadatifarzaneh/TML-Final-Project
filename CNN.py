import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load protein embeddings
data = torch.load("/Users/farzanehsaadati/UGA/TML/Project/protein_embeddings.pt")
protein_ids = list(data['Protein_ID'])  # Convert Protein_ID to a Python list
embeddings = data['Embedding']

# Ensure embeddings are in NumPy format
if isinstance(embeddings, list):
    embeddings = np.array(embeddings)
elif isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.numpy()

assert len(protein_ids) == embeddings.shape[0], "Mismatch between IDs and embeddings"

# Dataset and DataLoader
class ProteinDataset(Dataset):
    def __init__(self, embeddings, ids):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx].transpose(1, 0)  # Transpose for CNN: [seq_len, emb_dim] -> [emb_dim, seq_len]
        return embedding, self.ids[idx]

dataset = ProteinDataset(embeddings, protein_ids)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# CNN Model
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, sequence_length, num_classes=2):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)  # Slightly reduced dropout
        self._compute_flattened_size(embedding_dim, sequence_length)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_flattened_size(self, embedding_dim, sequence_length):
        dummy_input = torch.zeros(1, embedding_dim, sequence_length)
        x = torch.relu(self.bn1(self.conv1(dummy_input)))
        x = torch.relu(self.bn2(self.conv2(x)))
        self.flattened_size = x.numel()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

sequence_length = embeddings.shape[1]
embedding_dim = embeddings.shape[2]
num_classes = 2  # Example: binary classification
model = ProteinCNN(embedding_dim=embedding_dim, sequence_length=sequence_length, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(20):  # Extended epochs for better convergence
    model.train()
    for inputs, ids in dataloader:
        batch_size = inputs.size(0)
        targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)  # Example dummy targets
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

def integrated_gradients(model, inputs, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    inputs.requires_grad_()  # Ensure inputs allow gradient computation

    # Generate scaled inputs between baseline and actual input
    scaled_inputs = torch.stack([
        baseline + (float(i) / steps) * (inputs - baseline)
        for i in range(steps + 1)
    ])

    grads = []  # Store gradients for each scaled input
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)  # Re-attach computation graph
        outputs = model(scaled_input)
        max_class = outputs.max(dim=1).values.sum()  # Target max class output
        model.zero_grad()  # Clear previous gradients
        max_class.backward()  # Compute gradients
        grads.append(scaled_input.grad.clone())  # Save gradients

    grads = torch.stack(grads)  # Combine gradients
    avg_grads = grads.mean(dim=0)  # Average gradients across all steps
    integrated_grads = (inputs - baseline) * avg_grads  # Calculate integrated gradients

    return integrated_grads

# Visualization
def visualize_saliency_map_with_regions(saliency_map, title="Enhanced Saliency Map"):
    # Normalize the saliency map
    saliency_map = saliency_map.squeeze().detach().numpy()
    saliency_map_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)

    # Highlight specific regions (e.g., enhance values in a certain range)
    enhanced_map = saliency_map_normalized.copy()
    enhanced_map[200:400, 300:500] *= 2  # Example: Amplify values in a region

    # Plot with the enhanced colormap
    plt.figure(figsize=(14, 8))
    plt.imshow(
        enhanced_map,
        aspect='auto',
        cmap='plasma',  # Distinct colormap for better differentiation
        interpolation='nearest'
    )
    plt.colorbar(label="Saliency Value")
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Embedding Dimension")
    plt.tight_layout()
    plt.show()



