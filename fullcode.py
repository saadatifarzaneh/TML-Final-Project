import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import EsmModel, EsmTokenizer
import matplotlib.pyplot as plt
from Bio import SeqIO

# Step 1: Load and Parse the Fasta File
def load_fasta(file_path):
    sequences = []
    ids = []
    for record in SeqIO.parse(file_path, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    return ids, sequences

file_path = "/Users/farzanehsaadati/UGA/TML/Project/pkfold_hs_curated.fa"  # Path to the Fasta file
protein_ids, protein_sequences = load_fasta(file_path)


# Generate protein embeddings using ESM2
class ProteinEmbeddingGenerator:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)[1:-1].numpy()  # Remove [CLS] and [SEP]

embedding_generator = ProteinEmbeddingGenerator()
embeddings = [embedding_generator.get_embedding(seq) for seq in protein_sequences]
embeddings = np.array(embeddings, dtype=object)

# Normalize embeddings
mean_embedding = np.mean([e.mean(axis=0) for e in embeddings], axis=0)
std_embedding = np.std([e.std(axis=0) for e in embeddings], axis=0)
normalized_embeddings = [(e - mean_embedding) / std_embedding for e in embeddings]

# Dataset and DataLoader
class ProteinDataset(Dataset):
    def __init__(self, embeddings, ids):
        self.embeddings = embeddings
        self.ids = ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)  # [seq_len, emb_dim]
        return embedding.transpose(0, 1), self.ids[idx]  # Transpose to [emb_dim, seq_len]

dataset = ProteinDataset(normalized_embeddings, protein_ids)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define CNN model
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, sequence_length, num_classes=2):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        
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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

embedding_dim = normalized_embeddings[0].shape[1]
sequence_length = normalized_embeddings[0].shape[0]
model = ProteinCNN(embedding_dim, sequence_length, num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN model
for epoch in range(10):
    model.train()
    for inputs, ids in dataloader:
        batch_size = inputs.size(0)
        targets = torch.randint(0, 2, (batch_size,), dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Generate saliency maps
def generate_saliency_map(model, inputs):
    model.eval()
    inputs.requires_grad_()
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()
    saliency = inputs.grad.abs()
    return saliency

sample_input, _ = dataset[0]
sample_input = sample_input.unsqueeze(0)
saliency_map = generate_saliency_map(model, sample_input)

# Normalize and visualize saliency map
saliency_map_normalized = saliency_map / torch.max(torch.abs(saliency_map))
plt.figure(figsize=(10, 5))
plt.imshow(saliency_map_normalized.squeeze().detach().numpy(), aspect='auto', cmap='viridis')
plt.colorbar(label="Saliency Value")
plt.title("Saliency Map for Protein Embedding")
plt.xlabel("Sequence Position")
plt.ylabel("Embedding Dimension")
plt.show()