import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import esm
import torch.nn.functional as F
from Bio import SeqIO

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)
model.eval()

# Protein sequence dataset class for FASTA file
class ProteinSequenceDataset(Dataset):
    def __init__(self, fasta_file):
        self.records = list(SeqIO.parse(fasta_file, "fasta"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        seq = str(record.seq)
        id = record.id
        return seq, id

# Batch collate function for protein sequences
def protein_collate_batch(batch):
    seqs, ids = zip(*batch)
    batch_labels, batch_strs, batch_tokens = alphabet.get_batch_converter()(list(zip(ids, seqs)))
    lengths = (batch_tokens != alphabet.padding_idx).sum(1)
    return batch_tokens.to(DEVICE), lengths, ids

# Load the data from FASTA file
fasta_file = 'pkfold_hs_curated.fa'
dataset = ProteinSequenceDataset(fasta_file)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=protein_collate_batch)

# Process protein data
def process_protein_data(data_loader, fixed_dim=1024):
    sequence_representations = []
    ids_list = []
    for batch_tokens, batch_lens, ids in data_loader:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        for i, tokens_len in enumerate(batch_lens):
            seq_rep = token_representations[i, 1 : tokens_len - 1]
            pad_size = fixed_dim - seq_rep.size(0)
            if pad_size > 0:
                seq_rep = F.pad(seq_rep, (0, 0, 0, pad_size), "constant", 0)
            else:
                seq_rep = seq_rep[:fixed_dim]
            sequence_representations.append(seq_rep.cpu())  # Keeping as tensor for saving
            ids_list.append(ids[i])

    return ids_list, sequence_representations

# Generate embeddings
ids_list, sequence_representations = process_protein_data(data_loader)

# Save to .pt format
torch.save({'Protein_ID': ids_list, 'Embedding': sequence_representations}, 'protein_embeddings.pt')
