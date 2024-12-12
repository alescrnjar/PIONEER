import torch


def lookup_ohe_to_dna(one_hot_seq):
    #if type(one_hot_seq)!=torch.tensor:
    one_hot_seq=torch.tensor(one_hot_seq)
    lookup_table = ['A', 'C', 'G', 'T', 'N']
    #indices = torch.argmax(one_hot_seq, dim=2)  # Shape (b, L)
    indices = torch.argmax(one_hot_seq, dim=1) 
    #all_zero_mask = torch.sum(one_hot_seq, dim=2) == 0
    all_zero_mask = torch.sum(one_hot_seq, dim=1) == 0
    indices[all_zero_mask] = 4
    return [''.join([lookup_table[idx] for idx in batch]) for batch in indices]

def lookup_dna_to_ohe(sequences):
    lookup_table = ['A', 'C', 'G', 'T', 'N']
    char_to_index = {char: idx for idx, char in enumerate(lookup_table)}
    indexed_sequences = [[char_to_index[char] for char in seq] for seq in sequences]     # Convert the sequences into a list of indices using the lookup table
    index_tensor = torch.tensor(indexed_sequences, dtype=torch.long)  # Shape (b, L)     # Convert the indexed sequences to a PyTorch tensor (shape: b, L)
    one_hot_tensor = torch.nn.functional.one_hot(index_tensor, num_classes=len(lookup_table))  # Shape (b, L, 5) Step 1: Use one-hot encoding to convert to shape (b, L, 4) # we need to ensure that 'N' is handled as a 0-filled vector
    one_hot_acgt = one_hot_tensor[:, :, :4]  # Shape (b, L, 4)     # Step 2: Remove the last dimension ('N') to get a (b, L, 4) shape for A, C, G, T, and keep track of 'N' separately.
    one_hot_acgt = one_hot_acgt.permute(0, 2, 1)  # Shape (b, 4, L) # Step 3: Permute to get the desired output shape (b, 4, L)
    return one_hot_acgt

