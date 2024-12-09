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


# if __name__=='__main__':

#     # import torch

#     # # Example tensors
#     # x = torch.tensor([1, 2, 3, 4, 5])
#     # y = torch.tensor([3, 4])

#     # # Get the elements in x but not in y
#     # difference = x[~torch.isin(x, y)]
#     # print(difference)


#     # import torch

#     # # Example 3D tensors with shape (b, 4, L)
#     # b, L = 2, 5
#     # x = torch.randint(0, 10, (b, 4, L))  # Example tensor x
#     # y = torch.randint(0, 10, (b, 4, L))  # Example tensor y

#     # # Check which elements in `x` are not in `y` across each row (broadcasted comparison)
#     # difference_mask = ~torch.isin(x, y)

#     # # Apply mask to filter elements in x that are not in y
#     # difference = torch.where(difference_mask, x, torch.tensor(float('nan')))  # Fill in non-difference with NaNs

#     # print(f"Tensor x:\n{x}\n")
#     # print(f"Tensor y:\n{y}\n")
#     # print(f"Difference (x - y):\n{difference}\n")


#     # import torch
#     # import tqdm

#     # # Example one-hot encoded sequence of shape (b, 4, L)
#     # b=100000
#     # L=230
#     # # Example with 4 possible values, like DNA bases (A, C, G, T)
#     # one_hot_seq = torch.eye(4)[torch.randint(0, 4, (b, L))]  # Shape (b, L, 4)
#     # #print(one_hot_seq)

#     # # The lookup table: For instance, mapping [A, C, G, T]
#     # lookup_table = ['A', 'C', 'G', 'T']

#     # # Step 1: Get the index of the hot (1) element along the one-hot dimension (axis 2)
#     # indices = torch.argmax(one_hot_seq, dim=2)  # Shape (b, L)
#     # #print(indices)

#     # # Step 2: Convert indices to characters using vectorized indexing
#     # # Use a list comprehension and join the strings per batch
#     # sequences = [''.join([lookup_table[idx] for idx in batch]) for batch in indices]
#     # print(sequences)

#     # # Output the result
#     # for i, seq in tqdm.tqdm(enumerate(sequences)):
#     #     print(f"Batch {i} sequence: {seq}")

#     import torch
#     import datetime

#     # Example one-hot encoded sequence of shape (b, 4, L)
#     b=100000
#     b1=1000
#     #
#     # b=10
#     # b1=7

#     L=230

#     def random_ohe(b):
#         one_hot_seq = torch.eye(4)[torch.randint(0, 4, (b, L))].permute(0, 2, 1)  # Reshape to (b, 4, L)mask = torch.rand(b, L) < 0.2
#         mask = torch.rand(b, L) < 0.2
#         mask = mask.unsqueeze(1).expand(-1, 4, -1)  # Expand mask to shape (b, 4, L)
#         one_hot_seq[mask] = 0
#         return one_hot_seq

#     one_hot_seq=random_ohe(b)
#     one_hot_seq1=random_ohe(b1)

#     pool=[1,2,6,7,9,1]
#     #pool=list(set(pool)-set(pool))
#     #print(pool)
#     xtrain=[1,2,3,4,5,6]
#     pool=list(set(pool)-set(xtrain))
#     print(pool)

#     print(datetime.datetime.now())
#     sequences=lookup_ohe_to_dna(one_hot_seq)
#     print(datetime.datetime.now())
#     sequences1=lookup_ohe_to_dna(one_hot_seq1)
#     print(type(sequences),len(sequences),one_hot_seq.shape)
#     #print(sequences[0])
#     print(sequences[0:3])
#     print(datetime.datetime.now())
#     final_tensor=lookup_dna_to_ohe(sequences)
#     #print(final_tensor)
#     print(f"{final_tensor.shape=}")
#     print(datetime.datetime.now())


#     #import is_seq_in_xtrain
#     import numpy as np

#     proposed_X=torch.tensor(np.array(lookup_dna_to_ohe(list(set(lookup_ohe_to_dna(one_hot_seq))-set(lookup_ohe_to_dna(one_hot_seq1))))),dtype=torch.float32)
#     print(f"{type(proposed_X)=}")
#     print(f"{proposed_X.shape=}")
#     print(datetime.datetime.now())

#     # print("0")
#     # one=lookup_ohe_to_dna(one_hot_seq)
#     # print("1")
#     # two=lookup_ohe_to_dna(one_hot_seq1)
#     # print("2")
#     # three=set(one)
#     # print("3")
#     # four=set(two)
#     # print("4")
#     # five=list(three-four)
#     # print("5")
#     # print(five)
#     # proposed_X=torch.tensor(np.array(five)) 
#     # print("6")

#     # Output the result
#     #for i, seq in enumerate(sequences):
#     #    print(f"Batch {i} sequence: {seq}")

#     print("SCRIPT END")





if __name__=='__main__':

    import torch
    import datetime

    b=3
    b1=5

    L=7

    def random_ohe(b):
        one_hot_seq = torch.eye(4)[torch.randint(0, 4, (b, L))].permute(0, 2, 1)  # Reshape to (b, 4, L)mask = torch.rand(b, L) < 0.2
        mask = torch.rand(b, L) < 0.2
        mask = mask.unsqueeze(1).expand(-1, 4, -1)  # Expand mask to shape (b, 4, L)
        one_hot_seq[mask] = 0
        return one_hot_seq

    one_hot_seq=random_ohe(b)
    one_hot_seq1=random_ohe(b1)

    sequences=lookup_ohe_to_dna(one_hot_seq)
    sequences1=lookup_ohe_to_dna(one_hot_seq1)

    sequences1[0]=sequences[0]

    print(f"{sequences=}")
    print(f"{sequences1=}")

    x=lookup_dna_to_ohe(sequences)
    x1=lookup_dna_to_ohe(sequences1)

    print(f"{list(set(sequences)-set(sequences1))=}")

    import numpy as np

    proposed_X=torch.tensor(np.array(lookup_dna_to_ohe(list(set(lookup_ohe_to_dna(x))-set(lookup_ohe_to_dna(x1))))),dtype=torch.float32) 
    print(f"{proposed_X.shape=}")

    # proposed_X=torch.tensor(np.array(lookup_dna_to_ohe(list(set(lookup_ohe_to_dna(one_hot_seq))-set(lookup_ohe_to_dna(one_hot_seq1))))),dtype=torch.float32)
    # print(f"{type(proposed_X)=}")
    # print(f"{proposed_X.shape=}")

    print("SCRIPT END")