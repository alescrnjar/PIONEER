# https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py
import torch
import numpy as np
import augment

def _sample_aug_combos(batch_size=1, hard_aug=True, max_augs_per_seq=1, max_num_aug=1): 
    """Set the number of augmentations and randomly select augmentations to apply
    to each sequence.
    """
    # determine the number of augmentations per sequence
    if hard_aug:
        batch_num_aug = max_augs_per_seq * np.ones((batch_size,), dtype=int)
    else:
        batch_num_aug = np.random.randint(1, max_augs_per_seq + 1, (batch_size,))

    # randomly choose which subset of augmentations from augment_list
    aug_combos = [ list(sorted(np.random.choice(max_num_aug, sample, replace=False))) for sample in batch_num_aug ] #[[0, 2], [2, 4], [1, 3], [0, 3], [1, 4], [1, 4], [0, 1], ... x batch_size (if e.g. max_num_aug=5 and max_aug_per_seq=2)
    return aug_combos

def augment_max_len(augment_list):
    """Determine whether insertions are applied to determine the insert_max,
    which will be applied to pad other sequences with random DNA.

    Parameters
    ----------
    augment_list : list
        List of augmentations.

    Returns
    -------
    int
        Value for insert max.
    """
    insert_max = 0
    for augment in augment_list:
        if hasattr(augment, 'insert_max'):
            insert_max = augment.insert_max
    return insert_max

def _pad_end(x, insert_max):
    """Add random DNA padding of length insert_max to the end of each sequence in batch."""
    N, A, L = x.shape
    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    padding = torch.stack([a[p.multinomial(insert_max, replacement=True)].transpose(0,1) for _ in range(N)]).to(x.device)
    x_padded = torch.cat( [x, padding.to(x.device)], dim=2 )
    return x_padded

def _apply_augment(x, augment_list, want_pad=True, hard_aug=True, max_augs_per_seq=1, max_num_aug=1):
    insert_max=augment_max_len(augment_list)
    """Apply augmentations to each sequence in batch, x."""
    # number of augmentations per sequence
    #aug_combos = _sample_aug_combos(x.shape[0]) #AC orig
    aug_combos = _sample_aug_combos(batch_size=x.shape[0], hard_aug=hard_aug, max_augs_per_seq=max_augs_per_seq, max_num_aug=max_num_aug)

    # apply augmentation combination to sequences
    x_new = []
    for aug_indices, seq in zip(aug_combos, x):
        seq = torch.unsqueeze(seq, dim=0)
        insert_status = True   # status to see if random DNA padding is needed
        for aug_index in aug_indices:
            seq = augment_list[aug_index](seq)
            if hasattr(augment_list[aug_index], 'insert_max'):
                insert_status = False
        if want_pad: # QUIQUIURG grep "which will be applied to pad other sequences with random DNA." so should always be FALSE???
            if insert_status:
                if insert_max:
                    #print(f"pre pad: {FUNCTIONS_4_DALdna.ohe_to_seq(seq.squeeze(0), four_zeros_ok=True)}")
                    seq = _pad_end(seq, insert_max)
                    #print(f"postpad: {FUNCTIONS_4_DALdna.ohe_to_seq(seq.squeeze(0), four_zeros_ok=True)}")
        x_new.append(seq)
    return torch.cat(x_new)

def random_ohe_seq(seq_len):
    indices = torch.randint(0, 4, size=(1, 1, seq_len))  # Assuming 4 classes and 7 elements
    one_hot = torch.zeros(1, 4, seq_len)  # Shape: (1, 4, 7)
    one_hot.scatter_(1, indices, 1)
    return one_hot



#######################################

if __name__=='__main__':

    import FUNCTIONS_4_DALdna
    
    # https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf#scrollTo=WOS4yxXwWrxN
    
    """
    augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    #augment.RandomRC(rc_prob=0.5),
    augment.RandomInsertion(insert_min=0, insert_max=20),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomNoise(noise_mean=0, noise_std=0.3),
    augment.RandomMutation(mutate_frac=0.05),
    ]
    """

    #want_pad=True
    want_pad=False #QUIQUIURG  grep "which will be applied to pad other sequences with random DNA." so should always be FALSE???

    augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=10), #25% of 39 approx to 40
    #augment.RandomRC(rc_prob=0.5),
    augment.RandomInsertion(insert_min=0, insert_max=10),
    augment.RandomTranslocation(shift_min=0, shift_max=5),
    augment.RandomNoise(noise_mean=0, noise_std=0.3),
    augment.RandomMutation(mutate_frac=0.05),
    ]

    np.random.seed(41)
    torch.manual_seed(41)

    """
    for _ in range(5):
        x=random_ohe_seq(seq_len=39)
        print(f"{x.shape=}")
        for j in [0,1,-2,-1]:
            x[:,:,j]=torch.zeros(4)
        #print(f"{x.shape=}")
        dna=FUNCTIONS_4_DALdna.ohe_to_seq(x.squeeze(0), four_zeros_ok=True)
        print(f"{dna}")
        x_new=_apply_augment(x, augment_list, want_pad=want_pad)
        #print(f"{x_new=}")
        dna_new=FUNCTIONS_4_DALdna.ohe_to_seq(x_new.squeeze(0), four_zeros_ok=True)
        print(f"{dna_new}")
        print(f"{dna==dna_new=}")
        print()
    """

    #exit()

    X=torch.empty(0)
    for i in range(200):
        x=random_ohe_seq(seq_len=39)
        X=torch.cat((X,x),axis=0)
    print(f"{X.shape=}")
    #dna=FUNCTIONS_4_DALdna.ohe_to_seq(x.squeeze(0), four_zeros_ok=True)
    #print(f"{dna}")
    X_new=_apply_augment(X, augment_list, want_pad=want_pad)
    print(f"{X_new.shape=}")
    #dna_new=FUNCTIONS_4_DALdna.ohe_to_seq(x_new.squeeze(0), four_zeros_ok=True)
    #print(f"{dna_new}")
    #print(f"{dna==dna_new=}")
    print()