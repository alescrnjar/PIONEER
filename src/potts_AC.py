import numpy as np
import random
import FUNCTIONS_4_DALdna

# https://github.com/debbiemarkslab/EVcouplings/tree/develop
# https://en.wikipedia.org/wiki/Potts_model

def initialize_sequence_str(length):
    alphabet_choices=['A', 'C', 'G', 'T']
    return np.random.choice(alphabet_choices, size=length)

def initialize_sequence(length):
    #alphabet_choices=['A', 'C', 'G', 'T']
    alphabet_choices=[np.array([0.,0.,0.,1.]), np.array([0.,0.,1.,0.]), np.array([0.,1.,0.,0.]), np.array([1.,0.,0.,0.]) ]

    #return np.random.choice(alphabet_choices, size=length)
    #return random.choice(alphabet_choices, size=length)

    seq=np.zeros((length,4))
    for i in range(length):
        seq[i]=np.array(alphabet_choices[np.random.randint(4)])
    return seq
        

def potts_model(sequence, beta=1.0):
    #alphabet_choices=['A', 'C', 'G', 'T']
    alphabet_choices=[np.array([0.,0.,0.,1.]), np.array([0.,0.,1.,0.]), np.array([0.,1.,0.,0.]), np.array([1.,0.,0.,0.]) ]

    new_sequence = sequence.copy()

    for i in range(len(sequence)):
        neighbors = [sequence[(i-1) % len(sequence)], sequence[(i+1) % len(sequence)]] # For each position i, consider its neighboring positions ((i-1) % len(sequence) and (i+1) % len(sequence)).
        energy_diff = sum([1 if nucleotide != neighbor else 0 for nucleotide, neighbor in zip(sequence[i], neighbors)]) # For each neighboring nucleotide, compare it with the nucleotide at position i. If they are different, increment the energy difference (energy_diff) by 1. This reflects a penalty for differences in neighboring nucleotides.

        if np.random.rand() < np.exp(-beta * energy_diff):
            #new_sequence[i] = np.random.choice([n for n in alphabet_choices if n != sequence[i]])
            #new_sequence[i] = random.choice([n for n in alphabet_choices if n != sequence[i]])
            idx=alphabet_choices.index(new_sequence[i])
            while idx==alphabet_choices.index(new_sequence[i]):
                idx=np.random.randint(len(alphabet_choices))
            new_sequence[i]=alphabet_choices[idx]

    return new_sequence

if __name__=='__main__':

    # Example usage:
    sequence_length = 7 #249
    initial_sequence = initialize_sequence(sequence_length)
    print(f"{initial_sequence=}")
    #print(f"initial sequence:\n {''.join(initial_sequence)}")

    # Run the Potts model for a certain number of iterations
    num_iterations = 1000
    for _ in range(num_iterations):
        initial_sequence = potts_model(initial_sequence)

    print('Generated DNA sequence:\n', ''.join(initial_sequence))

    ohed=FUNCTIONS_4_DALdna.dna_to_one_hot(str(''.join(initial_sequence))).transpose(1,0)
    #print(ohed)
    #ohed=np.transpose(ohed,(1,0))
    print(ohed)
    print(f"{ohed.shape=}")

    print("WARNING: does it change anything to make it into an MCMC?")