import numpy as np
from hmmlearn import hmm

# Define the number of states: 1 background + 8 match + 63 spacer = 72
n_states = 72
n_match_states = 8
n_spacer_states = 63

# Define the transition matrix
trans_mat = np.zeros((n_states, n_states))

# Background state transitions
trans_mat[0, 0] = 0.9  # Stay in background
trans_mat[0, 1] = 0.1  # Move to first match state

# Match state transitions
for i in range(1, n_match_states):
    trans_mat[i, i + 1] = 1.0  # Move to next match state

# Transition from last match state to first spacer state
trans_mat[n_match_states, n_match_states + 1] = 1.0

# Spacer state transitions
for i in range(n_match_states + 1, n_match_states + n_spacer_states - 1):
    trans_mat[i, i + 1] = 0.95  # Stay in spacer state
    trans_mat[i, 0] = 0.05      # Transition back to background state

# Fix the transition for the last spacer state to ensure it sums to 1
trans_mat[n_match_states + n_spacer_states - 1, 0] = 1.0

# Emission probabilities
# Assuming simple DNA (A, C, G, T) with equal probabilities
emission_prob = np.tile([0.25, 0.25, 0.25, 0.25], (n_states, 1))

# Initial state probabilities
start_prob = np.zeros(n_states)
start_prob[0] = 1.0  # Start from background state

# Create HMM model
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_prob
model.transmat_ = trans_mat
model.emissionprob_ = emission_prob

# Generate a sample sequence
seq_length = 1000
X, Z = model.sample(seq_length)

# Decode the sequence to find hidden states (for example, to identify TFBS locations)
logprob, states = model.decode(X, algorithm="viterbi")

# Print the sequence and the hidden states
print("Generated sequence:", X.ravel())
print("Hidden states:", states)
