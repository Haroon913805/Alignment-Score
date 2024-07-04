import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices

matrix = substitution_matrices.load("BLOSUM62")

aligner = Align.PairwiseAligner(
    mode="local",
    open_gap_score=-11.0,
    extend_gap_score=-3.0,
    substitution_matrix=matrix
)

queryfasta = "query.fasta"
dbfasta = "db.fasta"

query_sequences =list(SeqIO.parse(queryfasta , "fasta"))
db_sequences = list(SeqIO.parse(dbfasta, "fasta"))

for query in query_sequences:
    scores = []
    lengths = []
    for db in db_sequences:
        score = aligner.score(query.seq, db.seq)
        scores.append(score)
        lengths.append(len(db.seq))
    
    plt.scatter(lengths, scores, label=f'{query.id}')
    plt.xlabel('Length of Database Sequences')
    plt.ylabel('Alignment Score')
    plt.title(f'Scatter Plot for Query Sequence {query.id}')
    plt.legend()
    plt.show()

interval_size = 50  
for query in query_sequences:
    scores = []
    lengths = []
    z_scores = []
    
    for db in db_sequences:
        score = aligner.score(query.seq, db.seq)
        scores.append(score)
        lengths.append(len(db.seq))
    
    unique_lengths = sorted(set(lengths))
    for length in unique_lengths:
        interval_scores = [scores[i] for i in range(len(scores)) if lengths[i] == length]
        if len(interval_scores) > 1:
            mu = np.mean(interval_scores)
            sigma = np.std(interval_scores)
            for score in interval_scores:
                z = (score - mu) / sigma
                z_scores.append(z)
    
    plt.scatter(lengths, z_scores, label=f'{query.id}')
    plt.xlabel('Length of Database Sequences')
    plt.ylabel('Z-Score')
    plt.title(f'Z-Score Plot for Query Sequence {query.id}')
    plt.legend()
    plt.show()

import random

for query in query_sequences:
    scrambled_scores = []
    
    scrambled_seq = ''.join(random.sample(str(query.seq), len(query.seq)))
    
    for db in db_sequences:
        score = aligner.score(scrambled_seq, db.seq)
        scrambled_scores.append(score)
    
    threshold = np.percentile(scrambled_scores, 95)  
    print(f'Significance threshold for {query.id}: {threshold}')