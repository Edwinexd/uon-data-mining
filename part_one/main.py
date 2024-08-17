
from typing import List
from utils import GMLBuilder, build_matrix, hamming_distance_vector, jaccard_similarity, mst_prim, pretty_print, write_to_matrix
from reader import get_elections


# 1) a (hamming distance)
elections = get_elections()

matrix = build_matrix(len(elections))

for i, election in enumerate(elections):
    for j, election2 in enumerate(elections):
        write_to_matrix(matrix, i, j, hamming_distance_vector(list(election.to_binary_vector()), list(election2.to_binary_vector())))

pretty_print(matrix)

# 1) a (jaccard similarity)
matrix_jaccard = build_matrix(len(elections), -1.0)

for i, election in enumerate(elections):
    for j, election2 in enumerate(elections):
        write_to_matrix(matrix_jaccard, i, j, jaccard_similarity(list(election.to_binary_vector()), list(election2.to_binary_vector())))

pretty_print(matrix_jaccard)

# 1) b - comparing columns / attributes
columns = []
for election in elections:
    for i, value in enumerate(election.to_binary_vector()):
        if len(columns) <= i:
            columns.append([])
        columns[i].append(value)

matrix_columns = build_matrix(len(columns))
matrix_columns_jaccard = build_matrix(len(columns), -1.0)

for i, column in enumerate(columns):
    for j, column2 in enumerate(columns):
        write_to_matrix(matrix_columns, i, j, hamming_distance_vector(column, column2))
        write_to_matrix(matrix_columns_jaccard, i, j, jaccard_similarity(column, column2))

pretty_print(matrix_columns)
pretty_print(matrix_columns_jaccard)

# 1) c - mst, using prim
hamming_distance_mst = GMLBuilder("hamming_distance_mst.gml")

mst_prim(matrix, hamming_distance_mst, [str(election.year) for election in elections])
hamming_distance_mst.write()

# 2) Relative Neighborhood Graph

