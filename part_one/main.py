
from utils import GMLBuilder, build_matrix, hamming_distance_vector, jaccard_similarity, k_nearest_neighbor_graph, mst_prim, pretty_print, relative_neighborhood_graph, write_to_matrix
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
        # TODO: Just do this for distance?
        write_to_matrix(matrix_jaccard, i, j, 1.0 - jaccard_similarity(list(election.to_binary_vector()), list(election2.to_binary_vector())))

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
hamming_distance_rng = GMLBuilder("hamming_distance_rng.gml")
relative_neighborhood_graph(matrix, hamming_distance_rng, [str(election.year) for election in elections])
hamming_distance_rng.write()

# 3) Jaccard Similarity MST
jaccard_similarity_mst = GMLBuilder("jaccard_similarity_mst.gml")
mst_prim(matrix_jaccard, jaccard_similarity_mst, [str(election.year) for election in elections])
jaccard_similarity_mst.write()

# 4) Jaccard Similarity RNG
jaccard_similarity_rng = GMLBuilder("jaccard_similarity_rng.gml")
relative_neighborhood_graph(matrix_jaccard, jaccard_similarity_rng, [str(election.year) for election in elections])
jaccard_similarity_rng.write()


# 5) Hamming Distance 2-NN graph
hamming_distance_2nn = GMLBuilder("hamming_distance_2nn.gml")
k_nearest_neighbor_graph(matrix, hamming_distance_2nn, [str(election.year) for election in elections], k_count=2)
hamming_distance_2nn.write()

# 6) Jaccard Similarity 2-NN graph
jaccard_similarity_2nn = GMLBuilder("jaccard_similarity_2nn.gml")
k_nearest_neighbor_graph(matrix_jaccard, jaccard_similarity_2nn, [str(election.year) for election in elections], k_count=2)
jaccard_similarity_2nn.write()
