
import random

from utils import GMLBuilder, build_matrix, k_nearest_neighbor_graph, mst_prim, relative_neighborhood_graph, write_to_matrix


with open('concrete_data/Concrete_Data.csv', 'r', encoding="utf-8") as f:
    data = f.read()

# split the data by new line
data = data.split('\n')
header = [val for val in data[0].split(',')]
data = data[1:-1] # excluding last line since it is empty
data = [[float(val) for val in line.split(',')] for line in data]

concrete_data = data


# Reduce to 100 samples, deterministic sampling
random.seed(0)
data = random.sample(data, 100)

concrete_matrix = build_matrix(len(data), float("-inf"))

for i, entry in enumerate(data):
    for j, other_entry in enumerate(data):
        if i == j:
            write_to_matrix(concrete_matrix, i, j, 0)
        # My distance function for this is the sum of all differences between all attributes (excl. target) 
        concrete_matrix[i][j] = sum(abs(a - b) for a, b in zip(entry[:-1], other_entry[:-1]))
        write_to_matrix(concrete_matrix, i, j, concrete_matrix[i][j])

concrete_graph = GMLBuilder("concrete_neighborhood.gml")
# Pylint is complaining about the float list not being subscriptable...
# pylint: disable=unsubscriptable-object
relative_neighborhood_graph(concrete_matrix, concrete_graph, [str(val[-1]) for val in data])
concrete_graph.write()


# Part 2 - Churn
# I'm only writing code to do a train/test split here, rest of it in weka
with open('churn_data/full.csv', 'r', encoding="utf-8") as f:
    data = f.read()

# split the data by new line
data = data.split('\n')
header = [val for val in data[0].split(',')]
data = data[1:-1] # excluding last line since it is empty
data = [[val for val in line.split(',')] for line in data]

print(len(header), len(data[0]))

random.seed(0)
random.shuffle(data)
# 80/20 split
num_train = int(len(data) * 0.8)
train, test = data[:num_train], data[num_train:]

with open('churn_data/train.csv', 'w', encoding="utf-8") as f:
    f.write(','.join(header))
    f.write("\n")
    f.write('\n'.join([','.join(line) for line in train]))

with open('churn_data/test.csv', 'w', encoding="utf-8") as f:
    f.write(','.join(header))
    f.write("\n")
    f.write('\n'.join([','.join(line) for line in test]))

# 5) MST
# using complete dataset this time
churn_matrix = build_matrix(len(concrete_data), float("-inf"))
for i, entry in enumerate(concrete_data):
    for j, other_entry in enumerate(concrete_data):
        if i == j:
            write_to_matrix(churn_matrix, i, j, 0)
        churn_matrix[i][j] = sum(abs(a - b) for a, b in zip(entry[:-1], other_entry[:-1]))
        write_to_matrix(churn_matrix, i, j, churn_matrix[i][j])

concrete_mst = GMLBuilder("concrete_mst.gml")
mst_prim(churn_matrix, concrete_mst, [str(val[-1]) for val in concrete_data])
concrete_mst.write()

# knn
concrete_knn = GMLBuilder("concrete_knn.gml")
# TODO: What k?
k_nearest_neighbor_graph(churn_matrix, concrete_knn, [str(val[-1]) for val in concrete_data], k_count=5)
concrete_knn.write()

concrete_intersection = GMLBuilder.intersection("concrete_intersection.gml", concrete_knn, concrete_mst)
concrete_intersection.write()


# @attribute Q1 {0,1}
# @attribute Q2 {0,1}
# @attribute Q3 {0,1}
# @attribute Q4 {0,1}
# @attribute Q5 {0,1}
# @attribute Q6 {0,1}
# @attribute Q7 {0,1}
# @attribute Q8 {0,1}
# @attribute Q9 {0,1}
# @attribute Q10 {0,1}
# @attribute Q11 {0,1}
# @attribute Q12 {0,1}
data = """
0,0,0,0,1,0,0,1,1,0,0,0
0,0,0,0,1,0,0,0,0,0,0,0
0,1,0,0,1,0,1,0,0,0,0,1
0,0,0,0,1,0,0,1,0,0,0,0
0,1,1,0,1,0,1,1,0,1,0,0
0,1,0,0,1,1,1,1,0,0,1,0
0,1,0,0,1,0,1,0,0,0,1,0
0,0,0,0,1,0,1,0,0,0,0,0
0,0,0,0,1,0,0,1,1,0,0,0
1,1,0,0,0,0,1,1,1,0,1,0
1,1,0,0,1,0,1,0,0,0,1,0
1,0,0,1,0,0,1,1,0,0,0,0
1,1,0,0,1,0,1,0,0,0,1,0
1,1,0,0,0,1,0,1,0,0,0,0
1,1,0,0,0,0,1,0,0,0,0,0
1,1,0,0,1,1,1,1,0,0,1,0
1,1,0,0,1,0,1,1,0,0,1,0
1,1,1,0,1,0,0,1,0,0,0,0"""
# print a set of {Q1...Q12} for each line if a 1 is present
data = data.split('\n')
for line in data:
    print([f"Q{i+1}" for i, val in enumerate(line.split(',')) if val == '1'])
