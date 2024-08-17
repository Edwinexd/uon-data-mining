

from typing import List, Optional, Tuple, TypeVar


def hamming_distance(string_a: str, string_b: str) -> int:
    count = 0
    for i, char in enumerate(string_a):
        if char != string_b[i]:
            count += 1
    return count

def hamming_distance_vector(a: List[int], b: List[int]) -> int:
    count = 0
    for i, num in enumerate(a):
        if num != b[i]:
            count += 1
    return count

# TODO: What is this thing?
def jaccard_similarity(a: List[int], b: List[int]) -> float:
    intersection = 0
    union = 0
    for i, num in enumerate(a):
        if num == 1 and b[i] == 1:
            intersection += 1
        if num == 1 or b[i] == 1:
            union += 1
    return intersection / union

class GMLBuilder:
    def __init__(self, path: str):
        self.path = path
        self.nodes: List[Tuple[int, str]] = []
        self.edges: List[Tuple[int, int]] = []
        self.written = False

    def add_node(self, id_: int, label: str):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        self.nodes.append((id_, label))

    def add_edge(self, source: int, target: int):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        self.edges.append((source, target))

    def write(self):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("graph [\n")
            for node in self.nodes:
                f.write(f"  node [\n    id {node[0]}\n    label \"{node[1]}\"\n  ]\n")
            for edge in self.edges:
                f.write(f"  edge [\n    source {edge[0]}\n    target {edge[1]}\n  ]\n")
            f.write("]\n")
        self.written = True

def mst_prim(matrix: List[List[int]], builder: GMLBuilder, labels: Optional[List[str]] = None) -> None:
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]
    edges: List[Tuple[int, int, int]] = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            edges.append((i, j, value))
    edges.sort(key=lambda x: x[2])
    included = [0]
    builder.add_node(0, labels[0])
    while len(included) < len(matrix):
        # Filtering out available usable edges (i.e. edges where from is in our open "set")
        # and then the ones which takes us somewhere new (i.e. target no in open set)
        available = list(filter(lambda x: x[0] in included and x[1] not in included, list(edges)))
        # if this were to fail there can exist no MST
        source, target, _ = available.pop()
        builder.add_node(target, labels[target])
        builder.add_edge(source, target)
        included.append(target)

def relative_neighborhood_graph(matrix: List[List[int]], builder: GMLBuilder, labels: Optional[List[str]] = None) -> None:
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]

    return None

T = TypeVar("T")

def build_matrix(size: int, default: T = -1) -> List[List[T]]:
    return [[default for _ in range(size)] for _ in range(size)]

def write_to_matrix(matrix: List[List[T]], first: int, second: int, value: T):
    matrix[first][second] = value
    matrix[second][first] = value

# TOOD REMOVE
def pretty_print(matrix: List[List[T]]):
    for row in matrix:
        for x in row:
            if (isinstance(x, float)):
                print(f"{x:.2f}".ljust(4), end=" ")
            else:
                print(str(x).ljust(2), end=" ")
        print()
