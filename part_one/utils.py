from typing import Dict, List, Optional, Set, Tuple, TypeVar


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
        # nodes: id -> (label, edges (targets))
        self.nodes: Dict[int, Tuple[str, Set[int]]] = {}
        self.written = False

    def add_node(self, id_: int, label: str):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        self.nodes[id_] = (label, set())

    def add_edge(self, source: int, target: int):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        self.nodes[source][1].add(target)


    def write(self):
        assert not self.written, "GML Builder can not be used after contents have been written to file"
        # source, target, bidirectional
        edges: List[Tuple[int, int, bool]] = []
        for source, (_, targets) in self.nodes.items():
            for target in targets:
                bidirectional = source in self.nodes[target][1]
                if not bidirectional:
                    edges.append((source, target, False))
                    continue
                # Only add bidirectional edge if not already added
                if (target, source, True) not in edges:
                    edges.append((source, target, True))

        with open(self.path, "w", encoding="utf-8") as f:
            f.write("graph [\n")
            for id_, (label, _) in self.nodes.items():
                f.write(f"  node [\n    id {id_}\n    label \"{label}\"\n  ]\n")
            for edge in edges:
                # Adding the graphics part removes the default arrow in yEd
                # TODO: Although one should prob be re-added for directed edges
                target_arrow_string = '			targetArrow	"standard"' if not edge[2] else ""
                f.write(f"  edge [\n    source {edge[0]}\n    target {edge[1]}\n    graphics [{target_arrow_string}]\n  ]\n")
            f.write("]\n")
        self.written = True

    @classmethod
    def intersection(cls, path: str, a: "GMLBuilder", b: "GMLBuilder") -> "GMLBuilder":
        """Assumes nodes have the same id:s in both graphs"""
        intersection = GMLBuilder(path)
        for id_, (label, _) in a.nodes.items():
            if id_ in b.nodes:
                intersection.add_node(id_, label)

        for id_, (_, targets) in a.nodes.items():
            if id_ not in intersection.nodes:
                continue
            for target in targets:
                if target in b.nodes:
                    intersection.add_edge(id_, target)

        return intersection

T = TypeVar("T", int, float)

def mst_prim(matrix: List[List[T]], builder: GMLBuilder, labels: Optional[List[str]] = None) -> None:
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]
    edges: List[Tuple[int, int, T]] = []
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
        source, target, _ = available[0]
        builder.add_node(target, labels[target])
        builder.add_edge(source, target)
        included.append(target)

def relative_neighborhood_graph(matrix: List[List[T]], builder: GMLBuilder, labels: Optional[List[str]] = None) -> None:
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]
    for i, label in enumerate(labels):
        builder.add_node(i, label)
    for i, row in enumerate(matrix):
        for j, distance in enumerate(row):
            if i == j:
                continue
            exists = False
            # If there exists a node that is closer to both i and j than i and j should not be connected
            for k in range(len(matrix)):
                if i == k or j == k:
                    continue
                if matrix[i][k] < distance and matrix[j][k] < distance:
                    exists = True
                    break
            if exists:
                continue

            builder.add_edge(i, j)

# TODO: Unsure if the implementation is correct
def k_nearest_neighbor_graph(matrix: List[List[T]], builder: GMLBuilder, labels: Optional[List[str]] = None, k_count: int = 2) -> None:
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]
    for i, label in enumerate(labels):
        builder.add_node(i, label)

    for i, row in enumerate(matrix):
        connected_to = []
        for j, distance in enumerate(row):
            if i == j:
                continue
            # amount of nodes with smaller distance to i than j
            smaller = 0
            for k, other_distance in enumerate(row):
                if i == k or j == k:
                    continue
                if other_distance < distance:
                    smaller += 1
                if smaller >= k_count:
                    break
            # if the current distance is not of the k:th smallest, there should be no edge
            if smaller >= k_count:
                continue
            
            builder.add_edge(i, j)
            connected_to.append((labels[j], distance))
        # print(f"Connected {labels[i]} to {connected_to}")
        


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
