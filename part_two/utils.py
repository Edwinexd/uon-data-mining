import math
from typing import Dict, List, Optional, Set, Tuple, TypeVar

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

def chebyshev_distance(a: List[T], b: List[T]) -> float:
    return max([abs(a[i] - b[i]) for i in range(len(a))])

def euclidean_distance(a: List[int], b: List[int]) -> float:
    return math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))

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
        builder.add_edge(target, source) # MST is undirected
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

def build_matrix(size: int, default: T = -1) -> List[List[T]]:
    return [[default for _ in range(size)] for _ in range(size)]

def write_to_matrix(matrix: List[List[T]], first: int, second: int, value: T):
    matrix[first][second] = value
    matrix[second][first] = value

def k_nearest_neighbor_classification(matrix: List[List[T]], class_mapping: Dict[int, int], target: int, k_count: int = 3) -> List[int]:
    # k-nearest neighbors edge targets from target node
    targets = []

    # Note: This implementation of k-NN may connect more than k-edges to vertices.
    # E.x. [0,3,3,3,3,4] the 0 and all the 3s will be included as all 3:s are considered 2:nd lowest weight.
    for i, distance in enumerate(matrix[target]):
        if i == target:
            continue
        # amount of nodes with smaller distance to i than j
        smaller = 0
        for j, other_distance in enumerate(matrix[target]):
            if i == j or target == j:
                continue
            if other_distance < distance:
                smaller += 1
            if smaller >= k_count:
                break
        # if the current distance is not of the k:th smallest, there should be no edge
        if smaller >= k_count:
            continue

        targets.append(i)

    # class: count
    neighbor_classes_count: Dict[int, int] = {}
    for neighbor in targets:
        neighbor_class = class_mapping[neighbor]
        if neighbor_class not in neighbor_classes_count:
            neighbor_classes_count[neighbor_class] = 0
        neighbor_classes_count[neighbor_class] += 1

    max_value = max(neighbor_classes_count.values())
    max_classes = [key for key, value in neighbor_classes_count.items() if value == max_value]

    return max_classes

class PerformanceMeasurer:
    def __init__(self, true_positive: int = 0, false_positive: int = 0, true_negative: int = 0, false_negative: int = 0):
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative
    
    # Sensitivity (recall)
    # True positive rate (TPR) = TP / (TP + FN)
    def sensitivity(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    # Specificity ()
    # True negative rate (TNR) = TN / (TN + FP)
    def specificity(self) -> float:
        return self.true_negative / (self.true_negative + self.false_positive)

    # Accuracy
    # (TP+TN)/(TP+TN+FP+FN)
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / (self.true_positive + self.true_negative + self.false_positive + self.false_negative)

    # Precision
    # TP / (TP + FP)
    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    # F1-score
    # 2 / ((1 / recall) + (1 / precision))
    def f1_score(self) -> float:
        recall = self.sensitivity()
        precision = self.precision()
        return 2 / ((1 / recall) + (1 / precision))

    # Matthews Correlation Coefficient
    # (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    def matthews_correlation_coefficient(self) -> float:
        return (self.true_positive * self.true_negative - self.false_positive * self.false_negative) / (((self.true_positive + self.false_positive) * (self.true_positive + self.false_negative) * (self.true_negative + self.false_positive) * (self.true_negative + self.false_negative)) ** 0.5)

    # Youden’s J statistic (roc)
    # sensitivity + specificity - 1
    def youdens_j_statistic(self) -> float:
        return self.sensitivity() + self.specificity() - 1
