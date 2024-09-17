import math
from typing import Dict, List, Optional, Set, Tuple, TypeVar

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

def proteins_distance(a: List[float], b: List[float]) -> float:
    # Chebyshev distance
    return max([abs(a[i] - b[i]) for i in range(len(a))])
    # import numpy as np
    # # cosine similarity => distance
    # return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def samples_distance(a: List[float], b: List[float]) -> float:
    # TODO: Use something else?
    return proteins_distance(a, b)

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

def build_matrix(size: int, default: T = -1) -> List[List[T]]:
    return [[default for _ in range(size)] for _ in range(size)]

def write_to_matrix(matrix: List[List[T]], first: int, second: int, value: T):
    matrix[first][second] = value
    matrix[second][first] = value

G = TypeVar("G", str, int)

# TODO NOT SURE IF THIS SHOULD BE KEPT
# ______________________________________________________
def _uniquely_identifies_classes(data: List[List[List[T]]], classes: List[G], selection: Set[int]) -> bool:
    mapping = {}
    for i, group in enumerate(data):
        for entry in group:
            key = tuple(val for j, val in enumerate(entry) if j in selection)
            if key in mapping:
                if mapping[key] != classes[i]:
                    print(f"Key {key} maps to both {mapping[key]} and {classes[i]}")
                    return False
            else:
                mapping[key] = classes[i]
    
    return True

def feature_selection(data: List[List[List[T]]], classes: List[G], current: Optional[Set[int]] = None, depth_remaining = 0) -> Set[int]:
    if current is None:
        current = set(range(len(data[0][0])))
        assert _uniquely_identifies_classes(data, classes, current), "Even initial selection does not uniquely identify classes!"

    # try removing each feature
    # call recursively
    local_optimal = current.copy()
    for val in current:
        new_current = current.copy()
        new_current.remove(val)

        if not _uniquely_identifies_classes(data, classes, new_current):
            continue

        print(f"Removing {val} results in a valid selection, {len(new_current)} features left")

        # So - for a perfect selection, we should try all possible combinations
        # as keeping this featurem might lead to a better selection later on
        # this is not feasible so I'm using a heurustic 
        # approach and accepting results of a single removal
        maybe_local_optimal = feature_selection(data, classes, new_current, depth_remaining-1)
        if len(maybe_local_optimal) < len(local_optimal):
            local_optimal = maybe_local_optimal

        # Heuristic if we have more depth we go
        if depth_remaining < 1:
            break


    return local_optimal

# ______________________________________________________

# Fayyad-Irani Discretization
def fayyad_irani_discretization(values: List[T]) -> List[Tuple[T, T]]:
    values.sort()

    # Entropy calculation
    def entropy(start: int, end: int) -> float:
        counts = {}
        for i in range(start, end):
            if values[i] not in counts:
                counts[values[i]] = 0
            counts[values[i]] += 1
        entropy = 0
        for count in counts.values():
            p = count / (end - start)
            entropy -= p * math.log2(p)
        return entropy
    
    # Information gain calculation
    def information_gain(start: int, end: int, split: int) -> float:
        total_entropy = entropy(start, end)
        left_entropy = entropy(start, split)
        right_entropy = entropy(split, end)
        return total_entropy - (left_entropy + right_entropy)
    
    # Find the best split
    best_split = 0
    best_information_gain = 0
    for i in range(1, len(values)):
        gain = information_gain(0, len(values), i)
        if gain > best_information_gain:
            best_information_gain = gain
            best_split = i
    
    return [(values[i], values[i+1]) for i in range(best_split)]


def feature_selection_2(data: List[List[T]], class_mapping: Dict[int, int]) -> Set[int]:
    return set([0])

# matrix - graph matrix
# classes - all classes
# class_mapping - mapping from index to class (index in classes)
# labels - labels for each node
def k_nearest_neighbor_classification(matrix: List[List[T]], class_mapping: Dict[int, int], target: int, classses_labels: Optional[List[str]] = None, k_count: int = 3) -> List[str]:
    if classses_labels is None:
        # unqiue classes
        classses_labels = [str(i) for i in set(class_mapping.values())]
    
    # k-nearest neighbors edge targets from target node
    targets = []

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

    return [classses_labels[class_] for class_ in max_classes]

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

# TOOD REMOVE
def pretty_print(matrix: List[List[T]]):
    for row in matrix:
        for x in row:
            if (isinstance(x, float)):
                print(f"{x:.2f}".ljust(4), end=" ")
            else:
                print(str(x).ljust(2), end=" ")
        print()
