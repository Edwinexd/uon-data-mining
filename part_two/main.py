from typing import Dict, List, Tuple, Union
from utils import GMLBuilder, PerformanceMeasurer, build_matrix, chebyshev_distance, euclidean_distance, k_nearest_neighbor_classification, mst_prim, relative_neighborhood_graph, write_to_matrix


# 1. Perform Explanatory Data Analysis (EDA) using Students’ Academic Performance Dataset (xAPI-Edu-Data.csv)
# a. Is there any strong correlation between variables? Pearson, Scatter plot, Correlogram
with open("xAPI-Edu-Data.csv", "r", encoding="utf-8") as file:
    data = file.read()

data = data.split("\n")
students_academic_performance_labels = data[0].split(",")
students_academic_performance = [i.split(",") for i in data[1:] if i]  # if i for removing final empty line

# Pearson requires numerical data
# encoding categorical data (I'm not taking into account ordinality of )
# label: (encodings, next number)
encodings: Dict[str, Tuple[Dict[str, int], int]] = {}

for i, label in enumerate(students_academic_performance_labels):
    if students_academic_performance[0][i].isnumeric():  # if it's a number, we don't need to convert it
        continue

    encodings[label] = ({}, 0)

for student in students_academic_performance:
    for i, value in enumerate(student):
        if students_academic_performance_labels[i] not in encodings:
            continue

        if value not in encodings[students_academic_performance_labels[i]][0]:
            encodings[students_academic_performance_labels[i]][0][value] = encodings[students_academic_performance_labels[i]][1]
            encodings[students_academic_performance_labels[i]] = (
                encodings[students_academic_performance_labels[i]][0],
                encodings[students_academic_performance_labels[i]][1] + 1,
            )

        student[i] = str(encodings[students_academic_performance_labels[i]][0][value])

# everything is now numerical but typed as string
students_academic_performance_numeric = [[int(i) for i in student] for student in students_academic_performance]


def pearson_correlation(x: List[int], y: List[int]) -> float:
    assert len(x) == len(y), "Entries must have same length"
    count = len(x)
    mean_of_x = sum(x) / count
    mean_of_y = sum(y) / count

    numerator = sum((val_x - mean_of_x) * (val_y - mean_of_y) for val_x, val_y in zip(x, y))
    denominator_x = sum((n - mean_of_x) ** 2 for n in x)
    denominator_y = sum((n - mean_of_y) ** 2 for n in y)
    denominator = (denominator_x * denominator_y) ** 0.5

    return numerator / denominator


correlations = []

for i, label in enumerate(students_academic_performance_labels):
    for j in range(i + 1, len(students_academic_performance_labels)):
        pearson = pearson_correlation(
            [student[i] for student in students_academic_performance_numeric],
            [student[j] for student in students_academic_performance_numeric],
        )
        correlations.append((label, students_academic_performance_labels[j], pearson))
        if abs(pearson) > 0.5:
            print(f"{label} and {students_academic_performance_labels[j]}: {pearson}")

# 10 most correlated
correlations.sort(key=lambda x: abs(x[2]), reverse=True)
print(correlations[:10]) # [('NationalITy', 'PlaceofBirth', 0.9064893804165973), ('raisedhands', 'VisITedResources', 0.691571705469296), ('raisedhands', 'AnnouncementsView', 0.643917767918391), ('VisITedResources', 'AnnouncementsView', 0.5945000269505766), ('ParentAnsweringSurvey', 'ParentschoolSatisfaction', 0.539874839229609), ('VisITedResources', 'StudentAbsenceDays', -0.49902965870610955), ('PlaceofBirth', 'Topic', 0.48611736628792546), ('raisedhands', 'StudentAbsenceDays', -0.463881741110979), ('NationalITy', 'Topic', 0.46379786780165744), ('StageID', 'GradeID', -0.42537767486722944)]

# b. From analysis, True or False
# 1) Students "raised by their mums" are more actively involved in study-related works (raising hands, discussion) (0.2 marks)
index_of_relation = students_academic_performance_labels.index("Relation")
index_of_raised_hands = students_academic_performance_labels.index("raisedhands")
index_of_discussion = students_academic_performance_labels.index("Discussion")

encodings_relation_mom = encodings["Relation"][0]["Mum"]
students_raised_mom = [student for student in students_academic_performance_numeric if student[index_of_relation] == encodings_relation_mom]
students_not_raised_mom = [student for student in students_academic_performance_numeric if student[index_of_relation] != encodings_relation_mom]

raised_hands_mom = sum(student[index_of_raised_hands] for student in students_raised_mom) / len(students_raised_mom)
raised_hands_not_mom = sum(student[index_of_raised_hands] for student in students_not_raised_mom) / len(students_not_raised_mom)

discussion_mom = sum(student[index_of_discussion] for student in students_raised_mom) / len(students_raised_mom)
discussion_not_mom = sum(student[index_of_discussion] for student in students_not_raised_mom) / len(students_not_raised_mom)

print(raised_hands_mom, raised_hands_not_mom, discussion_mom, discussion_not_mom) # 60.20, 37.43, 44.17, 42.68
# True since both means are higher for students raised by their moms


# 2) No apparent gender bias when it comes to subject/topic choices (0.2 marks)
encodings_gender_male = encodings["gender"][0]["M"]
subjects_by_gender: Dict[str, Dict[str, int]] = {}
total_by_gender: Dict[str, int] = {}

for student in students_academic_performance_numeric:
    gender = student[students_academic_performance_labels.index("gender")]
    gender_name = [i for i in encodings["gender"][0].items() if i[1] == gender][0][0]
    topic = student[students_academic_performance_labels.index("Topic")]
    topic_name = [i for i in encodings["Topic"][0].items() if i[1] == topic][0][0]
    if gender_name not in total_by_gender:
        total_by_gender[gender_name] = 0
    
    total_by_gender[gender_name] += 1

    if gender_name not in subjects_by_gender:
        subjects_by_gender[gender_name] = {}

    if topic_name not in subjects_by_gender[gender_name]:
        subjects_by_gender[gender_name][topic_name] = 0

    subjects_by_gender[gender_name][topic_name] += 1
    
print(subjects_by_gender)
subjects_by_gender_percentages = {}
for gender, topics in subjects_by_gender.items():
    subjects_by_gender_percentages[gender] = [(topic, count / total_by_gender[gender]) for topic, count in topics.items()]
    subjects_by_gender_percentages[gender].sort(key=lambda x: x[0])
print(subjects_by_gender_percentages)
# True - there is no apparant bias

# 3) Girls seem to have better overall performance than boys (0.2 marks)
# True - proportionally way more boys in the lowest and somewhat more in the medium although the same in the highest i.e. proportionally girls did better 
# 4) Boys are generally a bit more open to discussions, visiting resources, and raising hands (0.2 marks)
index_of_gender = students_academic_performance_labels.index("gender")
index_of_discussion = students_academic_performance_labels.index("Discussion")
index_of_visiting_resources = students_academic_performance_labels.index("VisITedResources")
index_of_raised_hands = students_academic_performance_labels.index("raisedhands")

encodings_male = encodings["gender"][0]["M"]
students_male = [student for student in students_academic_performance_numeric if student[index_of_gender] == encodings_male]
students_female = [student for student in students_academic_performance_numeric if student[index_of_gender] != encodings_male]

discussion_male = sum(student[index_of_discussion] for student in students_male) / len(students_male)
discussion_female = sum(student[index_of_discussion] for student in students_female) / len(students_female)

visiting_resources_male = sum(student[index_of_visiting_resources] for student in students_male) / len(students_male)
visiting_resources_female = sum(student[index_of_visiting_resources] for student in students_female) / len(students_female)

raised_hands_male = sum(student[index_of_raised_hands] for student in students_male) / len(students_male)
raised_hands_female = sum(student[index_of_raised_hands] for student in students_female) / len(students_female)

print(discussion_male, visiting_resources_male, raised_hands_male)
print(discussion_female, visiting_resources_female, raised_hands_female)
print(discussion_male, discussion_female, visiting_resources_male, visiting_resources_female, raised_hands_male, raised_hands_female)
# Female has higher in all three, so False

# 5) Those who participated more (higher counts in Discussion, Announcement Views, Raised Hands), usually performs better (0.2 marks)
index_of_announcement_views = students_academic_performance_labels.index("AnnouncementsView")
for performance in ["L", "M", "H"]:
    index_of_performance = students_academic_performance_labels.index("Class")
    performance_encodings = encodings["Class"][0][performance]
    students_performance = [student for student in students_academic_performance_numeric if student[index_of_performance] == performance_encodings]
    discussion_performance = sum(student[index_of_discussion] for student in students_performance) / len(students_performance)
    announcement_views_performance = sum(student[index_of_announcement_views] for student in students_performance) / len(students_performance)
    raised_hands_performance = sum(student[index_of_raised_hands] for student in students_performance) / len(students_performance)
    print(performance, discussion_performance, announcement_views_performance, raised_hands_performance)

# L 30.834645669291337 15.5748031496063 16.88976377952756
# M 43.791469194312796 40.96208530805687 48.93838862559242
# H 53.66197183098591 53.38028169014085 70.28873239436619
# True

# Exercise 2 (3 marks). Using the “Training” set of the Alzheimer’s Disease3 dataset, you will need to find several proximity graphs: 
# a) Minimum Spanning Tree (MST) for samples (0.5 marks)
with open("alzheimers_disease/training.csv", "r", encoding="utf-8") as file:
    data_raw = file.read()

data = data_raw.split("\n")
alzheimers_header = data[0].split(",")
data_entries = [i.split(",") for i in data[1:] if i]  # if i for removing final empty

# samples are the columns
samples_labels = [f"{label}-{i}" for i, label in enumerate(alzheimers_header[1:])]
samples_data = [[float(entry[i+1]) for entry in data_entries] for i in range(len(samples_labels))]

samples_matrix = build_matrix(len(samples_labels))

for i, sample in enumerate(samples_data):
    for j, sample2 in enumerate(samples_data):
        write_to_matrix(samples_matrix, i, j, chebyshev_distance(list(sample), list(sample2)))

alzheimers_samples_mst = GMLBuilder("alzheimers_samples_mst.gml")
mst_prim(samples_matrix, alzheimers_samples_mst, samples_labels)
alzheimers_samples_mst.write()

# b) Minimum Spanning Tree (MST) for proteins (0.5 marks)

proteins: List[Tuple[str, List[float]]] = [(entry[0], [float(i) for i in entry[1:]]) for entry in data_entries]
proteins_labels = [entry[0] for entry in proteins]
proteins_data = [entry[1] for entry in proteins]

proteins_matrix = build_matrix(len(proteins_data), default=0.0)

for i, sample in enumerate(proteins_data):
    for j, sample2 in enumerate(proteins_data):
        write_to_matrix(proteins_matrix, i, j, chebyshev_distance(list(sample), list(sample2)))


alzheimers_proteins_mst = GMLBuilder("alzheimers_proteins_mst.gml")
mst_prim(proteins_matrix, alzheimers_proteins_mst, proteins_labels)
alzheimers_proteins_mst.write()

# c) The Relative neighbourhood graph (RNG) for samples  (1 mark)
alzheimers_samples_rng = GMLBuilder("alzheimers_samples_rng.gml")
relative_neighborhood_graph(samples_matrix, alzheimers_samples_rng, samples_labels)
alzheimers_samples_rng.write()

# d) The Relative neighbourhood graph (RNG) for proteins (1 mark)
alzheimers_proteins_rng = GMLBuilder("alzheimers_proteins_rng.gml")
relative_neighborhood_graph(proteins_matrix, alzheimers_proteins_rng, proteins_labels)
alzheimers_proteins_rng.write()

# Exercise 3 (3 marks). Using the Alzheimer’s Disease Training dataset:
# a) Implement a feature selection technique that selects a subset of features (from the total of 120 measured proteins) (1 mark). 

# To make feature selection easier I'll be converting all the floats to ordered ints
samples_data_normalized: List[List[int]] = [[0 for _ in range(len(samples_data[0]))] for _ in range(len(samples_data))]
number_of_buckets = 4
# feature: List[float] (lower bounds)
buckets_mapping: Dict[int, List[float]] = {}
for column in range(len(samples_data[0])):
    sorted_data = sorted([entry[column] for entry in samples_data])
    # split into n buckets
    buckets = [sorted_data[int(len(sorted_data) * i / number_of_buckets)] for i in range(0, number_of_buckets)]
    buckets_mapping[column] = buckets
    for i, row in enumerate(samples_data):
        bucket = 0
        if row[column] < buckets[0]:
            bucket = 0
        elif row[column] < buckets[1]:
            bucket = 1
        elif row[column] < buckets[2]:
            bucket = 2
        else:
            bucket = 3

        samples_data_normalized[i][column] = bucket

print(buckets_mapping)

print(samples_data_normalized[0])

# Checking pearson correlation for each feature and the class
correlations = []
print(samples_data_normalized[0])
for i in range(len(samples_data_normalized[0])):
    flattend = [int(entry[i]) for entry in samples_data_normalized]
    pearson = pearson_correlation(flattend, [1 if label.startswith("AD") else 0 for label in samples_labels])
    correlations.append((i, pearson, proteins_labels[i]))

cut_off = 0.3
correlations = [correlation for correlation in correlations if abs(correlation[1]) > cut_off]

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(len(correlations), correlations) # 14 features with correlation > 0.3

selection = set([correlation[0] for correlation in correlations])

def reduce_data(data: List[List[int]], selection: set) -> List[List[int]]:
    return [[entry[i] for i in selection] for entry in data]

print(samples_data_normalized[0])
samples_data_reduced = reduce_data(samples_data_normalized, selection)
print(samples_data_reduced[0])


# b) Implement a classifier system that “learns” from the Training set (1 mark). 
# k-nn is implemneted in utils.py
samples_data_reduced_matrix = build_matrix(len(samples_data_reduced))
for i, sample in enumerate(samples_data_reduced):
    for j, sample2 in enumerate(samples_data_reduced):
        write_to_matrix(samples_data_reduced_matrix, i, j, euclidean_distance(list(sample), list(sample2)))

# copy of the pre computed matrix but with an additional column for the new entry
classification_matrix = build_matrix(len(samples_data_reduced) + 1)
for i, column in enumerate(samples_data_reduced_matrix):
    for j, value in enumerate(column):
        write_to_matrix(classification_matrix, i, j, value)

class_mapping = {}
for i, data in enumerate(samples_data_reduced):
    class_mapping[i] = 1 if samples_labels[i].startswith("AD") else 0

def set_sample(samples_data: List[List[int]], classification_matrix: List[List[int]], sample: List[int]):
    # last index in classification_matrix is used
    for i, value in enumerate(samples_data):
        write_to_matrix(classification_matrix, len(samples_data), i, euclidean_distance(value, sample))

def classify(data_reduced: List[List[int]], classification_matrix: List[List[int]], class_mapping: Dict[int, int], sample: List[int]) -> int:
    set_sample(data_reduced, classification_matrix, sample)
    return k_nearest_neighbor_classification(classification_matrix, class_mapping, len(data_reduced), k_count=3)[0]

print("Classifying first sample in training")
print(classify(samples_data_reduced, classification_matrix, class_mapping, samples_data_reduced[0]), class_mapping[0])

# c) For the first three datasets (Training, Test, and the one of MCI-labelled samples), apply the feature selection method implemented in 3 (a)
# and use the reduced datasets to train the classifier implemented in 3(b).
with open("alzheimers_disease/test_set_ad.csv", "r", encoding="utf-8") as file:
    test_ad = file.read()
    # I will be classifying all non-ad as NDC and ignore classification between NDC and OD

test_ad = test_ad.split("\n")
test_ad_header = test_ad[0].split(",")
test_ad_entries = [i.split(",") for i in test_ad[1:] if i]  # if i for removing final empty

# samples are the columns
test_ad_labels = [f"{label}-{i}" for i, label in enumerate(test_ad_header[1:])]
test_ad_data = [[float(entry[i+1]) for entry in test_ad_entries] for i in range(len(test_ad_labels))]

# normalize
test_ad_data_normalized = [[0 for _ in range(len(test_ad_data[0]))] for _ in range(len(test_ad_data))]
for i, sample in enumerate(test_ad_data):
    for j, value in enumerate(sample):
        bucket = 0
        if value < buckets_mapping[j][0]:
            bucket = 0
        elif value < buckets_mapping[j][1]:
            bucket = 1
        elif value < buckets_mapping[j][2]:
            bucket = 2
        else:
            bucket = 3

        test_ad_data_normalized[i][j] = bucket

test_ad_data_reduced = reduce_data(test_ad_data_normalized, selection)
test_ad_actual_classes = [1 if label.startswith("AD") else 0 for label in test_ad_labels]

with open("alzheimers_disease/test_set_mci.csv", "r", encoding="utf-8") as file:
    test_mci = file.read()
    # I'll be using follow up as the target class

test_mci = test_mci.split("\n")
# I'm skipping the first row since its the "class" but for this dataset we only care about the follow up
test_mci_header = test_mci[1].split(",")
test_mci_entries = [i.split(",") for i in test_mci[2:] if i]  # if i for removing final empty

# samples are the columns
test_mci_labels = [f"{label}-{i}" for i, label in enumerate(test_mci_header[1:])]
test_mci_data = [[float(entry[i+1]) for entry in test_mci_entries] for i in range(len(test_mci_labels))]

# normalize
test_mci_data_normalized = [[0 for _ in range(len(test_mci_data[0]))] for _ in range(len(test_mci_data))]
for i, sample in enumerate(test_mci_data):
    for j, value in enumerate(sample):
        bucket = 0
        if value < buckets_mapping[j][0]:
            bucket = 0
        elif value < buckets_mapping[j][1]:
            bucket = 1
        elif value < buckets_mapping[j][2]:
            bucket = 2
        else:
            bucket = 3

        test_mci_data_normalized[i][j] = bucket

test_mci_data_reduced = reduce_data(test_mci_data_normalized, selection)
test_mci_actual_classes = [1 if label.startswith("AD") else 0 for label in test_mci_labels]

# Now calculate the performance of your classifier according to the following measures and discuss the results obtained:
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# combining the test sets and then iterating over them
for sample, actual_class in zip(test_ad_data_reduced + test_mci_data_reduced, test_ad_actual_classes + test_mci_actual_classes):
    classification = classify(samples_data_reduced, classification_matrix, class_mapping, sample)
    if classification == actual_class:
        if actual_class == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        if actual_class == 1:
            false_negatives += 1
        else:
            false_positives += 1

results = PerformanceMeasurer(true_positives, false_positives, true_negatives, false_negatives)

print("Results Alzheimer's Dataset")
print(results.sensitivity())
print(results.specificity())
print(results.accuracy())
print(results.f1_score())
print(results.matthews_correlation_coefficient())
print(results.youdens_j_statistic())


# Exercise 4) Repeat 2 and 3 with your own dataset
# I'll be using iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
# pylint complains about iris typing from sklearn but this is fine
# pylance also has complaints hence the type ignore-s
# pylint: disable=no-member
iris_data: List[List[float]] = [list(i) for i in iris.data] # type: ignore
iris_labels: List[int] = list(iris.target) # type: ignore
# pylint: disable=no-member
iris_labels_names: List[str] = [iris.target_names[i] for i in iris_labels] # type: ignore

# and for 2 b and d
iris_features = [[entry[i] for entry in iris_data] for i in range(len(iris_data[0]))]

# 2a (mst for entries)
iris_matrix = build_matrix(len(iris_data))
for i, sample in enumerate(iris_data):
    for j, sample2 in enumerate(iris_data):
        # samples distance is chebyshev distance which I consider ok for this
        write_to_matrix(iris_matrix, i, j, chebyshev_distance(sample, sample2))

iris_mst = GMLBuilder("iris_mst.gml")
mst_prim(iris_matrix, iris_mst, iris_labels_names)
iris_mst.write()

# 2b (mst for features)
# not really interesting since there are only 4 features in iris dataset
iris_features_matrix = build_matrix(len(iris_features))
for i, sample in enumerate(iris_features):
    for j, sample2 in enumerate(iris_features):
        write_to_matrix(iris_features_matrix, i, j, chebyshev_distance(sample, sample2))

iris_features_mst = GMLBuilder("iris_features_mst.gml")
mst_prim(iris_features_matrix, iris_features_mst, [f"Feature-{i}" for i in range(len(iris_features))])
iris_features_mst.write()

# 2c (rng for entries)
iris_rng = GMLBuilder("iris_rng.gml")
relative_neighborhood_graph(iris_matrix, iris_rng, iris_labels_names)
iris_rng.write()

# 2d (rng for features)
iris_features_rng = GMLBuilder("iris_features_rng.gml")
relative_neighborhood_graph(iris_features_matrix, iris_features_rng, [f"Feature-{i}" for i in range(len(iris_features))])
iris_features_rng.write()


# 3a (feature selection)

# normalizing the floats to ints
iris_data_normalized: List[List[int]] = [[0 for _ in range(len(iris_data[0]))] for _ in range(len(iris_data))]
number_of_buckets = 4
# feature: List[float] (lower bounds)
buckets_mapping: Dict[int, List[float]] = {}
for column in range(len(iris_data[0])):
    sorted_data = sorted([entry[column] for entry in iris_data])
    # split into n buckets
    buckets = [sorted_data[int(len(sorted_data) * i / number_of_buckets)] for i in range(0, number_of_buckets)]
    buckets_mapping[column] = buckets
    for i, row in enumerate(iris_data):
        bucket = 0
        if row[column] < buckets[0]:
            bucket = 0
        elif row[column] < buckets[1]:
            bucket = 1
        elif row[column] < buckets[2]:
            bucket = 2
        else:
            bucket = 3

        iris_data_normalized[i][column] = bucket


# apply correlation-based feature selection
correlations = []
for i in range(len(iris_data_normalized[0])):
    flattend = [int(entry[i]) for entry in iris_data_normalized]
    pearson = pearson_correlation(flattend, iris_labels)
    correlations.append((i, pearson, f"Feature-{i}"))

cut_off = 0.3
correlations = [correlation for correlation in correlations if abs(correlation[1]) > cut_off]

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(len(correlations), correlations) # 3 features (0,2,3) all have high correlation
# [(2, 0.8598562065918868, 'Feature-2'), (3, 0.833113617831571, 'Feature-3'), (0, 0.7470009847411124, 'Feature-0')]

selection = set([correlation[0] for correlation in correlations])
iris_data_reduced = reduce_data(iris_data_normalized, selection)

# 3b (classifier)
# I'm splitting the data into training and test sets (70/30 split)
from random import shuffle

combined = list(zip(iris_data_reduced, iris_labels))
shuffle(combined)
number_training = int(len(combined) * 0.7)
training = combined[:number_training]
test = combined[number_training:]

training_data, training_labels = ([entry[0] for entry in training], [entry[1] for entry in training])
test_data, test_labels = ([entry[0] for entry in test], [entry[1] for entry in test])

training_matrix = build_matrix(len(training_data) + 1)
for i, column in enumerate(training_data):
    for j, value in enumerate(column):
        write_to_matrix(training_matrix, i, j, value)

class_mapping = {i: label for i, label in enumerate(training_labels)}

print(classify(training_data, training_matrix, class_mapping, test_data[0]), test_labels[0]) # should be the same

# 3c (performance)
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for sample, actual_class in zip(test_data, test_labels):
    classification = classify(training_data, training_matrix, class_mapping, sample)
    if classification == actual_class:
        if actual_class == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        if actual_class == 1:
            false_negatives += 1
        else:
            false_positives += 1

results = PerformanceMeasurer(true_positives, false_positives, true_negatives, false_negatives)

print("Results Iris Dataset")
print(results.sensitivity())
print(results.specificity())
print(results.accuracy())
print(results.f1_score())
print(results.matthews_correlation_coefficient())
print(results.youdens_j_statistic())

print("Conclusion: very bad results ")

# Exercise 5) 
# a) Show that a solution P = {P1, ... P4} is compatible with Incumbend, Challenger
# P1 = 0 * 1 1 0    P2 = 1 0 1 0 0  P3 = 0 1 0 0 *  P4 = 0 * * 0 0 
with open("USPresidency.csv", "r", encoding="utf-8") as file:
    data = file.read()

data = data.split("\n")
us_presidency_labels = data[0].split(",")
us_presidency = [i.split(",") for i in data[1:] if i]  # if i for removing final
us_presidency_reduced = [[entry[4], entry[5], entry[8], entry[9], entry[12], entry[13]] for entry in us_presidency]
us_presidency_reduced = [[int(entry) for entry in row] for row in us_presidency_reduced]

def pattern_matches(pattern: List[Union[int, str]], row: List[int]) -> bool:
    return all(pattern[j] == "*" or pattern[j] == row[j] for j in range(len(pattern)))

def is_valid_pattern_set(patterns: List[List[Union[int, str]]], rows: List[List[int]]) -> bool:
    # discovered pattern index to the class it corresponds to
    pattern_corresponds = {}
    # class: entry_index: applicable_pattern (-1 if none)
    class_mappings = {}
    for i, entry in enumerate(rows):
        if entry[-1] not in class_mappings:
            class_mappings[entry[-1]] = {}
        class_mappings[entry[-1]][i] = -1
        for j, pattern in enumerate(patterns):
            if pattern_matches(pattern, entry):
                class_mappings[entry[-1]][i] = j
                if j not in pattern_corresponds:
                    pattern_corresponds[j] = entry[-1]
                    continue
                if pattern_corresponds[j] != entry[-1]:
                    # A pattern corresponds to multiple classes!
                    return False
    
    # all patterns should correspond to a single class
    # (comparing all patterns to match to the same class as the first pattern)
    if not all(i in pattern_corresponds and pattern_corresponds[i] == pattern_corresponds[0] for i in range(len(patterns))):
        return False
    
    # for the target class, all entries of that class should be covered by a pattern
    if -1 in class_mappings[pattern_corresponds[0]].values():
        return False
    
    return True

patterns_provided = [
    [0, "*", 1, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, "*"],
    [0, "*", "*", 0, 0],
]

print(is_valid_pattern_set(patterns_provided, us_presidency_reduced))  # True
# b) Find your own pattern identification for L=4
# targetting the challenger class
my_patterns = [
    # for 1960, all zeros except for q12 which is similar to 1928 which is all zeros
    # the only q12 with incumbent also has q5 as 1
    # so:
    [1, "*", 0, "*", "*"],
    ["*", "*", "*", 1, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0]
]

print(is_valid_pattern_set(my_patterns, us_presidency_reduced))  # True
