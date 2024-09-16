from typing import Dict, List, Tuple
from utils import GMLBuilder, build_matrix, feature_selection, mst_prim, proteins_distance, relative_neighborhood_graph, samples_distance, write_to_matrix, pretty_print


# 1. Perform Explanatory Data Analysis (EDA) using Students’ Academic Performance Dataset (xAPI-Edu-Data.csv)
# a. Is there any strong correlation between variables? Pearson, Scatter plot, Correlogram
with open("xAPI-Edu-Data.csv", "r", encoding="utf-8") as file:
    data = file.read()

data = data.split("\n")
students_academic_performance_labels = data[0].split(",")
students_academic_performance = [i.split(",") for i in data[1:] if i]  # if i for removing final empty line

# Sanity check
print(len(students_academic_performance))

# Pearson requires numerical data
# encoding categorical data (I'm not taking into account ordinality of )
# label: (encodings, next number)
encodings: Dict[str, Tuple[Dict[str, int], int]] = {}

for i, label in enumerate(students_academic_performance_labels):
    if students_academic_performance[0][i].isnumeric():  # if it's a number, we don't need it
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
    # TODO: Does mean make sense for categorical data?
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

# 3) Girls seem to have better overall performance than boys (0.2 marks)
# 4) Boys are generally a bit more open to discussions, visiting resources, and raising hands (0.2 marks)
# 4) Those who participated more (higher counts in Discussion, Announcement Views, Raised Hands), usually performs better (0.2 marks)


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
        write_to_matrix(samples_matrix, i, j, samples_distance(list(sample), list(sample2)))

alzheimers_samples_mst = GMLBuilder("alzheimers_samples_mst.gml")
mst_prim(samples_matrix, alzheimers_samples_mst, samples_labels)
alzheimers_samples_mst.write()

# b) Minimum Spanning Tree (MST) for proteins (0.5 marks)

proteins: List[Tuple[str, List[float]]] = [(entry[0], [float(i) for i in entry[1:]]) for entry in data_entries]
proteins_labels = [entry[0] for entry in proteins]
proteins_data = [entry[1] for entry in proteins]

proteins_matrix = build_matrix(len(proteins_data))

for i, sample in enumerate(proteins_data):
    for j, sample2 in enumerate(proteins_data):
        write_to_matrix(proteins_matrix, i, j, proteins_distance(list(sample), list(sample2)))


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

# AD and NDC
labels = ["AD", "NDC"]
data_groups = [[], []]
for i, entry in enumerate(samples_data):
    index = 0 if samples_labels[i].startswith("AD") else 1
    data_groups[index].append(entry)

print(feature_selection(data_groups, samples_labels))
