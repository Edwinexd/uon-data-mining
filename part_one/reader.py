
from typing import Dict, List, Tuple

# Year,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Target
class Election:
    year: int
    q1: int
    q2: int
    q3: int
    q4: int
    q5: int
    q6: int
    q7: int
    q8: int
    q9: int
    q10: int
    q11: int
    q12: int
    target: int

    def __init__(self, year: int, q1: int, q2: int, q3: int, q4: int, q5: int, q6: int, q7: int, q8: int, q9: int, q10: int, q11: int, q12: int, target: int):
        self.year = year
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7
        self.q8 = q8
        self.q9 = q9
        self.q10 = q10
        self.q11 = q11
        self.q12 = q12
        self.target = target

    def to_binary_vector(self) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
        return (self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7, self.q8, self.q9, self.q10, self.q11, self.q12)


def get_elections() -> List[Election]:
    elections = []
    with open("USPresidency.csv", "r", encoding="utf-8") as f:
        for num, line in enumerate(f):
            # Header of csv
            if num == 0:
                continue
            parts = line.split(",")
            elections.append(Election(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]), int(parts[10]), int(parts[11]), int(parts[12]), int(parts[13])))

    return elections

