import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    counts = Counter(labels)
    total = len(labels)

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total_entropy = calculate_entropy([e[target_attr] for e in examples])
    total = len(examples)

    weighted_entropy = 0.0
    values = sorted(set(e[attr] for e in examples))

    for value in values:
        subset = [e for e in examples if e[attr] == value]
        weight = len(subset) / total
        weighted_entropy += weight * calculate_entropy(
            [e[target_attr] for e in subset]
        )

    return total_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(e[target_attr] for e in examples)
    return sorted(counts, key=lambda x: (-counts[x], x))[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    # If all examples belong to the same class
    labels = [e[target_attr] for e in examples]
    if len(set(labels)) == 1:
        return labels[0]

    # If no attributes remain
    if not attributes:
        return majority_class(examples, target_attr)

    # Select attribute with maximum information gain
    best_attr = max(
        attributes,
        key=lambda attr: calculate_information_gain(examples, attr, target_attr)
    )

    tree = {best_attr: {}}
    remaining = [attr for attr in attributes if attr != best_attr]

    # Split examples by sorted attribute values
    for value in sorted(set(e[best_attr] for e in examples)):
        subset = [e for e in examples if e[best_attr] == value]

        tree[best_attr][value] = learn_decision_tree(
            subset, remaining, target_attr
        )

    return tree