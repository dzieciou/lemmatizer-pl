from typing import List

from lemmatizer.morphology import Category


def build_categories_index(categories: List[Category]):
    # Flatten list of lists (or tuple of tuples) to a single list
    all_values = []
    for category in categories:
        all_values.extend(category.values)
    return build_index(all_values)

def build_index(values: List):
    return {value: i for i, value in enumerate(values)}



