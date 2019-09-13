from typing import List


def _flatten(list_of_lists: List[List]) -> List:
    all_values = []
    for list in list_of_lists:
        all_values.extend(list)
    return all_values

def build_categories_index(categories: List[List]):
    # Flatten list of lists (or tuple of tuples) to a single list
    all_values = _flatten(categories)
    return build_index(all_values)

def build_index(values: List):
    return {value: i for i, value in enumerate(values)}



