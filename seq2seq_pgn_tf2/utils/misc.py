import copy

def merge_dict(dict1, dict2):
    """Merges :obj:`dict2` into :obj:`dict1`.
    Args:
        dict1: The base dictionary.
        dict2: The dictionary to merge.
    Returns:
        The merged dictionary :obj:`dict1`.
    """
    for key, value in dict2.items():
        if isinstance(value, dict):
            dict1[key] = merge_dict(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1

def clone_layer(layer):
    """Clones a layer."""
    return copy.deepcopy(layer)
