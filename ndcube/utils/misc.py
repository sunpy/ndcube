__all__ = ['unique_sorted']


def unique_sorted(iterable):
    """
    Return unique values in the order they are first encountered in the iterable.
    """
    lookup = set()  # a temporary lookup set
    return [ele for ele in iterable if ele not in lookup and lookup.add(ele) is None]
