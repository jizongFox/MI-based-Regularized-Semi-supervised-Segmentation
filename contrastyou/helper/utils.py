# dictionary helper functions
import collections
from typing import Union, Dict

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def toDataLoaderIterator(loader_or_iter: Union[DataLoader, _BaseDataLoaderIter]):
    if not isinstance(loader_or_iter, (_BaseDataLoaderIter, DataLoader)):
        raise TypeError(f"{loader_or_iter} should an instance of DataLoader or _BaseDataLoaderIter, "
                        f"given {loader_or_iter.__class__.__name__}.")
    return loader_or_iter if isinstance(loader_or_iter, _BaseDataLoaderIter) else iter(loader_or_iter)


# make a flatten dictionary to be printablely nice.
def nice_dict(input_dict: Dict[str, Union[int, float]]) -> str:
    """
    this function is to return a nice string to dictionary displace propose.
    :param input_dict: dictionary
    :return: string
    """
    assert isinstance(
        input_dict, dict
    ), f"{input_dict} should be a dict, given {type(input_dict)}."
    is_flat_dict = True
    for k, v in input_dict.items():
        if isinstance(v, dict):
            is_flat_dict = False
            break
    flat_dict = input_dict if is_flat_dict else flatten_dict(input_dict, sep="")
    string_list = [f"{k}:{v:.3f}" for k, v in flat_dict.items()]
    return ", ".join(string_list)


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def multiply_iter(iter_a, iter_b):
    return [x * y for x, y in zip(iter_a, iter_b)]


def weighted_average_iter(a_list, weight_list):
    sum_weight = sum(weight_list)+1e-16
    return sum(multiply_iter(a_list, weight_list)) / sum_weight
