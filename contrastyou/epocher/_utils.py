import os
import warnings
from typing import List, Union

import numpy as np
from skimage.io import imsave
from torch import Tensor

from deepclustering2.type import to_device, torch


def unique_mapping(name_list):
    unique_map = np.unique(name_list)
    mapping = {}
    for i, u in enumerate(unique_map):
        mapping[u] = i
    return [mapping[n] for n in name_list]


def _string_list_adding(list1, list2):
    assert len(list1) == len(list2)
    return [x + "_" + y for x, y in zip(list1, list2)]


def preprocess_input_with_twice_transformation(data, device, non_blocking=True):
    [(image, target), (image_tf, target_tf)], filename, partition_list, group_list = \
        to_device(data[0], device, non_blocking), data[1], data[2], data[3]
    return (image, target), (image_tf, target_tf), filename, partition_list, group_list


def preprocess_input_with_single_transformation(data, device, non_blocking=True):
    return data[0][0].to(device, non_blocking=non_blocking), data[0][1].to(device, non_blocking=non_blocking), data[1], \
           data[2], data[3]


def unfold_position(features: torch.Tensor, partition_num=(4, 4), ):
    b, c, h, w = features.shape
    block_h = h // partition_num[0]
    block_w = w // partition_num[1]
    h_index = torch.arange(0, h - block_h + 1, block_h)
    w_index = torch.arange(0, w - block_w + 1, block_w)
    result = []
    result_flag = []
    for h in h_index:
        for w in w_index:
            result.append(features[:, :, h:h + block_h, w:w + block_w])
            for _ in range(b):
                result_flag.append((int(h), int(w)))
    return torch.cat(result, dim=0), result_flag


class GlobalLabelGenerator:

    def __init__(self, contrastive_on_patient=False, contrastive_on_partition=True) -> None:
        self._contrastive_on_patient = contrastive_on_patient
        self._contrastive_on_partition = contrastive_on_partition

    def __call__(self, partition_list: List[str], patient_list: List[str]) -> List[int]:
        assert len(partition_list) == len(patient_list), (len(partition_list), len(patient_list))
        batch_size = len(partition_list)

        final_string = [""] * batch_size
        if self._contrastive_on_patient:
            final_string = _string_list_adding(final_string, patient_list)

        if self._contrastive_on_partition:
            final_string = _string_list_adding(final_string, partition_list)

        return unique_mapping(final_string)


class LocalLabelGenerator(GlobalLabelGenerator):

    def __init__(self, ) -> None:
        super().__init__(True, True)

    def __call__(self, partition_list: List[str], patient_list: List[str], location_list: List[str]) -> List[int]:
        partition_list = [str(x) for x in partition_list]
        patient_list = [str(x) for x in patient_list]
        location_list = [str(x) for x in location_list]
        mul_factor = int(len(location_list) // len(patient_list))
        partition_list = partition_list * mul_factor
        patient_list = patient_list * mul_factor
        assert len(location_list) == len(partition_list)

        return super().__call__(_string_list_adding(patient_list, partition_list), location_list)


def _write_single_png(mask: Tensor, save_dir: str, filename: str):
    assert mask.shape.__len__() == 2, mask.shape
    mask = mask.cpu().detach().numpy()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        imsave(os.path.join(save_dir, (filename + ".png")), mask.astype(np.uint8))


def write_predict(predict_logit: Tensor, save_dir: str, filenames: Union[str, List[str]]):
    assert len(predict_logit.shape) == 4, predict_logit.shape
    if isinstance(filenames, str):
        filenames = [filenames, ]
    assert len(filenames) == len(predict_logit)
    predict_mask = predict_logit.max(1)[1]
    for m, f in zip(predict_mask, filenames):
        _write_single_png(m, os.path.join(save_dir, "pred"), f)


def write_img_target(image: Tensor, target: Tensor, save_dir: str, filenames: Union[str, List[str]]):
    if isinstance(filenames, str):
        filenames = [filenames, ]
    image = image.squeeze()
    target = target.squeeze()
    assert image.shape == target.shape
    for img, f in zip(image, filenames):
        _write_single_png(img*255, os.path.join(save_dir, "img"), f)
    for targ, f in zip(target, filenames):
        _write_single_png(targ, os.path.join(save_dir, "gt"), f)


if __name__ == '__main__':
    features = torch.randn(10, 3, 256, 256, requires_grad=True)

    a = unfold_position(features, partition_num=(3, 3))
    print()
