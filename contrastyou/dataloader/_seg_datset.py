import random
from abc import abstractmethod, ABCMeta
from copy import deepcopy as dcp
from typing import Union, List, Set

from torch.utils.data.sampler import Sampler


class ContrastDataset(metaclass=ABCMeta):
    """
    each patient has 2 code, the first code is the group_name, which is the patient id
    the second code is the partition code, indicating the position of the image slice.
    All patients should have the same partition numbers so that they can be aligned.
    For ACDC dataset, the ED and ES ventricular volume should be considered further
    """

    @abstractmethod
    def _get_partition(self, *args) -> Union[str, int]:
        """get the partition of a 2D slice given its index or filename"""
        pass

    @abstractmethod
    def _get_group(self, *args) -> Union[str, int]:
        """get the group name of a 2D slice given its index or filename"""
        pass

    @abstractmethod
    def show_paritions(self) -> List[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        pass

    def show_parition_set(self) -> Set[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        return set(self.show_paritions())

    @abstractmethod
    def show_groups(self) -> List[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        pass

    def show_group_set(self) -> Set[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        return set(self.show_groups())


class ContrastBatchSampler(Sampler):
    """
    This class is going to realize the sampling for different patients and from the same patients
    `we form batches by first randomly sampling m < M volumes. Then, for each sampled volume, we sample one image per
    partition resulting in S images per volume. Next, we apply a pair of random transformations on each sampled image and
    add them to the batch
    """

    class _SamplerIterator:

        def __init__(self, group2index, partion2index, group_sample_num=4, partition_sample_num=1,
                     shuffle=False) -> None:
            self._group2index, self._partition2index = dcp(group2index), dcp(partion2index)

            assert 1 <= group_sample_num <= len(self._group2index.keys()), group_sample_num
            self._group_sample_num = group_sample_num
            self._partition_sample_num = partition_sample_num
            self._shuffle = shuffle

        def __iter__(self):
            return self

        def __next__(self):
            batch_index = []
            cur_gsamples = random.sample(self._group2index.keys(), self._group_sample_num)
            assert isinstance(cur_gsamples, list), cur_gsamples
            # for each gsample, sample at most partition_sample_num slices per partion
            for cur_gsample in cur_gsamples:
                gavailableslices = self._group2index[cur_gsample]
                for savailbleslices in self._partition2index.values():
                    sampled_slices = random.sample(sorted(set(gavailableslices) & set(savailbleslices)),
                                                   self._partition_sample_num)
                    batch_index.extend(sampled_slices)
            if self._shuffle:
                random.shuffle(batch_index)
            return batch_index

    def __init__(self, dataset: ContrastDataset, group_sample_num=4, partition_sample_num=1, shuffle=False) -> None:
        self._dataset = dataset
        filenames = dcp(list(dataset._filenames.values())[0])
        group2index = {}
        partiton2index = {}
        for i, filename in enumerate(filenames):
            group = dataset._get_group(filename)
            if group not in group2index:
                group2index[group] = []
            group2index[group].append(i)
            partition = dataset._get_partition(filename)
            if partition not in partiton2index:
                partiton2index[partition] = []
            partiton2index[partition].append(i)
        self._group2index = group2index
        self._partition2index = partiton2index
        self._group_sample_num = group_sample_num
        self._partition_sample_num = partition_sample_num
        self._shuffle = shuffle

    def __iter__(self):
        return self._SamplerIterator(self._group2index, self._partition2index, self._group_sample_num,
                                     self._partition_sample_num, shuffle=self._shuffle)

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore
