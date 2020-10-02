import random
from typing import Callable, List

from PIL import Image
from torch import Tensor

from deepclustering2.augment import pil_augment
from deepclustering2.decorator import FixRandomSeed


class SequentialWrapper:

    def __init__(
        self,
        comm_transform: Callable[[Image.Image], Image.Image] = None,
        img_transform: Callable[[Image.Image], Tensor] = pil_augment.ToTensor(),
        target_transform: Callable[[Image.Image], Tensor] = pil_augment.ToLabel()
    ) -> None:
        """
        :param comm_transform: common geo-transformation
        :param img_transform: transformation only applied for images
        :param target_transform: transformation only applied for targets
        """
        self._comm_transform = comm_transform
        self._img_transform = img_transform
        self._target_transform = target_transform

    def __call__(self, imgs: List[Image.Image], targets: List[Image.Image] = None, comm_seed=None, img_seed=None,
                 target_seed=None):
        _comm_seed: int = int(random.randint(0, int(1e5))) if comm_seed is None else int(comm_seed)  # type ignore
        imgs_after_comm, targets_after_comm = imgs, targets
        if self._comm_transform:
            imgs_after_comm, targets_after_comm = [], []
            for img in imgs:
                with FixRandomSeed(_comm_seed):
                    img_ = self._comm_transform(img)
                    imgs_after_comm.append(img_)
            if targets:
                for target in targets:
                    with FixRandomSeed(_comm_seed):
                        target_ = self._comm_transform(target)
                        targets_after_comm.append(target_)
        imgs_after_img_transform = []
        targets_after_target_transform = []
        _img_seed: int = int(random.randint(0, int(1e5))) if img_seed is None else int(img_seed)  # type ignore
        for img in imgs_after_comm:
            with FixRandomSeed(_img_seed):
                img_ = self._img_transform(img)
                imgs_after_img_transform.append(img_)

        _target_seed: int = int(random.randint(0, int(1e5))) if target_seed is None else int(target_seed)  # type ignore
        if targets_after_comm:
            for target in targets_after_comm:
                with FixRandomSeed(_target_seed):
                    target_ = self._target_transform(target)
                    targets_after_target_transform.append(target_)

        if targets is None:
            targets_after_target_transform = None

        if targets_after_target_transform is None:
            return imgs_after_img_transform
        return [*imgs_after_img_transform, *targets_after_target_transform]

    def __repr__(self):
        return (
            f"comm_transform:{self._comm_transform}\n"
            f"img_transform:{self._img_transform}.\n"
            f"target_transform: {self._target_transform}"
        )


class SequentialWrapperTwice(SequentialWrapper):

    def __init__(self, comm_transform: Callable[[Image.Image], Image.Image] = None,
                 img_transform: Callable[[Image.Image], Tensor] = pil_augment.ToTensor(),
                 target_transform: Callable[[Image.Image], Tensor] = pil_augment.ToLabel(),
                 total_freedom=True) -> None:
        """
        :param total_freedom: if True, the two-time generated images are using different seeds for all aspect,
                              otherwise, the images are used different random seed only for img_seed
        """
        super().__init__(comm_transform, img_transform, target_transform)
        self._total_freedom = total_freedom

    def __call__(self, imgs: List[Image.Image], targets: List[Image.Image] = None, global_seed=None, **kwargs):
        global_seed = int(random.randint(0, int(1e5))) if global_seed is None else int(global_seed)  # type ignore
        with FixRandomSeed(global_seed):
            comm_seed1, comm_seed2 = int(random.randint(0, int(1e5))), int(random.randint(0, int(1e5)))
            img_seed1, img_seed2 = int(random.randint(0, int(1e5))), int(random.randint(0, int(1e5)))
            target_seed1, target_seed2 = int(random.randint(0, int(1e5))), int(random.randint(0, int(1e5)))
            if self._total_freedom:
                return [
                    super().__call__(imgs, targets, comm_seed1, img_seed1, target_seed1),
                    super().__call__(imgs, targets, comm_seed2, img_seed2, target_seed2),
                ]
            return [
                super().__call__(imgs, targets, comm_seed1, img_seed1, target_seed1),
                super().__call__(imgs, targets, comm_seed1, img_seed2, target_seed1),
            ]
