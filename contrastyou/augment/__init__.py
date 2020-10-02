from torchvision import transforms

from contrastyou.augment.sequential_wrapper import SequentialWrapperTwice, SequentialWrapper
from deepclustering2.augment import pil_augment
from .tensor_affine_transform import AffineTensorTransform


class ACDCTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),  # interpolation to be nearest
        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.8, 1.3], contrast=[0.8, 1.3], saturation=[0.8, 1.3]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),  # interpolation to be nearest
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        comm_transform=pil_augment.CenterCrop(224)
    )


class ACDCStrongTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomRotation(30),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        comm_transform=pil_augment.CenterCrop(224)
    )




transform_dict = {"strong": ACDCStrongTransforms, "simple": ACDCTransforms}
