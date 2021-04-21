from contextlib import contextmanager
from copy import deepcopy

import numpy as np
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.dataloader import ACDCSemiInterface, ACDCDataset
from semi_seg.augment import ACDCStrongTransforms

dataset_zoos = {
    "acdc": ACDCSemiInterface,

}
augment_zoos = {
    "acdc": ACDCStrongTransforms,

}


def get_dataloaders(config, group_val_patient=True):
    _config = deepcopy(config)
    dataset_name = _config["Data"].pop("name", "acdc")
    assert dataset_name in dataset_zoos.keys(), config["Data"]
    datainterface = dataset_zoos[dataset_name]
    augmentinferface = augment_zoos[dataset_name]

    data_manager = datainterface(root_dir=DATA_PATH, labeled_data_ratio=config["Data"]["labeled_data_ratio"],
                                 unlabeled_data_ratio=config["Data"]["unlabeled_data_ratio"])

    label_set, unlabel_set, val_set = data_manager._create_semi_supervised_datasets(  # noqa
        labeled_transform=augmentinferface.pretrain,
        unlabeled_transform=augmentinferface.pretrain,
        val_transform=augmentinferface.val
    )

    # labeled loader is with normal 2d slicing and InfiniteRandomSampler
    labeled_loader = DataLoader(
        label_set, sampler=InfiniteRandomSampler(
            label_set,
            shuffle=config["LabeledData"]["shuffle"]
        ),
        batch_size=config["LabeledData"]["batch_size"],
        num_workers=config["LabeledData"]["num_workers"],
        pin_memory=True
    )
    unlabeled_loader = DataLoader(
        unlabel_set, sampler=InfiniteRandomSampler(
            unlabel_set,
            shuffle=config["UnlabeledData"]["shuffle"]
        ),
        batch_size=config["UnlabeledData"]["batch_size"],
        num_workers=config["UnlabeledData"]["num_workers"],
        pin_memory=True
    )
    group_val_patient = group_val_patient if not dataset_name == "spleen" else False
    val_loader = DataLoader(
        val_set,
        batch_size=1 if group_val_patient else 4,
        batch_sampler=PatientSampler(
            val_set,
            grp_regex=val_set.dataset_pattern,
            shuffle=False
        ) if group_val_patient else None,
    )
    return labeled_loader, unlabeled_loader, val_loader


@contextmanager
def fix_numpy_seed(seed: int = 1):
    previous_state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(previous_state)


def create_val_loader(unlabeled_loader, test_loader):
    unlabeled_dataset = unlabeled_loader.dataset
    unlabeled_dataset: ACDCDataset
    patient_group = sorted(unlabeled_dataset.show_group_set())
    with fix_numpy_seed(1):
        val_patient = np.random.permutation(patient_group)[:5]

    def extract_patient(dataset, validated_patient):
        new_file_names = {}
        files_names = dataset._filenames  # noqa
        for k, v in files_names.items():
            new_file_names[k] = [x for x in v if dataset._get_group(x) in validated_patient]
        return new_file_names

    val_dataset = deepcopy(unlabeled_dataset)
    val_dataset._filenames = extract_patient(val_dataset, val_patient)
    test_dataset = test_loader.dataset
    val_dataset._transform = deepcopy(test_dataset._transform)

    group_val_patient = True

    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if group_val_patient else 4,
        batch_sampler=PatientSampler(
            val_dataset,
            grp_regex=val_dataset.dataset_pattern,
            shuffle=False
        ) if group_val_patient else None,
    )
    return val_loader
