from scipy.sparse import issparse  # noqa
import numpy  # noqa`

_ = issparse  # noqa

from deepclustering2.loss import KL_div

from pathlib import Path
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet

from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash
from deepclustering2.utils import set_benchmark
from semi_seg.trainer import trainer_zoos
from semi_seg.dataloader_helper import get_dataloaders, create_val_loader

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

# set reproducibility
set_benchmark(config.get("RandomSeed", 1))

labeled_loader, unlabeled_loader, test_loader = get_dataloaders(config)
val_loader = create_val_loader(unlabeled_loader, test_loader)

"""
tra_dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=tra_transforms, verbose=True)
val_dataset = ACDCDataset(root_dir=DATA_PATH, mode="val", transforms=val_transforms, verbose=True)

# Create DataLoader  
# since you are doing full supervision, you can simply do the following.

train_loader = DataLoader(tra_dataset, num_workers=4, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=1, batch_size=6, shuffle=False)

"""


trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]

model = UNet(**config["Arch"])

trainer = Trainer(
    model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
    val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(),
    configuration={**cmanager.config, **{"GITHASH": cur_githash}},
    **config["Trainer"]
)
trainer.init()
checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(checkpoint, strict=False)
trainer.start_training()
# trainer.inference(checkpoint=checkpoint)


# model = Unet()
# labeled_loader()
# trainer= Trainer() # defines the behavoir of train epocher evaluation epoch and possible test epoch,
# save results and save checkpoint, resuming from checkpoint. but not directly the training. (model, optimizer, statistic_data)

# epocher = TrainEpocher:
# model, optimizer, cur_epoch_num -> Epocher()
# epocher.run() -> updated_model, and result.
# result-> trainer, trainer to save the result with the cur_batch_num.


# To modify or inherent Trainer to remove some componement related to semisupervised. by overriding some function.

# To modify (iherentat) Trainn epocher that remove some componment relate to semi sueprvised  and
#  "leave some space for iterative trainig."

# puting them together by modifying the calling of Trainer with teh current epocher.

# to make a fully supervision training working. no iterative