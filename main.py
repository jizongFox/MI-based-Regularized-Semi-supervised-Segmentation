from scipy.sparse import issparse  # noqa

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
from semi_seg.dataloader_helper import get_dataloaders

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

# set reproducibility
set_benchmark(config.get("RandomSeed", 1))

labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]

model = UNet(**config["Arch"])

trainer = Trainer(
    model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
    val_loader=val_loader, sup_criterion=KL_div(),
    configuration={**cmanager.config, **{"GITHASH": cur_githash}},
    **config["Trainer"]
)
trainer.init()
checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(checkpoint, strict=False)
trainer.start_training()
# trainer.inference(checkpoint=checkpoint)
