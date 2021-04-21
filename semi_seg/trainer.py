import os
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Tuple

import torch
from deepclustering2 import optim
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer import Trainer
from deepclustering2.type import T_loader, T_loss
from torch import nn

from contrastyou import PROJECT_PATH
from semi_seg._utils import IICLossWrapper, ProjectorWrapper
from semi_seg.epocher import TrainEpocher, EvalEpocher, UDATrainEpocher, IICTrainEpocher, UDAIICEpocher, \
    InferenceEpocher

__all__ = ["trainer_zoos"]


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa

    feature_positions = ["Up_conv4", "Up_conv3"]

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, test_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base",
                 max_epoch: int = 100, num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init()
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _init(self):
        self.set_feature_positions(self._config["Trainer"]["feature_names"])
        feature_importance = self._config["Trainer"]["feature_importance"]
        assert isinstance(feature_importance, list), type(feature_importance)
        feature_importance = [float(x) for x in feature_importance]
        self._feature_importance = [x / sum(feature_importance) for x in feature_importance]
        assert len(self._feature_importance) == len(self.feature_positions)

    def _init_scheduler(self, optimizer):
        scheduler_dict = self._config.get("Scheduler", None)
        if scheduler_dict is None:
            return
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._config["Trainer"]["max_epoch"] - self._config["Scheduler"]["warmup_max"],
                eta_min=1e-7
            )
            scheduler = GradualWarmupScheduler(optimizer, scheduler_dict["multiplier"],
                                               total_epoch=scheduler_dict["warmup_max"],
                                               after_scheduler=scheduler)
            self._scheduler = scheduler

    def _init_optimizer(self):
        optim_dict = self._config["Optim"]
        self._optimizer = optim.__dict__[optim_dict["name"]](
            params=self._model.parameters(),
            **{k: v for k, v in optim_dict.items() if k != "name"}
        )

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = TrainEpocher(
            self._model, self._optimizer, self._labeled_loader, self._unlabeled_loader,
            self._sup_criterion, 0, self._num_batches, self._cur_epoch, self._device,
            feature_position=self.feature_positions, feature_importance=self._feature_importance
        )
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._model, val_loader=loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch(loader=self._val_loader)
                test_result, _ = self.eval_epoch(loader=self._test_loader)
            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result, test=test_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
            # save_checkpoint
            self.save(cur_score)
            # save storage result on csv file.
            self._storage.to_csv(self._save_dir)

    def inference(self, checkpoint=None):  # noqa
        if checkpoint is None:
            self.load_state_dict_from_path(os.path.join(self._save_dir, "best.pth"), strict=True)
        else:
            checkpoint = Path(checkpoint)
            if checkpoint.is_file():
                if not checkpoint.suffix == ".pth":
                    raise FileNotFoundError(checkpoint)
            else:
                assert checkpoint.exists()
                checkpoint = checkpoint / "best.pth"
            self.load_state_dict_from_path(str(checkpoint), strict=True)
        evaler = InferenceEpocher(self._model, val_loader=self._test_loader, sup_criterion=self._sup_criterion,
                                  cur_epoch=self._cur_epoch, device=self._device)
        evaler.set_save_dir(self._save_dir)
        result, cur_score = evaler.run()
        return result, cur_score

    @classmethod
    def set_feature_positions(cls, feature_positions):
        cls.feature_positions = feature_positions


class UDATrainer(SemiTrainer):

    def _init(self):
        super(UDATrainer, self)._init()
        config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._reg_weight = float(config["weight"])

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = UDATrainEpocher(self._model, self._optimizer, self._labeled_loader, self._unlabeled_loader,
                                  self._sup_criterion, reg_weight=self._reg_weight, num_batches=self._num_batches,
                                  cur_epoch=self._cur_epoch, device=self._device, reg_criterion=self._reg_criterion,
                                  feature_position=self.feature_positions, feature_importance=self._feature_importance)
        result = trainer.run()
        return result


class IICTrainer(SemiTrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["IICRegParameters"])
        self._projector_wrappers = ProjectorWrapper()
        self._projector_wrappers.init_encoder(
            feature_names=self.feature_positions,
            **config["EncoderParams"]
        )
        self._projector_wrappers.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        self._IIDSegWrapper = IICLossWrapper(
            feature_names=self.feature_positions,
            **config["LossParams"]
        )
        self._reg_weight = float(config["weight"])

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = IICTrainEpocher(
            self._model, self._projector_wrappers, self._optimizer, self._labeled_loader,
            self._unlabeled_loader, self._sup_criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, reg_weight=self._reg_weight,
            feature_position=self.feature_positions, feature_importance=self._feature_importance,
            IIDSegCriterionWrapper=self._IIDSegWrapper
        )
        result = trainer.run()
        return result

    def _init_optimizer(self):
        config = deepcopy(self._config["Optim"])
        self._optimizer = optim.__dict__[config["name"]](
            params=chain(self._model.parameters(), self._projector_wrappers.parameters()),
            **{k: v for k, v in config.items() if k != "name"}
        )


class UDAIICTrainer(IICTrainer):

    def _init(self):
        super(UDAIICTrainer, self)._init()
        self._iic_weight = deepcopy(self._reg_weight)
        self._reg_weight = 1.0
        UDA_config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[UDA_config["name"]]
        self._uda_weight = float(UDA_config["weight"])

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = UDAIICEpocher(
            self._model, self._projector_wrappers, self._optimizer, self._labeled_loader,
            self._unlabeled_loader, self._sup_criterion, self._reg_criterion, self._IIDSegWrapper,
            num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
            feature_position=self.feature_positions, cons_weight=self._uda_weight,
            iic_weight=self._iic_weight, feature_importance=self._feature_importance
        )
        result = trainer.run()
        return result


trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "udaiic": UDAIICTrainer,
}
