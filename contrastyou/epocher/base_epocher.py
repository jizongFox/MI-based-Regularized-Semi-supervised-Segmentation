import random
from typing import Union, Tuple

import torch
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher, proxy_trainer  # noqa
from deepclustering2.loss import simplex
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, UniversalDice
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.type import T_loss, T_optim, T_loader
from deepclustering2.utils import class2one_hot
from torch import nn
from torch.utils.data import DataLoader

from ._utils import preprocess_input_with_single_transformation, preprocess_input_with_twice_transformation


class EvalEpoch(_Epocher):

    def __init__(self, model: Union[Model, nn.Module], val_loader: DataLoader, sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        """
        :param model: Model or nn.Module instance, network
        :param val_loader: validation loader that is an instance of DataLoader, without infinitesampler
        :param sup_criterion: Supervised loss to record the val_loss
        :param cur_epoch: current epoch to record
        :param device: cuda or cpu
        """
        super().__init__(model, cur_epoch, device)
        assert isinstance(val_loader, DataLoader), f"`val_loader` should be an instance of `DataLoader`, " \
                                                   f"given {val_loader.__class__.__name__}"
        assert callable(sup_criterion), f"sup_criterion must be callable, given {sup_criterion.__class__.__name__}"
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    @classmethod
    def create_from_trainer(cls, trainer):
        pass

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert not self._model.training, self._model.training
        with tqdm(self._val_loader).set_desc_from_epocher(self) as indicator:
            for i, data in enumerate(indicator):
                images, targets, filename, partiton_list, group_list = self._preprocess_data(data, self._device)
                predict_logits = self._model(images)
                assert not simplex(predict_logits), predict_logits.shape
                onehot_targets = class2one_hot(targets.squeeze(1), 4)
                loss = self._sup_criterion(predict_logits.softmax(1), onehot_targets, disable_assert=True)
                self.meters["sup_loss"].add(loss.item())
                self.meters["ds"].add(predict_logits.max(1)[1], targets.squeeze(1), group_name=list(group_list))
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        report_dict = self.meters.tracking_status()
        return report_dict, report_dict["ds"]["DSC_mean"]

    @staticmethod
    def _preprocess_data(data, device, non_blocking=True):
        return preprocess_input_with_single_transformation(data, device, non_blocking)


class SimpleFineTuneEpoch(_Epocher):
    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, num_batches: int = 100,
                 sup_criterion: T_loss = None, cur_epoch=0, device="cpu") -> None:
        super().__init__(model, cur_epoch, device)
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        assert callable(sup_criterion), sup_criterion
        self._labeled_loader = labeled_loader
        self._sup_criterion = sup_criterion
        self._num_batches = num_batches
        self._optimizer = optimizer

    @classmethod
    def create_from_trainer(cls, trainer):
        return cls(
            model=trainer._model, optimizer=trainer._optimizer, labeled_loader=trainer._fine_tune_loader_iter,
            sup_criterion=trainer._sup_criterion, num_batches=trainer._num_batches, cur_epoch=trainer._cur_epoch,
            device=trainer._device
        )

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        report_dict: EpochResultDict
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, label_data in zip(indicator, self._labeled_loader):
                (labelimage, labeltarget), _, filename, partition_list, group_list \
                    = self._preprocess_data(label_data, self._device)
                predict_logits = self._model(labelimage)
                assert not simplex(predict_logits), predict_logits

                onehot_ltarget = class2one_hot(labeltarget.squeeze(1), 4)
                sup_loss = self._sup_criterion(predict_logits.softmax(1), onehot_ltarget)

                self._optimizer.zero_grad()
                sup_loss.backward()
                self._optimizer.step()

                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["ds"].add(predict_logits.max(1)[1], labeltarget.squeeze(1),
                                          group_name=list(group_list))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return preprocess_input_with_twice_transformation(data, device)


class MeanTeacherEpocher(SimpleFineTuneEpoch):

    def __init__(self, model: nn.Module, teacher_model: nn.Module, optimizer: T_optim, labeled_loader: T_loader,
                 tra_loader: T_loader, num_batches: int = 100, sup_criterion: T_loss = None,
                 reg_criterion: T_loss = None, cur_epoch=0, device="cpu", transform_axis=None,
                 reg_weight: float = 0.0, ema_updater=None) -> None:
        super().__init__(model, optimizer, labeled_loader, num_batches, sup_criterion, cur_epoch, device)
        self._teacher_model = teacher_model
        assert callable(reg_criterion), reg_weight
        self._reg_criterion = reg_criterion
        self._tra_loader = tra_loader
        self._transformer = TensorRandomFlip(transform_axis)
        print(self._transformer)
        self._reg_weight = float(reg_weight)
        assert ema_updater
        self._ema_updater = ema_updater

    @classmethod
    def create_from_trainer(cls, trainer):
        return cls(model=trainer._model, teacher_model=trainer._teacher_model, optimizer=trainer._optimizer,
                   labeled_loader=trainer._fine_tune_loader_iter, tra_loader=trainer._pretrain_loader,
                   sup_criterion=trainer._sup_criterion, reg_criterion=trainer._reg_criterion,
                   num_batches=trainer._num_batches,
                   cur_epoch=trainer._cur_epoch, device=trainer._device, transform_axis=trainer._transform_axis,
                   reg_weight=trainer._reg_weight, ema_updater=trainer._ema_updater)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("reg_weight", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        self._teacher_model.train()
        assert self._model.training, self._model.training
        assert self._teacher_model.training, self._teacher_model.training
        self.meters["lr"].add(self._optimizer.param_groups[0]["lr"])
        self.meters["reg_weight"].add(self._reg_weight)
        report_dict: EpochResultDict

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, label_data, all_data in zip(indicator, self._labeled_loader, self._tra_loader):
                (labelimage, labeltarget), _, filename, partition_list, group_list \
                    = self._preprocess_data(label_data, self._device)
                (unlabelimage, _), *_ = self._preprocess_data(label_data, self._device)

                seed = random.randint(0, int(1e6))
                with FixRandomSeed(seed):
                    unlabelimage_tf = torch.stack([self._transformer(x) for x in unlabelimage], dim=0)
                assert unlabelimage_tf.shape == unlabelimage.shape

                student_logits = self._model(torch.cat([labelimage, unlabelimage_tf], dim=0))
                if simplex(student_logits):
                    raise RuntimeError("output of the model should be logits, instead of simplex")
                student_sup_logits, student_unlabel_logits_tf = student_logits[:len(labelimage)], \
                                                                student_logits[len(labelimage):]

                with torch.no_grad():
                    teacher_unlabel_logits = self._teacher_model(unlabelimage)
                with FixRandomSeed(seed):
                    teacher_unlabel_logits_tf = torch.stack([self._transformer(x) for x in teacher_unlabel_logits])
                assert teacher_unlabel_logits.shape == teacher_unlabel_logits_tf.shape

                # calcul the loss
                onehot_ltarget = class2one_hot(labeltarget.squeeze(1), 4)
                sup_loss = self._sup_criterion(student_sup_logits.softmax(1), onehot_ltarget)

                reg_loss = self._reg_criterion(student_unlabel_logits_tf.softmax(1),
                                               teacher_unlabel_logits_tf.detach().softmax(1))
                total_loss = sup_loss + self._reg_weight * reg_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                # update ema
                self._ema_updater(ema_model=self._teacher_model, student_model=self._model)

                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["reg_loss"].add(reg_loss.item())
                    self.meters["ds"].add(student_sup_logits.max(1)[1], labeltarget.squeeze(1),
                                          group_name=list(group_list))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
        return report_dict
