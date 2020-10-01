import random
from typing import Union, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from contrastyou.trainer._utils import ClusterHead  # noqa
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, UniversalDice, MultipleAverageValueMeter, \
    MeterInterface, SurfaceMeter
from deepclustering2.models import Model, ema_updater as EMA_Updater
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot, ExceptionIgnorer
from semi_seg._utils import FeatureExtractor, ProjectorWrapper, IICLossWrapper


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return self._model.num_classes


class EvalEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], val_loader: T_loader, sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(val_loader), cur_epoch=cur_epoch, device=device)
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        report_dict = EpochResultDict()
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_logits = self._model(val_img)
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(EvalEpocher):

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("hd", SurfaceMeter(C=C, report_axises=report_axis, metername="hausdorff"))
        return meters

    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert self._model.training is False, self._model.training
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_logits = self._model(val_img)
            # write image
            write_img_target(val_img, val_target, self._save_dir, file_path)
            write_predict(val_logits, self._save_dir, file_path, )
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            with ExceptionIgnorer(RuntimeError):
                self.meters["hd"].add(val_logits.max(1)[1], val_target.squeeze(1))
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]


class TrainEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_weight: float, num_batches: int, cur_epoch=0,
                 device="cpu", feature_position=None, feature_importance=None) -> None:
        super().__init__(model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._reg_weight = reg_weight
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)
        assert isinstance(feature_position, list) and isinstance(feature_position[0], str), feature_position
        assert isinstance(feature_importance, list) and isinstance(feature_importance[0],
                                                                   (int, float)), feature_importance
        self._feature_position = feature_position
        self._feature_importance = feature_importance

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:
            for i, labeled_data, unlabeled_data in zip(self._indicator, self._labeled_loader, self._unlabeled_loader):
                seed = random.randint(0, int(1e7))
                labeled_image, labeled_target, labeled_filename, _, label_group = \
                    self._unzip_data(labeled_data, self._device)
                unlabeled_image, unlabeled_target, *_ = self._unzip_data(unlabeled_data, self._device)
                with FixRandomSeed(seed):
                    unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image], dim=0)
                assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                    (unlabeled_image_tf.shape, unlabeled_image.shape)

                predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
                label_logits, unlabel_logits, unlabel_tf_logits = \
                    torch.split(
                        predict_logits,
                        [len(labeled_image), len(unlabeled_image), len(unlabeled_image_tf)],
                        dim=0
                    )
                with FixRandomSeed(seed):
                    unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)
                assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, \
                    (unlabel_logits_tf.shape, unlabel_tf_logits.shape)
                # supervised part
                onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
                sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
                # regularized part
                reg_loss = self.regularization(
                    unlabeled_tf_logits=unlabel_tf_logits,
                    unlabeled_logits_tf=unlabel_logits_tf,
                    seed=seed,
                    unlabeled_image=unlabeled_image,
                    unlabeled_image_tf=unlabeled_image_tf,
                )
                total_loss = sup_loss + self._reg_weight * reg_loss
                # gradient backpropagation
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                # recording can be here or in the regularization method
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, target, filename, partition, group

    def regularization(self, *args, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class UDATrainEpocher(TrainEpocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss, reg_weight: float,
                 num_batches: int, cur_epoch: int = 0, device="cpu", feature_position=None,
                 feature_importance=None) -> None:
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_weight, num_batches,
                         cur_epoch, device, feature_position, feature_importance)
        self._reg_criterion = reg_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        reg_loss = self._reg_criterion(
            unlabeled_tf_logits.softmax(1),
            unlabeled_logits_tf.softmax(1).detach()
        )
        self.meters["uda"].add(reg_loss.item())
        return reg_loss


class MIDLPaperEpocher(UDATrainEpocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss,
                 iic_segcriterion: IIDSegmentationSmallPathLoss, uda_weight: float, iic_weight,
                 num_batches: int, cur_epoch: int = 0, device="cpu", feature_position=None,
                 feature_importance=None) -> None:
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_criterion, 1.0,
                         num_batches, cur_epoch, device, feature_position, feature_importance)
        self._iic_segcriterion = iic_segcriterion
        self._iic_weight = iic_weight
        self._uda_weight = uda_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MIDLPaperEpocher, self)._configure_meters(meters)
        meters.register_meter("iic_mi", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        uda_loss = super(MIDLPaperEpocher, self).regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed, *args, **kwargs
        )
        iic_loss = self._iic_segcriterion(unlabeled_tf_logits, unlabeled_logits_tf.detach())
        self.meters["iic_mi"].add(iic_loss.item())
        return uda_loss * self._uda_weight + iic_loss * self._iic_weight


class IICTrainEpocher(TrainEpocher):

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def __init__(self, model: Union[Model, nn.Module], projectors_wrapper: ProjectorWrapper, optimizer: T_optim,
                 labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criterion: T_loss,
                 IIDSegCriterionWrapper: IICLossWrapper,
                 reg_weight: float, num_batches: int, cur_epoch: int = 0, device="cpu", feature_position=None,
                 feature_importance=None) -> None:
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_weight, num_batches,
                         cur_epoch, device, feature_position, feature_importance)
        assert projectors_wrapper.feature_names == self._feature_position
        self._projectors_wrapper = projectors_wrapper
        assert IIDSegCriterionWrapper.feature_names == self._feature_position
        self._IIDSegCriterionWrapper = IIDSegCriterionWrapper

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        # todo: adding projectors here.
        feature_names = self._fextractor._feature_names  # noqa
        unlabeled_length = len(unlabeled_tf_logits) * 2
        iic_losses_for_features = []

        for i, (inter_feature, projector, criterion) \
            in enumerate(zip(self._fextractor, self._projectors_wrapper, self._IIDSegCriterionWrapper)):

            unlabeled_features = inter_feature[len(inter_feature) - unlabeled_length:]
            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)

            if isinstance(projector, ClusterHead):  # features from encoder
                unlabeled_features_tf = unlabeled_features
            else:
                with FixRandomSeed(seed):
                    unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features],
                                                        dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)
                )])
            )
            _iic_loss_list = [criterion(x, y) for x, y in zip(prob1, prob2)]
            _iic_loss = average_iter(_iic_loss_list)
            iic_losses_for_features.append(_iic_loss)
        reg_loss = weighted_average_iter(iic_losses_for_features, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in iic_losses_for_features]
        )))

        return reg_loss


class UDAIICEpocher(IICTrainEpocher):

    def __init__(self, model: Union[Model, nn.Module], projectors_wrapper: ProjectorWrapper, optimizer: T_optim,
                 labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss,
                 IIDSegCriterion: T_loss, num_batches: int, cur_epoch: int = 0, device="cpu",
                 feature_position=None, feature_importance=None, cons_weight=1,
                 iic_weight=0.1) -> None:
        super().__init__(model, projectors_wrapper, optimizer, labeled_loader, unlabeled_loader, sup_criterion,
                         IIDSegCriterion, 1.0, num_batches, cur_epoch, device, feature_position,
                         feature_importance)
        self._cons_weight = cons_weight
        self._iic_weight = iic_weight
        self._reg_criterion = reg_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        meters.register_meter("iic_weight", AverageValueMeter())
        meters.register_meter("uda_weight", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        self.meters["iic_weight"].add(self._iic_weight)
        self.meters["uda_weight"].add(self._cons_weight)
        iic_loss = IICTrainEpocher.regularization(
            self,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed
        )
        cons_loss = UDATrainEpocher.regularization(
            self,  # noqa
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed,
        )
        return self._cons_weight * cons_loss + self._iic_weight * iic_loss


class EntropyMinEpocher(TrainEpocher):
    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EntropyMinEpocher, self)._configure_meters(meters)
        meters.register_meter("entropy", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        reg_loss = Entropy()(unlabeled_logits_tf.softmax(1))
        self.meters["entropy"].add(reg_loss.item())
        return reg_loss


class MeanTeacherEpocher(TrainEpocher):

    def __init__(self, model: Union[Model, nn.Module], teacher_model: Union[Model, nn.Module], optimizer: T_optim,
                 ema_updater: EMA_Updater,
                 labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss,
                 reg_weight: float, num_batches: int, cur_epoch=0,
                 device="cpu", feature_position=None, feature_importance=None) -> None:
        assert type(model) == type(teacher_model), (type(model), type(teacher_model))
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_weight, num_batches,
                         cur_epoch, device, feature_position, feature_importance)
        self._reg_criterion = reg_criterion
        self._teacher_model = teacher_model
        self._ema_updater = ema_updater
        self._model.train()
        self._teacher_model.train()

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor,
        unlabeled_image_tf: Tensor, *args, **kwargs
    ):
        with torch.no_grad():
            teacher_unlabeled_logit = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_unlabeled_logit_tf = torch.stack(
                [self._affine_transformer(x) for x in teacher_unlabeled_logit],
                dim=0)
        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())
        # update teacher model here.
        self._ema_updater(self._teacher_model, self._model)
        return reg_loss


class IICMeanTeacherEpocher(IICTrainEpocher):
    def __init__(self, model: Union[Model, nn.Module], teacher_model: nn.Module, projectors_wrapper: ProjectorWrapper,
                 optimizer: T_optim, ema_updater: EMA_Updater,
                 labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss,
                 IIDSegCriterionWrapper: IICLossWrapper, num_batches: int, cur_epoch: int = 0,
                 device="cpu", feature_position=None, feature_importance=None, mt_weight=1,
                 iic_weight=0.1) -> None:
        super().__init__(model, projectors_wrapper, optimizer, labeled_loader, unlabeled_loader, sup_criterion,
                         IIDSegCriterionWrapper, 1.0, num_batches, cur_epoch, device, feature_position,
                         feature_importance)
        self._reg_criterion = reg_criterion
        self._teacher_model = teacher_model
        self._ema_updater = ema_updater
        self._teacher_model.train()
        self._model.train()
        self._mt_weight = float(mt_weight)
        self._iic_weight = float(iic_weight)
        assert self._reg_weight == 1.0, self._reg_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(IICMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with FeatureExtractor(self._teacher_model, self._feature_position) as self._teacher_fextractor:
            return super(IICMeanTeacherEpocher, self)._run()

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor = None,
        unlabeled_image_tf: Tensor = None,
        *args, **kwargs
    ):
        feature_names = self._fextractor._feature_names  # noqa
        unlabeled_length = len(unlabeled_tf_logits) * 2
        iic_losses_for_features = []

        with torch.no_grad():
            teacher_logits = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_logits_tf = torch.stack([self._affine_transformer(x) for x in teacher_logits], dim=0)
        assert teacher_logits_tf.shape == teacher_logits.shape, (teacher_logits_tf.shape, teacher_logits.shape)

        for i, (student_inter_feature, teacher_unlabled_features, projector, criterion) \
            in enumerate(zip(
            self._fextractor, self._teacher_fextractor,
            self._projectors_wrapper, self._IIDSegCriterionWrapper
        )):

            _student_unlabeled_features = student_inter_feature[len(student_inter_feature) - unlabeled_length:]
            student_unlabeled_features, student_unlabeled_tf_features = torch.chunk(_student_unlabeled_features, 2,
                                                                                    dim=0)
            assert teacher_unlabled_features.shape == student_unlabeled_tf_features.shape, \
                (teacher_unlabled_features.shape, student_unlabeled_tf_features.shape)

            if isinstance(projector, ClusterHead):  # features from encoder
                teacher_unlabeled_features_tf = teacher_unlabled_features
            else:
                with FixRandomSeed(seed):
                    teacher_unlabeled_features_tf = torch.stack(
                        [self._affine_transformer(x) for x in teacher_unlabled_features],
                        dim=0)
            assert teacher_unlabled_features.shape == teacher_unlabeled_features_tf.shape
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([teacher_unlabeled_features_tf, student_unlabeled_tf_features], dim=0)
                )])
            )
            _iic_loss_list = [criterion(x, y) for x, y in zip(prob1, prob2)]
            _iic_loss = average_iter(_iic_loss_list)
            iic_losses_for_features.append(_iic_loss)
        reg_loss = weighted_average_iter(iic_losses_for_features, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in iic_losses_for_features]
        )))
        uda_loss = UDATrainEpocher.regularization(
            self,
            unlabeled_tf_logits,
            teacher_logits_tf.detach(),
            seed,
        )

        # update ema
        self._ema_updater(self._teacher_model, self._model)

        return self._mt_weight * uda_loss + self._iic_weight * reg_loss
