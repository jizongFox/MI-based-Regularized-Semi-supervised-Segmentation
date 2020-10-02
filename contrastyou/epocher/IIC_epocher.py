import random

import torch
from torch import nn
from torch.nn import functional as F

from contrastyou.epocher._utils import unfold_position
from contrastyou.helper import average_iter
from deepclustering2 import optim
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.type import T_loader, T_loss
from .contrast_epocher import PretrainDecoderEpoch as _PretrainDecoderEpoch
from .contrast_epocher import PretrainEncoderEpoch as _PretrainEncoderEpoch
from ..arch import UNetFeatureExtractor
from ..losses.iic_loss import IIDLoss


class IICPretrainEcoderEpoch(_PretrainEncoderEpoch):

    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_encoder_loader: T_loader, contrastive_criterion: T_loss,
                 num_batches: int = 0, cur_epoch=0, device="cpu", group_option: str = "partition",
                 feature_extractor: UNetFeatureExtractor = None, iic_weight=1, disable_contrastive=False) -> None:
        """
        :param model:
        :param projection_head:
        :param projection_classifier: classification head
        :param optimizer:
        :param pretrain_encoder_loader: infinite dataloader with `total freedom = True`
        :param contrastive_criterion:
        :param num_batches:
        :param cur_epoch:
        :param device:
        :param iic_weight: iic weight_ratio
        """
        super().__init__(model, projection_head, optimizer, pretrain_encoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, group_option, feature_extractor)
        assert pretrain_encoder_loader is not None
        self._projection_classifier = projection_classifier
        self._iic_criterion = IIDLoss()
        self._iic_weight = iic_weight
        self._disable_contrastive = disable_contrastive

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("reg_weight", AverageValueMeter())
        meters.register_meter("iic_loss", AverageValueMeter())
        meters = super()._configure_meters(meters)
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self.meters["reg_weight"].add(self._iic_weight)

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, *features = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                en = self._feature_extractor(features)[0]
                global_enc, global_tf_enc = torch.chunk(F.normalize(self._projection_head(en), dim=1), chunks=2, dim=0)
                # projection_classifier gives a list of probabilities
                global_probs, global_tf_probs = list(
                    zip(*[torch.chunk(x, chunks=2, dim=0) for x in self._projection_classifier(en)]))
                # fixme: here lack of some code for IIC
                labels = self._label_generation(partition_list, group_list)
                contrastive_loss = self._contrastive_criterion(torch.stack([global_enc, global_tf_enc], dim=1),
                                                               labels=labels)
                iic_loss_list = [self._iic_criterion(x, y)[0] for x, y in zip(global_probs, global_tf_probs)]
                iic_loss = average_iter(iic_loss_list)
                if self._disable_contrastive:
                    total_loss = iic_loss
                else:
                    total_loss = self._iic_weight * iic_loss + contrastive_loss
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    self.meters["iic_loss"].add(iic_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict


class IICPretrainDecoderEpoch(_PretrainDecoderEpoch):

    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_decoder_loader: T_loader, contrastive_criterion: T_loss,
                 iicseg_criterion: T_loss, num_batches: int = 0, cur_epoch=0, device="cpu", disable_contrastive=False,
                 iic_weight=0.01, feature_extractor: UNetFeatureExtractor = None) -> None:
        super().__init__(model, projection_head, optimizer, pretrain_decoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, feature_extractor)
        self._projection_classifier = projection_classifier
        self._iic_criterion = iicseg_criterion
        self._disable_contrastive = disable_contrastive
        self._iic_weight = iic_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("reg_weight", AverageValueMeter())
        meters.register_meter("iic_loss", AverageValueMeter())
        meters = super()._configure_meters(meters)
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self.meters["reg_weight"].add(self._iic_weight)

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_decoder_loader):
                (img, _), (img_ctf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                seed = random.randint(0, int(1e5))
                with FixRandomSeed(seed):
                    img_gtf = torch.stack([self._transformer(x) for x in img], dim=0)
                assert img_gtf.shape == img.shape, (img_gtf.shape, img.shape)

                _, *features = self._model(torch.cat([img_gtf, img_ctf], dim=0), return_features=True)
                dn = self._feature_extractor(features)[0]
                dn_gtf, dn_ctf = torch.chunk(dn, chunks=2, dim=0)
                with FixRandomSeed(seed):
                    dn_ctf_gtf = torch.stack([self._transformer(x) for x in dn_ctf], dim=0)
                assert dn_ctf_gtf.shape == dn_ctf.shape, (dn_ctf_gtf.shape, dn_ctf.shape)
                dn_tf = torch.cat([dn_gtf, dn_ctf_gtf])
                local_enc_tf, local_enc_tf_ctf = torch.chunk(self._projection_head(dn_tf), chunks=2, dim=0)

                # IIC part
                local_probs, local_tf_probs = list(
                    zip(*[torch.chunk(x, chunks=2, dim=0) for x in self._projection_classifier(dn_tf)]))
                iic_loss_list = [self._iic_criterion(x, y) for x, y in zip(local_probs, local_tf_probs)]
                iic_loss = average_iter(iic_loss_list)
                # IIC part ends

                local_enc_unfold, _ = unfold_position(local_enc_tf, partition_num=(2, 2))
                local_tf_enc_unfold, _fold_partition = unfold_position(local_enc_tf_ctf, partition_num=(2, 2))
                b, *_ = local_enc_unfold.shape
                local_enc_unfold_norm = F.normalize(local_enc_unfold.view(b, -1), p=2, dim=1)
                local_tf_enc_unfold_norm = F.normalize(local_tf_enc_unfold.view(b, -1), p=2, dim=1)

                labels = self._label_generation(partition_list, group_list, _fold_partition)
                contrastive_loss = self._contrastive_criterion(
                    torch.stack([local_enc_unfold_norm, local_tf_enc_unfold_norm], dim=1),
                    labels=labels
                )
                if torch.isnan(contrastive_loss):
                    raise RuntimeError(contrastive_loss)
                if torch.isnan(iic_loss):
                    raise RuntimeError(iic_loss)

                if self._disable_contrastive:
                    total_loss = iic_loss
                else:
                    total_loss = self._iic_weight * iic_loss + contrastive_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    self.meters["iic_loss"].add(iic_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict


"""
class CrossIICPretrainDecoderEpoch(_PretrainDecoderEpoch):
    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_decoder_loader: T_loader, contrastive_criterion: T_loss,
                 num_batches: int = 0, cur_epoch=0, device="cpu",
                 feature_extractor: UNetFeatureExtractor = None) -> None:
        super().__init__(model, projection_head, optimizer, pretrain_decoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, feature_extractor)
        assert len(feature_extractor.feature_names) > 1, feature_extractor
        self._projection_classifier = projection_classifier
        self._iic_criterion = IIDSegmentationLoss(padding=1)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("iic_loss", AverageValueMeter())
        meters.delete_meter("contrastive_loss")
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_decoder_loader):
                (img, _), (img_ctf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                seed = random.randint(0, int(1e5))
                with FixRandomSeed(seed):
                    img_gtf = torch.stack([self._transformer(x) for x in img], dim=0)
                assert img_gtf.shape == img.shape, (img_gtf.shape, img.shape)

                _, *features = self._model(torch.cat([img_gtf, img_ctf], dim=0), return_features=True)
                dn1, dn2 = self._feature_extractor(features)
                dn1_gtf, dn1_ctf = torch.chunk(dn1, chunks=2, dim=0)
                dn2_gtf, dn2_ctf = torch.chunk(dn2, chunks=2, dim=0)

                with FixRandomSeed(seed):
                    dn1_ctf_gtf = torch.stack([self._transformer(x) for x in dn1_ctf], dim=0)
                with FixRandomSeed(seed):
                    dn2_ctf_gtf = torch.stack([self._transformer(x) for x in dn2_ctf], dim=0)

                assert dn1_ctf_gtf.shape == dn1_ctf.shape, (dn1_ctf_gtf.shape, dn1_ctf.shape)
                assert dn2_ctf_gtf.shape == dn2_ctf.shape, (dn2_ctf_gtf.shape, dn2_ctf.shape)

                dn1_tf = torch.cat([dn1_gtf, dn1_ctf_gtf])
                dn2_tf = torch.cat([dn2_gtf, dn2_ctf_gtf])

                # local_probs1
                local_enc_tf, local_enc_tf_ctf = torch.chunk(self._projection_head(dn_tf), chunks=2, dim=0)
                local_enc_probs, local_enc_tf_probs = torch.chunk(self._projection_classifier(dn_tf), chunks=2, dim=0)

                local_enc_unfold, _ = unfold_position(local_enc_tf, partition_num=(2, 2))
                local_tf_enc_unfold, _fold_partition = unfold_position(local_enc_tf_ctf, partition_num=(2, 2))
                b, *_ = local_enc_unfold.shape
                local_enc_unfold_norm = F.normalize(local_enc_unfold.view(b, -1), p=2, dim=1)
                local_tf_enc_unfold_norm = F.normalize(local_tf_enc_unfold.view(b, -1), p=2, dim=1)

                labels = self._label_generation(partition_list, group_list, _fold_partition)
                contrastive_loss = self._contrastive_criterion(
                    torch.stack([local_enc_unfold_norm, local_tf_enc_unfold_norm], dim=1),
                    labels=labels
                )
                if torch.isnan(contrastive_loss):
                    raise RuntimeError(contrastive_loss)
                self._optimizer.zero_grad()
                contrastive_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict
"""
