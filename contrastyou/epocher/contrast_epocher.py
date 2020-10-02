import random
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from deepclustering2 import optim
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher, proxy_trainer  # noqa
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.type import T_loader, T_loss
from ._utils import preprocess_input_with_twice_transformation, unfold_position, GlobalLabelGenerator, \
    LocalLabelGenerator
from ..arch import UNetFeatureExtractor


class PretrainEncoderEpoch(_Epocher):
    """using a pretrained network to train with a data loader with contrastive loss."""

    def __init__(self, model: nn.Module, projection_head: nn.Module, optimizer: optim.Optimizer,
                 pretrain_encoder_loader: T_loader, contrastive_criterion: T_loss, num_batches: int = 0,
                 cur_epoch=0, device="cpu", group_option: str = None,
                 feature_extractor: UNetFeatureExtractor = None) -> None:
        """
        PretrainEncoder Epocher
        :param model: nn.Module for a model
        :param projection_head: shallow projection head
        :param optimizer: optimizer for both network and shallow projection head.
        :param pretrain_encoder_loader: dataloader for epocher
        :param contrastive_criterion: contrastive loss, can be any loss given the normalized norm.
        :param num_batches: num_batches to be used
        :param cur_epoch: current epoch
        :param device: device for images
        :param group_option: group option for contrastive loss
        :param feature_extractor: feature extractor defined in trainer
        """
        super().__init__(model, cur_epoch, device)
        self._projection_head = projection_head
        self._optimizer = optimizer
        self._pretrain_encoder_loader = pretrain_encoder_loader
        self._contrastive_criterion = contrastive_criterion
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        self._num_batches = num_batches
        assert isinstance(group_option, str) and group_option in ("partition", "patient", "both"), group_option
        self._group_option = group_option
        self._init_label_generator(self._group_option)
        assert isinstance(feature_extractor, UNetFeatureExtractor), feature_extractor
        self._feature_extractor = feature_extractor

    def _init_label_generator(self, group_option):
        contrastive_on_partition = False
        contrastive_on_patient = False

        if group_option == "partition":
            contrastive_on_partition = True
        if group_option == "patient":
            contrastive_on_patient = True
        if group_option == "both":
            contrastive_on_patient = True
            contrastive_on_partition = True
        self._global_contrastive_label_generator = GlobalLabelGenerator(
            contrastive_on_partition=contrastive_on_partition,
            contrastive_on_patient=contrastive_on_patient
        )

    @classmethod
    def create_from_trainer(cls, trainer):
        # todo: complete the code here
        pass

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("contrastive_loss", AverageValueMeter())
        meters.register_meter("lr", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, *features = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                en = self._feature_extractor(features)[0]
                global_enc, global_tf_enc = torch.chunk(F.normalize(self._projection_head(en), dim=1), chunks=2, dim=0)
                labels = self._label_generation(partition_list, group_list)
                contrastive_loss = self._contrastive_criterion(
                    torch.stack([global_enc, global_tf_enc], dim=1),
                    labels=labels
                )
                self._optimizer.zero_grad()
                contrastive_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return preprocess_input_with_twice_transformation(data, device)

    def _label_generation(self, partition_list: List[str], group_list: List[str]):
        """override this to provide more mask """
        return self._global_contrastive_label_generator(partition_list=partition_list,
                                                        patient_list=group_list)


class PretrainDecoderEpoch(PretrainEncoderEpoch):
    """using a pretrained network to train with a dataloader, for decoder part"""

    def __init__(self, model: nn.Module, projection_head: nn.Module, optimizer: optim.Optimizer,
                 pretrain_decoder_loader: T_loader, contrastive_criterion: T_loss, num_batches: int = 0, cur_epoch=0,
                 device="cpu", feature_extractor: UNetFeatureExtractor = None) -> None:
        super().__init__(model, projection_head, optimizer, pretrain_decoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, "both", feature_extractor)
        self._pretrain_decoder_loader = self._pretrain_encoder_loader
        self._transformer = TensorRandomFlip(axis=[1, 2], threshold=0.5)

    def _init_label_generator(self, group_option):
        self._local_contrastive_label_generator = LocalLabelGenerator()

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
                dn = self._feature_extractor(features)[0]
                dn_gtf, dn_ctf = torch.chunk(dn, chunks=2, dim=0)
                with FixRandomSeed(seed):
                    dn_ctf_gtf = torch.stack([self._transformer(x) for x in dn_ctf], dim=0)
                assert dn_ctf_gtf.shape == dn_ctf.shape, (dn_ctf_gtf.shape, dn_ctf.shape)
                dn_tf = torch.cat([dn_gtf, dn_ctf_gtf])
                local_enc_tf, local_enc_tf_ctf = torch.chunk(self._projection_head(dn_tf), chunks=2, dim=0)
                # todo: convert representation to distance
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

    def _label_generation(self, partition_list: List[str], patient_list: List[str], location_list: List[str]):  # noqa
        return self._local_contrastive_label_generator(partition_list=partition_list, patient_list=patient_list,
                                                       location_list=location_list)



"""
class SemiContrastEpoch(_SimpleFineTuneEpoch):

    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, tra_loader: T_loader,
                 num_batches: int = 100,
                 sup_criterion: T_loss = None, cur_epoch=0, device="cpu", group_option="partition") -> None:
        super().__init__(model, optimizer, labeled_loader, num_batches, sup_criterion, cur_epoch, device)
        self._reg_criterion = SupConLoss()
        self._tra_loader = tra_loader
        self._group_option = group_option
        self._init_label_generator(self._group_option)

    def _init_label_generator(self, group_option):
        contrastive_on_partition = False
        contrastive_on_patient = False

        if group_option == "partition":
            contrastive_on_partition = True
        if group_option == "patient":
            contrastive_on_patient = True
        if group_option == "both":
            contrastive_on_patient = True
            contrastive_on_partition = True
        self._global_contrastive_label_generator = GlobalLabelGenerator(
            contrastive_on_partition=contrastive_on_partition,
            contrastive_on_patient=contrastive_on_patient
        )

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        report_dict: EpochResultDict
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, label_data, unlab_data in zip(indicator, self._labeled_loader, self._tra_loader):
                (labelimage, labeltarget), _, filename, partition_list, group_list \
                    = self._preprocess_data(label_data, self._device)
                (img,_), (img_tf,_)

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
"""
