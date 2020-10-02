import itertools
import os

import torch
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler

from contrastyou.arch import UNetFeatureExtractor, UNet
from contrastyou.epocher.IIC_epocher import IICPretrainEcoderEpoch, IICPretrainDecoderEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from contrastyou.trainer._utils import ClusterHead, ProjectionHead, LocalProjectionHead, LocalClusterHead
from contrastyou.trainer.contrast_trainer import ContrastTrainer


class IICContrastTrainer(ContrastTrainer):

    def pretrain_encoder_init(self, group_option: str, lr=1e-6, weight_decay=1e-5, multiplier=300, warmup_max=10,

                              num_clusters=10, num_subheads=10, iic_weight=1, disable_contrastive=False, ctemperature=1,
                              ctype: str = "linear", ptype: str = "mlp", extract_position: str = "Conv5",

                              ):
        # adding optimizer and scheduler
        self._extract_position = extract_position
        self._feature_extractor = UNetFeatureExtractor(self._extract_position)
        self._projector_contrastive = ProjectionHead(
            input_dim=UNet.dimension_dict[self._extract_position],
            output_dim=256,
            head_type=ptype
        )  # noqa
        self._projector_iic = ClusterHead(
            input_dim=UNet.dimension_dict[self._extract_position],
            num_clusters=num_clusters, head_type=ctype, T=ctemperature, num_subheads=num_subheads
        )
        self._optimizer = torch.optim.Adam(
            itertools.chain(self._model.parameters(),  # noqa
                            self._projector_contrastive.parameters(),
                            self._projector_iic.parameters()),  # noqa
            lr=lr, weight_decay=weight_decay
        )  # noqa
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,  # noqa
            self._max_epoch_train_encoder - warmup_max,
            0
        )
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)  # noqa

        self._group_option = group_option  # noqa
        self._disable_contrastive = disable_contrastive

        # set augmentation method as `total_freedom = True`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
        self._pretrain_loader.dataset._transform._total_freedom = True  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa

        # contrastive loss
        self._contrastive_criterion = SupConLoss()

        # iic weight
        self._iic_weight = iic_weight

    def pretrain_encoder_run(self):
        self.to(self._device)
        self._model.disable_grad_all()
        self._model.enable_grad(from_="Conv1", util=self._extract_position)

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_encoder):
            pretrain_encoder_dict = IICPretrainEcoderEpoch(
                model=self._model, projection_head=self._projector_contrastive,
                projection_classifier=self._projector_iic, optimizer=self._optimizer,
                pretrain_encoder_loader=self._pretrain_loader_iter, contrastive_criterion=self._contrastive_criterion,
                num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
                group_option=self._group_option, feature_extractor=self._feature_extractor, iic_weight=self._iic_weight,
                disable_contrastive=self._disable_contrastive,
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_ENCODER=pretrain_encoder_dict)
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_encoder"))

    def pretrain_decoder_init(self, lr: float = 1e-6, weight_decay: float = 0.0,
                              multiplier: int = 300, warmup_max=10,

                              num_clusters=20, ctemperature=1, num_subheads=10,
                              extract_position="Up_conv3",
                              enable_grad_from="Conv1", ptype="mlp", ctype="mlp", iic_weight=1,
                              disable_contrastive=False,
                              padding=0, patch_size=512,

                              ):
        # feature_exactor
        self._extract_position = extract_position
        self._feature_extractor = UNetFeatureExtractor(self._extract_position)
        projector_input_dim = UNet.dimension_dict[extract_position]
        # if disable_encoder's gradient
        self._enable_grad_from = enable_grad_from

        # adding optimizer and scheduler
        self._projector_contrastive = LocalProjectionHead(
            projector_input_dim,
            head_type=ptype,
            output_size=(4, 4)
        )
        self._projector_iic = LocalClusterHead(
            projector_input_dim,
            num_clusters=num_clusters,
            num_subheads=num_subheads,
            head_type=ctype, T=ctemperature
        )
        self._optimizer = torch.optim.Adam(
            itertools.chain(
                self._model.parameters(),
                self._projector_contrastive.parameters(),
                self._projector_iic.parameters(),
            ),
            lr=lr,
            weight_decay=weight_decay
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_decoder - warmup_max, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)

        # set augmentation method as `total_freedom = False`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
        self._pretrain_loader.dataset._transform._total_freedom = False  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa

        # contrastive_loss
        self._contrastive_criterion = SupConLoss()
        self._disable_contrastive = disable_contrastive
        self._iicseg_criterion = IIDSegmentationSmallPathLoss(padding=padding, patch_size=patch_size)
        print(self._iicseg_criterion)

        # iic weight
        self._iic_weight = iic_weight

    def pretrain_decoder_run(self):
        self._model.disable_grad_all()
        self._model.enable_grad(from_=self._enable_grad_from, util=self._extract_position)
        self.to(self._device)

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_decoder):
            pretrain_decoder_dict = IICPretrainDecoderEpoch(
                model=self._model, projection_head=self._projector_contrastive,
                projection_classifier=self._projector_iic, optimizer=self._optimizer,
                pretrain_decoder_loader=self._pretrain_loader_iter, contrastive_criterion=self._contrastive_criterion,
                iicseg_criterion=self._iicseg_criterion, num_batches=self._num_batches, cur_epoch=self._cur_epoch,
                device=self._device, disable_contrastive=self._disable_contrastive, iic_weight=self._iic_weight,
                feature_extractor=self._feature_extractor,
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_DECODER=pretrain_decoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_decoder"))
