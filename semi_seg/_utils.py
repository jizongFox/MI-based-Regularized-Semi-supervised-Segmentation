from itertools import repeat
from typing import List, Union

from torch import nn, Tensor
from torch._six import container_abcs

from contrastyou.arch import UNet
from contrastyou.losses.iic_loss import IIDLoss as _IIDLoss, IIDSegmentationSmallPathLoss
from contrastyou.trainer._utils import LocalClusterHead as _LocalClusterHead, ClusterHead as _EncoderClusterHead


class IIDLoss(_IIDLoss):

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        return super().forward(x_out, x_tf_out)[0]


def _filter_encodernames(feature_list):
    encoder_list = UNet().encoder_names
    return list(filter(lambda x: x in encoder_list, feature_list))


def _filter_decodernames(feature_list):
    decoder_list = UNet().decoder_names
    return list(filter(lambda x: x in decoder_list, feature_list))


def _nlist(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            assert len(x) == n, (len(x), n)
            return x
        return list(repeat(x, n))

    return parse


class FeatureExtractor(nn.Module):
    class _FeatureExtractor:
        def __call__(self, _, input, result):
            self.feature = result

    def __init__(self, net: UNet, feature_names: Union[List[str], str]) -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in UNet().component_names, f

    def __enter__(self):
        self._feature_exactors = {}
        self._hook_handlers = {}
        for f in self._feature_names:
            extractor = self._FeatureExtractor()
            handler = getattr(self._net, f).register_forward_hook(extractor)
            self._feature_exactors[f] = extractor
            self._hook_handlers[f] = handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._hook_handlers.items():
            v.remove()
        del self._feature_exactors, self._hook_handlers

    def __getitem__(self, item):
        if item in self._feature_exactors:
            return self._feature_exactors[item].feature
        return super().__getitem__(item)

    def get_feature_from_num(self, num):
        feature = self._feature_names[num]
        return self[feature]

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature


class LocalClusterWrappaer(nn.Module):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False
    ) -> None:
        super(LocalClusterWrappaer, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(self))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)

        self._clusters = nn.ModuleDict()

        for f, h, c, s, n in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                 self._normalize):
            self._clusters[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                num_clusters=c,
                num_subheads=s,
                normalize=n
            )

    def __len__(self):
        return len(self._feature_names)

    def __iter__(self):
        for k, v in self._clusters.items():
            yield v

    def __getitem__(self, item):
        if item in self._clusters.keys():
            return self._clusters[item]
        return super().__getitem__(item)

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _LocalClusterHead(*args, **kwargs)


class EncoderClusterWrapper(LocalClusterWrappaer):
    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _EncoderClusterHead(*args, **kwargs)


class ProjectorWrapper(nn.Module):
    ENCODER_INITIALIZED = False
    DECODER_INITIALIZED = False

    def init_encoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False
    ):
        self._encoder_names = _filter_encodernames(feature_names)
        self._encoder_projectors = EncoderClusterWrapper(
            self._encoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self.ENCODER_INITIALIZED = True

    def init_decoder(self,
                     feature_names: Union[str, List[str]],
                     head_types: Union[str, List[str]] = "linear",
                     num_subheads: Union[int, List[int]] = 5,
                     num_clusters: Union[int, List[int]] = 10,
                     normalize: Union[bool, List[bool]] = False
                     ):
        self._decoder_names = _filter_decodernames(feature_names)
        self._decoder_projectors = LocalClusterWrappaer(
            self._decoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self.DECODER_INITIALIZED = True

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names

    def __getitem__(self, item):
        if self.ENCODER_INITIALIZED and item in self._encoder_projectors._feature_names:
            return self._encoder_projectors[item]
        elif self.DECODER_INITIALIZED and item in self._decoder_projectors._feature_names:
            return self._decoder_projectors[item]
        raise IndexError(item)

    def __iter__(self):
        if (self.ENCODER_INITIALIZED and self.DECODER_INITIALIZED) is not True:
            raise RuntimeError(f"Encoder_projectors or Decoder_projectors are not initialized "
                               f"in {self.__class__.__name__}.")
        if self.ENCODER_INITIALIZED:
            yield from self._encoder_projectors
        if self.DECODER_INITIALIZED:
            yield from self._decoder_projectors


class IICLossWrapper(nn.Module):

    def __init__(self,
                 feature_names: Union[str, List[str]],
                 paddings: Union[int, List[int]],
                 patch_sizes: Union[int, List[int]]) -> None:
        super().__init__()
        self._encoder_features = _filter_encodernames(feature_names)
        self._decoder_features = _filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)
        self._LossModuleDict = nn.ModuleDict()

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = IIDLoss()
        if len(self._decoder_features) > 0:
            paddings = _nlist(len(self._decoder_features))(paddings)
            patch_sizes = _nlist(len(self._decoder_features))(patch_sizes)
            for f, p, size in zip(self._decoder_features, paddings, patch_sizes):
                self._LossModuleDict[f] = IIDSegmentationSmallPathLoss(padding=p, patch_size=size)

    def __getitem__(self, item):
        if item in self._LossModuleDict.keys():
            return self._LossModuleDict[item]
        raise IndexError(item)

    def __iter__(self):
        for k, v in self._LossModuleDict.items():
            yield v

    def items(self):
        return self._LossModuleDict.items()

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features
