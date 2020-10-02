from collections import OrderedDict
from typing import Union, List

import torch
from torch import nn

__all__ = ["UNet"]


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    dimension_dict = {
        "Conv1": 16,
        "Conv2": 32,
        "Conv3": 64,
        "Conv4": 128,
        "Conv5": 256,
        "Up_conv5": 128,
        "Up_conv4": 64,
        "Up_conv3": 32,
        "Up_conv2": 16
    }

    def __init__(self, input_dim=3, num_classes=1):
        super(UNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=input_dim, out_ch=16)
        self.Conv2 = conv_block(in_ch=16, out_ch=32)
        self.Conv3 = conv_block(in_ch=32, out_ch=64)
        self.Conv4 = conv_block(in_ch=64, out_ch=128)
        self.Conv5 = conv_block(in_ch=128, out_ch=256)

        self.Up5 = up_conv(in_ch=256, out_ch=128)
        self.Up_conv5 = conv_block(in_ch=256, out_ch=128)

        self.Up4 = up_conv(in_ch=128, out_ch=64)
        self.Up_conv4 = conv_block(in_ch=128, out_ch=64)

        self.Up3 = up_conv(in_ch=64, out_ch=32)
        self.Up_conv3 = conv_block(in_ch=64, out_ch=32)

        self.Up2 = up_conv(in_ch=32, out_ch=16)
        self.Up_conv2 = conv_block(in_ch=32, out_ch=16)

        self.DeConv_1x1 = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_features=False):
        # encoding path
        e1 = self.Conv1(x)  # 16 224 224
        # e1-> Conv1

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)  # 32 112 112
        # e2 -> Conv2

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)  # 64 56 56
        # e3->Conv3

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)  # 128 28 28
        # e4->Conv4

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # 256 14 14
        # e5->Conv5

        # decoding + concat path
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)  # 128 28 28
        # d5->Up5+Up_conv5

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)  # 64 56 56
        # d4->Up4+Up_conv4

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # 32 112 112
        # d3->Up3+upconv3

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # 16 224 224
        # d2->up2+upconv2

        d1 = self.DeConv_1x1(d2)  # 4 224 224
        # d1->Decov1x1
        if return_features:
            return d1, (e5, e4, e3, e2, e1), (d5, d4, d3, d2)
        return d1

    def disable_grad_encoder(self):
        self._set_grad(self.encoder_names, False)

    def enable_grad_encoder(self):
        self._set_grad(self.encoder_names, True)

    def disable_grad_decoder(self):
        self._set_grad(self.decoder_names, False)

    def enable_grad_decoder(self):
        self._set_grad(self.decoder_names, True)

    def enable_grad_util(self, name: str):
        assert name in self.component_names, name
        index = self.component_names.index(name)
        self._set_grad(self.component_names[:index + 1], True)

    def disable_grad_util(self, name):
        assert name in self.component_names, name
        index = self.component_names.index(name)
        self._set_grad(self.component_names[:index + 1], False)

    def enable_grad_all(self):
        self._set_grad(self.component_names, True)

    def disable_grad_all(self):
        self._set_grad(self.component_names, False)

    def enable_grad(self, from_: str, util: str):
        assert from_ in self.component_names, from_
        assert util in self.component_names, util
        from_index = self.component_names.index(from_)
        util_index = self.component_names.index(util)
        assert from_index <= util_index, (from_, util)
        self._set_grad(self.component_names[from_index:util_index + 1], True)

    def disable_grad(self, from_: str, util: str):
        assert from_ in self.component_names, from_
        assert util in self.component_names, util
        from_index = self.component_names.index(from_)
        util_index = self.component_names.index(util)
        assert from_index <= util_index, (from_, util)
        self._set_grad(self.component_names[from_index:util_index + 1], False)

    def _set_grad(self, name_list, requires_grad=False):
        for n in name_list:
            for parameter in getattr(self, n).parameters():
                parameter.requires_grad = requires_grad

    @property
    def encoder_names(self):
        return ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]

    @property
    def decoder_names(self):
        return ["Up5", "Up_conv5", "Up4", "Up_conv4", "Up3", "Up_conv3", "Up2", "Up_conv2", "DeConv_1x1"]

    @property
    def component_names(self):
        return self.encoder_names + self.decoder_names

    def weight_norm(self):
        weights = OrderedDict()
        for name, p in self.named_parameters():
            weights[name] = p.norm().item()
        return weights


class FeatureExtractor:
    encoder_names = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
    decoder_names = ["Up_conv5", "Up_conv4", "Up_conv3", "Up_conv2"]
    names = encoder_names + decoder_names

    def __init__(self, feature_names: Union[List[str], str]) -> None:
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        for f in feature_names:
            assert f in self.names, f
        self._feature_names: List[str] = feature_names

    @property
    def feature_names(self):
        return self._feature_names

    def __call__(self, features) -> List[torch.Tensor]:
        (e5, e4, e3, e2, e1), (d5, d4, d3, d2) = features
        return_list = []

        for f in self._feature_names:
            if f in self.encoder_names:
                index = self.encoder_names.index(f)
                return_list.append(locals()[f"e{index + 1}"])
            else:
                index = self.decoder_names.index(f)
                return_list.append(locals()[f"d{5 - index}"])
        return return_list

    def __repr__(self):
        def list2string(a_list):
            if len(a_list) == 1:
                return str(a_list[0])
            else:
                return ", ".join(a_list)

        return f"{self.__class__.__name__} with features to be extracted at {list2string(self._feature_names)}."
