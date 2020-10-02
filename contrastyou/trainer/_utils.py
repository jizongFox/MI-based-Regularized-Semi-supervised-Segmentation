from torch import nn, Tensor
from torch.nn import functional as F


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, *_ = features.shape
        return features.view(b, -1)


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 0.1) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)


class Normalize(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return nn.functional.normalize(input, p=2, dim=1)


class Identical(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input


class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp") -> None:
        super().__init__()
        assert head_type in ("mlp", "linear")
        if head_type == "mlp":
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
            )
        else:
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, features):
        return self._header(features)


class LocalProjectionHead(nn.Module):
    """
    return a fixed feature size
    """

    def __init__(self, input_dim, head_type="mlp", output_size=(4, 4)) -> None:
        super().__init__()
        assert head_type in ("mlp", "linear"), head_type
        self._output_size = output_size
        if head_type == "mlp":
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 3, 1, 1),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
            )
        else:
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 3, 1, 1),
            )

    def forward(self, features):
        b, c, h, w = features.shape
        out = self._projector(features)
        # fixme: Upsampling and interpolate don't pass the gradient correctly.
        return F.adaptive_max_pool2d(out, output_size=self._output_size)
        # return out


class ClusterHead(nn.Module):
    def __init__(self, input_dim, num_clusters=5, num_subheads=10, head_type="linear", T=1, normalize=False) -> None:
        super().__init__()
        assert head_type in ("linear", "mlp"), head_type
        self._input_dim = input_dim
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, self._num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, 128),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Linear(128, num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [
            init_sub_header(head_type)
            for _ in range(self._num_subheads)
        ]

        self._headers = nn.ModuleList(headers)

    def forward(self, features):
        return [x(features) for x in self._headers]


class LocalClusterHead(nn.Module):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, input_dim, head_type="linear", num_clusters=10, num_subheads=10, T=1, interm_dim=64, normalize=False) -> None:
        super().__init__()
        assert head_type in ("linear", "mlp"), head_type
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
                return nn.Sequential(
                    nn.Conv2d(input_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_dim, interm_dim, 1, 1, 0),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv2d(interm_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [init_sub_header(head_type) for _ in range(num_subheads)]
        self._headers = nn.ModuleList(headers)

    def forward(self, features):
        return [x(features) for x in self._headers]
