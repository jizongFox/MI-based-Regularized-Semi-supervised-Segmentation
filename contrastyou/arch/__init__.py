from deepclustering2.arch import _register_arch
from .unet import UNet

_register_arch("ContrastUnet", UNet)
