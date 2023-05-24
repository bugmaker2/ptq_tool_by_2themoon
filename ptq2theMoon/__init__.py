import torch
from torchvision import transforms, datasets
import ppq.lib as PFL
from ppq import *
from ppq.core import QuantizationPolicy, QuantizationProperty, RoundingPolicy
from ppq.api import *
from ppq.quantization.optim import *
from typing import Union
import re
import timm
from ppq.IR import *
from .Utilities.Imagenet import (evaluate_mmlab_module_with_imagenet,
                                 evaluate_onnx_module_with_imagenet,
                                 evaluate_ppq_module_with_imagenet,
                                 evaluate_torch_module_with_imagenet,
                                 load_imagenet_from_directory)
from ppq.lib import register_network_quantizer