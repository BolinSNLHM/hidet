# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=redefined-builtin
from .create import full
from .arithmetic import add, subtract, multiply, divide, negative, sqrt, rsqrt, where, maximum, minimum, reciprocal
from .arithmetic import bitwise_and, bitwise_invert, bitwise_or, bitwise_xor, bitwise_right_shift, bitwise_left_shift
from .arithmetic import ceil, exp, log, abs
from .compare import equal, less, greater, less_equal, greater_equal, logical_not, logical_and
from .transform import squeeze, unsqueeze, flatten, concat, cast, take, rearrange, strided_slice, split, pad, conv_pad
from .pool import avg_pool2d, adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d
from .pool import max_pool2d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
from .activation import relu, leaky_relu, sigmoid, hardsigmoid, clip, relu6, prelu, gelu, silu, hardswish
from .activation import logsigmoid, celu, hardshrink, softplus, softsign, tanhshrink, softshrink
from .activation import softmax, softmin, hardtanh
from .image import resize2d
from .cumulative import cumsum
from .special import barrier
from .conv1d import conv1d
from .conv1d_transpose import conv1d_transpose
from .attention import attention
from .conv2d import conv2d
from .conv2d_transpose import conv2d_transpose
from .conv3d import conv3d
from .conv3d_transpose import conv3d_transpose
from .matmul import batch_matmul, matmul

from .matmul import BatchMatmulOp, MatmulOp, Matmulx86Op
from .conv2d import Conv2dOp
from .arithmetic import ErfOp, PowOp, AddOp, SubtractOp, MultiplyOp, DivideOp, WhereOp
from .compare import EqualOp
from .reduce import ReduceSumOp, ReduceMeanOp
from .reduce import mean, min, max, sum, var, argmin, argmax
from .transform import PadOp, ConcatOp

from . import utils
from . import arithmetic_resolve
