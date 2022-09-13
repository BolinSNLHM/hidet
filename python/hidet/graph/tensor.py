from __future__ import annotations

import ctypes
from functools import partial
from typing import List, Optional, Tuple, Sequence, Union

import numpy as np

import hidet.runtime.storage
from hidet.ffi import cuda, cuda_kernels
from hidet.ir.type import ScalarType
from hidet.ir.layout import DataLayout, RowMajorLayout
from hidet.runtime.storage import Storage
from hidet.utils import prod


def convert(v):
    if isinstance(v, (float, int)):
        dtype_map = {
            float: 'float32',
            int: 'int64'
        }
        return full(shape=[1], fill_value=v, dtype=dtype_map[type(v)])
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    """An n-dimension array, could be symbolic or concrete.

    This class defines an n-dimension array.

    Attributes
    ----------
    shape: List[int]
        The shape of the tensor.

    dtype: str
        The data type of the tensor.

    device: str
        The device of the tensor.

    layout: DataLayout
        The data layout of the tensor.

    storage: Optional[Storage]
        The storage of the tensor. None indicates it is a symbolic tensor.

    trace: Optional[Tuple[Operator, int]]
        Where this tensor is derived from. A trace = (op, i) indicates that this tensor is the i-th output of the op operator.
    """

    def __init__(self,
                 shape: Sequence[int],
                 dtype: str,
                 device: str,
                 storage: Optional[Storage],
                 layout: DataLayout = None,
                 trace: Optional[Tuple['Operator', int]] = None):
        from hidet.graph.operator import Operator
        self.shape: List[int] = [int(v) for v in shape]
        self.dtype: str = str(dtype)
        self.device: str = device
        self.storage: Optional[Storage] = storage
        self.layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self.trace: Optional[Tuple[Operator, int]] = trace

    def __neg__(self) -> Tensor:
        from .ops import neg
        return neg(self)

    def __add__(self, other) -> Tensor:
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        from .ops import add
        return add(other, self)

    def __sub__(self, other) -> Tensor:
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other) -> Tensor:
        from .ops import multiply
        return multiply(self, other)

    def __rmul__(self, other):
        from .ops import multiply
        return multiply(other, self)

    def __truediv__(self, other) -> Tensor:
        from .ops import divide
        return divide(self, other)

    def __str__(self):
        head = self.signature()
        if self.storage:
            array_str = str(self.cpu().numpy())
            return '{}\n{}'.format(head, array_str)
        else:
            if self.trace is None:
                return head
            else:
                return '{}\nfrom {}'.format(head, self.trace)

    def __getitem__(self, item):
        from hidet.graph.ops import strided_slice
        if not isinstance(item, tuple):
            item = tuple([item])
        rank = len(self.shape)
        if all(not isinstance(v, slice) for v in item) and len(item) == rank:
            # element access
            return strided_slice(self, starts=list(item), ends=[v + 1 for v in item]).numpy().flatten()[0]
        else:
            while len(item) < rank:
                item = item + (slice(None, None, None),)
            starts, ends, steps = [], [], []
            squeeze_dims = []
            for dim, v in enumerate(item):
                if isinstance(v, int):
                    squeeze_dims.append(dim)
                    starts.append(v)
                    ends.append(v + 1)
                    steps.append(1)
                else:
                    assert isinstance(v, slice)
                    starts.append(v.start if v.start is not None else 0)
                    ends.append(v.stop if v.stop is not None else self.shape[dim])
                    steps.append(v.step if v.step is not None else 1)
            sliced = strided_slice(self, starts, ends, strides=steps).squeeze(squeeze_dims)
            return sliced

    def __iter__(self):
        raise TypeError('hidet.Tensor does not support iteration.')

    def __getstate__(self):
        if self.storage:
            data = self.detach().numpy()
        else:
            data = None

        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'data': data,
            'layout': self.layout,
            'trace': self.trace
        }

    def __setstate__(self, state):
        data = state['data']
        if data is not None:
            assert isinstance(data, np.ndarray)
            tensor = from_numpy(data)
            if state['device'] == 'cuda':
                tensor = tensor.cuda()
            storage = tensor.storage
        else:
            storage = None

        self.shape = state['shape']
        self.dtype = state['dtype']
        self.device = state['device']
        self.storage = storage
        self.layout = state['layout']
        self.trace = state['trace']

    def signature(self) -> str:
        return "Tensor(shape={}, dtype='{}', device='{}')".format(self.shape, self.dtype, self.device)

    @property
    def nbytes(self):
        return prod(self.shape) * ScalarType(self.dtype).nbytes()

    @property
    def num_elements(self):
        return prod(self.shape)

    @property
    def op(self):
        return self.trace[0] if self.trace else None

    def scalar(self) -> Union[float, int]:
        if len(self.shape) != 0:
            raise ValueError('Can not convert a Tensor with shape {} to a scalar.'.format(self.shape))
        value = self.numpy().tolist()
        assert isinstance(value, (int, float))
        return value

    def contiguous(self):
        if isinstance(self.layout, RowMajorLayout):
            return self
        return self.reshape(self.shape)

    def reshape(self, shape: Sequence[int]):
        from .ops import reshape
        return reshape(self, shape)

    def squeeze(self, dims: Union[int, Sequence[int]]):
        from .ops import squeeze
        return squeeze(self, dims)

    def unsqueeze(self, dims: Union[int, Sequence[int]]):
        from .ops import unsqueeze
        return unsqueeze(self, dims)

    def rearrange(self, plan: List[List[int]]):
        from .ops import rearrange
        return rearrange(self, plan)

    def flatten(self, start_dim=0, end_dim=None):
        from .ops import flatten
        return flatten(self, start_dim, end_dim)

    def transpose(self, axes: Optional[Sequence[int]]):
        from .ops import transpose
        return transpose(self, axes)

    def barrier(self) -> Tensor:
        from .ops import barrier
        return barrier(self)

    def sum(self, dims: Union[int, List[int]], keep_dim: bool = False):
        from .ops import reduce_sum
        return reduce_sum(self, dims=dims, keep_dim=keep_dim)

    def mean(self, dims: Union[int, List[int]], keep_dim: bool = False):
        from .ops import reduce_mean
        return reduce_mean(self, dims=dims, keep_dim=keep_dim)

    def rsqrt(self):
        from .ops import rsqrt
        return rsqrt(self)

    def cast(self, dtype):
        from .ops import cast
        return cast(self, dtype)

    def to(self, dtype: str):
        from .ops import cast
        return cast(self, dtype)

    def cpu(self):
        if self.device == 'cpu':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda(self):
        if self.device == 'cuda':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cuda', self.storage.cuda() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def copy(self) -> Tensor:
        if self.trace is not None:
            raise ValueError('Please use .detach() to detach a trace variable first before copying.')
        return Tensor(
            shape=list(self.shape),
            dtype=self.dtype,
            device=self.device,
            storage=self.storage.copy(),
            layout=self.layout,
            trace=None
        )

    def detach(self):
        if self.trace is None:
            return self
        else:
            return Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                storage=self.storage,
                layout=self.layout,
                trace=None
            )

    def numpy(self) -> np.ndarray:
        if self.device != 'cpu':
            return self.cpu().numpy()
        # convert if this tensor is not in row major layout
        storage = self.contiguous().storage

        # because numpy does not support bfloat16 and tfloat32, we convert them into float32
        if self.dtype in ['bfloat16', 'tfloat32']:
            return self.cast('float32').numpy()
        else:
            array = storage.as_array(num_elements=prod(self.shape), dtype=self.dtype)
            return array.reshape(self.shape)

    def torch(self):
        import torch
        torch_tensor = torch.empty(
            size=self.shape,
            dtype=getattr(torch, self.dtype),
            device=self.device
        )
        if self.storage is None:
            # convert a symbolic tensor to a dummy torch tensor
            return torch_tensor
        cuda.memcpy_async(
            src_addr=self.storage.addr,
            dst_addr=torch_tensor.data_ptr(),
            num_bytes=self.nbytes,
            kind=cuda.DeviceToDevice
        )
        return torch_tensor


def empty(shape, dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None):
    """Create an uninitialized tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    num_bytes = prod(shape) * ScalarType(dtype).nbytes()
    storage = Storage.new(device, num_bytes)
    return Tensor(shape, dtype, device, storage, layout)


def symbol(shape: Sequence[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    """Create a symbolic tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.

    """
    return Tensor(shape, dtype, device, None, layout)


def zeros(shape: Sequence[int], dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with zero.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    tensor = empty(shape, dtype, device, layout)
    cuda.memset_async(tensor.storage.addr, tensor.nbytes, value=0)
    return tensor


def ones(shape, dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with one.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    value_map = {
        'float32': 1.0,
        'int32': 1,
        'int64': 1
    }
    if dtype in value_map:
        return full(shape, value_map[dtype], dtype, device, layout)
    else:
        if dtype in ['float16', 'bool']:
            f32_tensor = ones(shape, 'float32', device, layout)
            return f32_tensor.cast(dtype)
        else:
            raise NotImplementedError('Not implemented ones for dtype {}, please create a float32 tensor and cast to this type'.format(dtype))


def full(shape, fill_value, dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with given constant.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    fill_value: Union[float, int]
        The constant to initialize the new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    tensor = empty(shape, dtype, device, layout)
    cuda_kernels.fill_value(tensor.storage.addr, num_elements=tensor.num_elements, value=fill_value, dtype=dtype)
    return tensor


def randn(shape, dtype='float32', mean=0.0, stddev=1.0, device='cuda', layout=None) -> Tensor:
    """Create a tensor with uniformly distributed values.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    mean: float
        The mean of the uniform distribution.

    stddev: float
        The standard deviation of the uniform distribution.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.

    Examples
    --------
    >>> randn([2, 3])
    Tensor(shape=[2, 3], dtype='float32', device='cuda')
    [[ 0.10720467 -1.6906018   0.06347568]
     [-0.37061226  0.562728    1.857547  ]]
    """
    tensor = empty(shape, dtype, device, layout)
    if dtype == 'float32':
        cuda.generate_normal(tensor.storage.addr, num_elements=prod(tensor.shape), mean=mean, stddev=stddev)
    else:
        float32_tensor = randn_like(tensor, dtype='float32')  # todo: add mean and stddev
        return float32_tensor.cast(dtype=dtype)
    return tensor


def randint(low: int, high=None, shape: Sequence[int] = (), dtype: str = 'int32') -> Tensor:
    dtype_map = {
        'int32': np.int32,
        'int64': np.int64
    }
    if dtype not in dtype_map:
        return randint(low=low, high=high, shape=shape, dtype='int32').cast(dtype)
    return array(np.random.randint(low=low, high=high, size=shape, dtype=dtype_map[dtype]))


def _tensor_like(constructor, data, shape, dtype, device, layout):
    shape = data.shape if shape is None else shape
    dtype = data.dtype if dtype is None else dtype
    device = data.device if device is None else device
    layout = data.layout if layout is None else layout
    return constructor(shape=shape, dtype=dtype, device=device, layout=layout)


def empty_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(empty, data, shape, dtype, device, layout)


def symbol_like(data: Tensor, shape=None, dtype=None, device=None, layout=None):
    """ Create a symbol tensor like an existing tensor.

    Parameters
    ----------
    data: Tensor
        The information of this tensor will be used to create the symbol tensor.

    shape: Optional[Sequence[int]]
        The shape of the new tensor.

    dtype: Optional[str]
        The data type of the new tensor.

    device: Optional[str]
        The device of the new tensor.

    layout: Optional[DataLayout]
        The data layout of the new tensor.

    Returns
    -------
    ret: Tensor
        The created symbol tensor.
    """
    return _tensor_like(symbol, data, shape, dtype, device, layout)


def zeros_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(zeros, data, shape, dtype, device, layout)


def ones_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(ones, data, shape, dtype, device, layout)


def full_like(data: Tensor, fill_value, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(partial(full, fill_value=fill_value), data, shape, dtype, device, layout)


def randn_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(randn, data, shape, dtype, device, layout)


def randint_like(data: Tensor, low: int, high: Optional[int] = None, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None):
    return _tensor_like(partial(randint, low=low, high=high), data, shape, dtype, device, layout)


def void_pointer_to_uint64(p):
    ret = ctypes.cast(ctypes.addressof(p), ctypes.POINTER(ctypes.c_uint64)).contents
    return ret.value


def from_numpy(nparray: np.ndarray) -> Tensor:
    dtype_convert = {
        np.dtype(np.float64): 'float64',
        np.dtype(np.float32): 'float32',
        np.dtype(np.int64): 'int64',
        np.dtype(np.int32): 'int32',
        np.dtype(np.float16): 'float16',
        np.dtype(np.bool): 'bool',
        np.dtype(np.uint8): 'uint8'
    }
    if nparray.dtype not in dtype_convert:
        raise NotImplementedError("Do not support convert np.ndarray with data type '{}'.".format(nparray.dtype))
    nparray = nparray.copy(order='C')  # make the data layout like C, which is contiguous
    tensor = empty(shape=nparray.shape, dtype=dtype_convert[nparray.dtype], device='cpu')
    cuda.memcpy_async(src_addr=void_pointer_to_uint64(nparray.ctypes.data_as(ctypes.c_void_p)),
                      dst_addr=tensor.storage.addr,
                      num_bytes=tensor.nbytes,
                      kind=cuda.HostToHost)
    cuda.device_synchronize()
    return tensor


def from_torch(torch_tensor):
    """ Create a hidet tensor from pytorch tensor.

    The created tensor shared the same memory as given pytorch tensor. Thus, any content
    modification on one tensor would be reflected on the other one.

    Parameters
    ----------
    torch_tensor: torch.Tensor
        The pytorch tensor.

    Returns
    -------
    ret: Tensor
        The created hidet tensor.
    """
    import torch
    if not isinstance(torch_tensor, torch.Tensor):
        raise ValueError('Expect a torch.Tensor, got {}'.format(type(torch_tensor)))
    if not torch_tensor.is_cuda:
        raise ValueError('Expect the torch tensor is on cuda device.')
    dtype_convert = {
        torch.float32: 'float32',
        torch.float16: 'float16',
        torch.int32: 'int32',
        torch.int64: 'int64'
    }
    return Tensor(
        shape=[int(v) for v in torch_tensor.size()],
        dtype=dtype_convert[torch_tensor.dtype],
        device='cuda',
        storage=hidet.runtime.storage.TorchStorage(torch_tensor),
    )


def array(obj: Union[List, Tuple, np.ndarray, Tensor]) -> Tensor:
    if isinstance(obj, np.ndarray):
        return from_numpy(obj)
    elif isinstance(obj, Tensor):
        return obj
    else:
        return from_numpy(np.array(obj))
