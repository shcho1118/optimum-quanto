import ast
from copy import copy

import numpy as np
import torch
from torch.utils import _pytree as pytree


__all__ = ["MarlinPackedTensor"]


# From: https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py#L40
def _get_perm():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


_perm = _get_perm()


# From: https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py#L102
def pack(unpacked: torch.Tensor):
    w = unpacked
    N, K = w.shape
    w = unpacked.t()
    w = w.reshape((K // 16, 16, N // 16, 16))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((K // 16, N * 16))
    res = w
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    p = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        p |= res[:, i::8] << 4 * i
    p = torch.from_numpy(p.astype(np.int32)).to(w.device)
    return p


class MarlinPackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, size, stride, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.dtype == torch.int32
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.uint8, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, size, stride, requires_grad=False):
        self._data = data

    def __repr__(self):
        return f"MarlinPackedTensor({self._data})"

    @classmethod
    def pack(cls, t):
        data = pack(t)
        return MarlinPackedTensor(data, t.size(), t.stride())

    def unpack(self):
        # FIXME: should implement unpack function
        raise NotImplementedError("unpack() is not implemented yet.")

    @property
    def dtype(self):
        return torch.uint8

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        # Since meta can be used for serialization, use only AST compatible strings
        meta = {
            "size": str(list(self.size())),
            "stride": str(self.stride()),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        assert len(meta) == 2
        data = inner_tensors["_data"]
        # Meta should contain only AST compatible strings
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return MarlinPackedTensor(data, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Convert back to tensor before calling any operation except detach and move
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return MarlinPackedTensor(data, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.uint8)
            if dtype != torch.uint8:
                raise ValueError(f"MarlinPackedTensor are torch.uint8 only and cannot be moved to {dtype}.")
            device = kwargs.get("device", t.device)
            # MarlinPackedTensor can only be moved to CUDA devices
            if device.type == "cuda":
                data_kwargs = copy(kwargs)
                data_kwargs["dtype"] = t._data.dtype
                data = op(t._data, **data_kwargs)
                return MarlinPackedTensor(data, t.size(), t.stride())
        args, kwargs = pytree.tree_map_only(MarlinPackedTensor, lambda x: x.unpack(), (args, kwargs or {}))
        return op(*args, **kwargs)

    def numpy(self):
        return self.unpack().cpu().numpy()
