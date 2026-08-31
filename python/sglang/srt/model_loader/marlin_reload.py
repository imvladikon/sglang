# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming reload support for non-canonical quantized kernel layouts.

The implementation follows the layer-wise restore/process/copy-back contract
introduced by vLLM PR #32133.  SGLang's ordinary tensor update path receives
checkpoint/model-format tensors, while an FP8 Marlin layer is already in a
different kernel format.  Loading directly into that packed storage is invalid.

Marlin layers are always enrolled. Other quantized layers are enrolled only
when post-processing changes their tensor contract, while canonical native
block-FP8 layers keep the existing direct update path.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from inspect import BoundArguments
from types import MethodType
from weakref import WeakKeyDictionary

import torch
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.model_loader.weight_utils import default_weight_loader
from torch.utils._python_dispatch import TorchDispatchMode

LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
_LAYER_REF_SENTINEL = object()


@dataclass
class _MarlinReloadInfo:
    restore_tensors: LayerTensors
    restore_device: torch.device
    restore_none_parameters: set[str] = field(default_factory=set)
    restore_none_buffers: set[str] = field(default_factory=set)
    restore_non_persistent_buffers: set[str] = field(default_factory=set)
    kernel_tensors: LayerTensors | None = None
    kernel_none_parameters: set[str] = field(default_factory=set)
    kernel_none_buffers: set[str] = field(default_factory=set)
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)
    load_numel: int = 0
    load_numel_total: int = 0
    enabled: bool = False
    active: bool = False

    def reset_transaction(self) -> None:
        self.kernel_tensors = None
        self.kernel_none_parameters.clear()
        self.kernel_none_buffers.clear()
        self.kernel_non_persistent_buffers.clear()
        self.loaded_weights.clear()
        self.load_numel = 0
        self.load_numel_total = 0
        self.active = False


_RELOAD_INFO: WeakKeyDictionary[torch.nn.Module, _MarlinReloadInfo] = (
    WeakKeyDictionary()
)


def _direct_tensors(layer: torch.nn.Module) -> LayerTensors:
    return (
        {name: value for name, value in layer._parameters.items() if value is not None},
        {name: value for name, value in layer._buffers.items() if value is not None},
    )


def _all_direct_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    parameters, buffers = _direct_tensors(layer)
    return parameters | buffers


def _none_tensor_names(layer: torch.nn.Module) -> tuple[set[str], set[str]]:
    return (
        {name for name, value in layer._parameters.items() if value is None},
        {name for name, value in layer._buffers.items() if value is None},
    )


def _sanitize_layer_refs(tensor: torch.Tensor, layer: torch.nn.Module) -> None:
    for name, value in tensor.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is layer:
            tensor.__dict__[name] = value.__func__.__get__(_LAYER_REF_SENTINEL)


def _restore_layer_refs(tensor: torch.Tensor, layer: torch.nn.Module) -> None:
    for name, value in tensor.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is _LAYER_REF_SENTINEL:
            tensor.__dict__[name] = value.__func__.__get__(layer)


def _to_meta_tensor(tensor: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    meta = tensor.data.to(device="meta")
    meta.__class__ = tensor.__class__
    meta.__dict__ = tensor.__dict__.copy()
    _sanitize_layer_refs(meta, layer)
    return meta


def _capture_restore_tensors(layer: torch.nn.Module) -> LayerTensors:
    parameters, buffers = _direct_tensors(layer)
    return (
        {name: _to_meta_tensor(value, layer) for name, value in parameters.items()},
        {name: _to_meta_tensor(value, layer) for name, value in buffers.items()},
    )


def record_marlin_reload_metadata(
    model: torch.nn.Module, target_device: torch.device
) -> None:
    """Record model-format tensors before initial quantization post-processing."""
    enrolled = 0
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            continue
        none_parameters, none_buffers = _none_tensor_names(layer)
        enabled = bool(getattr(quant_method, "use_marlin", False))
        _RELOAD_INFO[layer] = _MarlinReloadInfo(
            restore_tensors=_capture_restore_tensors(layer),
            restore_device=target_device,
            restore_none_parameters=none_parameters,
            restore_none_buffers=none_buffers,
            restore_non_persistent_buffers=set(layer._non_persistent_buffers_set),
            enabled=enabled,
        )
        enrolled += int(enabled)
    model._sglang_marlin_reload_layers = enrolled
    model._sglang_marlin_reload_active = False


def _tensor_signature(
    tensors: LayerTensors,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    parameters, buffers = tensors
    return {
        f"parameter:{name}": (tuple(value.shape), value.dtype)
        for name, value in parameters.items()
    } | {
        f"buffer:{name}": (tuple(value.shape), value.dtype)
        for name, value in buffers.items()
    }


def _uses_transformed_scale(tensors: LayerTensors) -> bool:
    parameters, buffers = tensors
    return any(
        getattr(value, "format_ue8m0", False)
        for value in (*parameters.values(), *buffers.values())
    )


def finalize_marlin_reload_metadata(model: torch.nn.Module) -> None:
    """Enable reload only for layers whose kernel format is non-canonical.

    Marlin is always included. Other quantized backends are included when
    post-processing changes tensor names/shapes/dtypes or transforms scales
    (for example Blackwell DeepGEMM's UE8M0 layout). Native Hopper block-FP8
    keeps its canonical layout and therefore stays on the direct update path.
    """
    enrolled = 0
    for layer in model.modules():
        info = _RELOAD_INFO.get(layer)
        if info is None:
            continue
        quant_method = getattr(layer, "quant_method", None)
        info.enabled = bool(getattr(quant_method, "use_marlin", False)) or (
            _tensor_signature(info.restore_tensors)
            != _tensor_signature(_direct_tensors(layer))
            or info.restore_none_parameters != _none_tensor_names(layer)[0]
            or info.restore_none_buffers != _none_tensor_names(layer)[1]
            or _uses_transformed_scale(_direct_tensors(layer))
        )
        enrolled += int(info.enabled)
    model._sglang_marlin_reload_layers = enrolled


def has_marlin_reload_metadata(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "_sglang_marlin_reload_layers", 0))


def _delete_direct_tensors(layer: torch.nn.Module) -> None:
    for name in list(layer._parameters) + list(layer._buffers):
        if hasattr(layer, name):
            delattr(layer, name)


def _restore_model_format_on_meta(
    layer: torch.nn.Module, info: _MarlinReloadInfo
) -> None:
    _delete_direct_tensors(layer)
    parameters, buffers = info.restore_tensors
    for name, template in parameters.items():
        value = template.data.to(device="meta")
        value.__class__ = template.__class__
        value.__dict__ = template.__dict__.copy()
        _restore_layer_refs(value, layer)
        layer.register_parameter(name, value)
    for name in info.restore_none_parameters:
        layer.register_parameter(name, None)
    for name, template in buffers.items():
        value = template.data.to(device="meta")
        value.__class__ = template.__class__
        value.__dict__ = template.__dict__.copy()
        _restore_layer_refs(value, layer)
        layer.register_buffer(
            name,
            value,
            persistent=name not in info.restore_non_persistent_buffers,
        )
    for name in info.restore_none_buffers:
        layer.register_buffer(
            name,
            None,
            persistent=name not in info.restore_non_persistent_buffers,
        )


def _materialize_tensor(meta: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = torch.empty_strided(
        tuple(meta.shape),
        tuple(meta.stride()),
        dtype=meta.dtype,
        device=device,
        requires_grad=False,
    )
    value.__class__ = meta.__class__
    value.__dict__ = meta.__dict__.copy()
    return value


def _materialize_layer(layer: torch.nn.Module, info: _MarlinReloadInfo) -> None:
    for name, value in list(_all_direct_tensors(layer).items()):
        if not value.is_meta:
            continue
        materialized = _materialize_tensor(value, info.restore_device)
        _restore_layer_refs(materialized, layer)
        setattr(layer, name, materialized)


def _get_weight_loader(tensor: torch.Tensor) -> Callable:
    return getattr(tensor, "weight_loader", None) or default_weight_loader


def _set_weight_loader(tensor: torch.Tensor, loader: Callable) -> None:
    # SGLang's BasevLLMParameter exposes a read-only ``weight_loader``
    # property backed by ``_weight_loader``. Plain Parameters used by a few
    # model implementations store the callable directly.
    if hasattr(tensor, "_weight_loader"):
        tensor._weight_loader = loader
    else:
        tensor.weight_loader = loader


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    loader = _get_weight_loader(tensor)
    while getattr(loader, "__name__", None) == "marlin_reload_loader":
        loader = loader.__wrapped__
    return loader


class _CopyCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if (
            func is torch.ops.aten.copy_.default
            or func in (torch.ops.aten.fill_.Scalar, torch.ops.aten.fill_.Tensor)
        ) and args[0].device.type == "meta":
            self.copied_numel += args[0].numel()
        return func(*args, **kwargs)


def _count_loaded_numel(
    loader: Callable, bound_args: BoundArguments
) -> tuple[int, object]:
    with _CopyCounter() as counter:
        result = loader(*bound_args.args, **bound_args.kwargs)
    param = bound_args.arguments.get("param")
    count = counter.copied_numel
    if isinstance(param, torch.Tensor):
        count = min(count, param.numel())
    return count, result


def _wrap_weight_loaders(layer: torch.nn.Module, info: _MarlinReloadInfo) -> None:
    for param_name, param in _all_direct_tensors(layer).items():
        loader = _get_original_loader(param)
        signature = inspect.signature(loader)

        @wraps(loader, assigned=("__doc__", "__annotations__"))
        def marlin_reload_loader(
            *args,
            __layer=layer,
            __info=info,
            __name=param_name,
            __loader=loader,
            __signature=signature,
            **kwargs,
        ):
            if not __info.active:
                raise RuntimeError(
                    "FP8 Marlin reload loader used outside a transaction"
                )
            bound = __signature.bind(*args, **kwargs)
            bound.apply_defaults()
            __info.loaded_weights.append((__name, bound))
            loaded_numel, result = _count_loaded_numel(__loader, bound)
            __info.load_numel += loaded_numel
            if __info.load_numel > __info.load_numel_total:
                raise RuntimeError(
                    "FP8 Marlin reload received more tensor elements than the "
                    f"model-format layer owns: {__info.load_numel} > "
                    f"{__info.load_numel_total}"
                )
            if __info.load_numel == __info.load_numel_total:
                _process_layer(__layer, __info)
            return result

        marlin_reload_loader.__name__ = "marlin_reload_loader"
        marlin_reload_loader.__wrapped__ = loader
        _set_weight_loader(param, marlin_reload_loader)


def _validate_kernel_layout(layer: torch.nn.Module, info: _MarlinReloadInfo) -> None:
    assert info.kernel_tensors is not None
    old_parameters, old_buffers = info.kernel_tensors
    new_parameters, new_buffers = _direct_tensors(layer)
    new_none_parameters, new_none_buffers = _none_tensor_names(layer)
    if old_parameters.keys() != new_parameters.keys():
        raise RuntimeError(
            "FP8 Marlin reload changed kernel parameter names: "
            f"old={sorted(old_parameters)}, new={sorted(new_parameters)}"
        )
    if old_buffers.keys() != new_buffers.keys():
        raise RuntimeError(
            "FP8 Marlin reload changed kernel buffer names: "
            f"old={sorted(old_buffers)}, new={sorted(new_buffers)}"
        )
    if info.kernel_none_parameters != new_none_parameters:
        raise RuntimeError(
            "FP8 Marlin reload changed None parameter registrations: "
            f"old={sorted(info.kernel_none_parameters)}, "
            f"new={sorted(new_none_parameters)}"
        )
    if info.kernel_none_buffers != new_none_buffers:
        raise RuntimeError(
            "FP8 Marlin reload changed None buffer registrations: "
            f"old={sorted(info.kernel_none_buffers)}, "
            f"new={sorted(new_none_buffers)}"
        )
    for name, old in old_parameters.items():
        new = new_parameters[name]
        if (old.shape, old.dtype, old.device) != (new.shape, new.dtype, new.device):
            raise RuntimeError(
                f"FP8 Marlin kernel layout changed for parameter {name}: "
                f"old={(tuple(old.shape), old.dtype, old.device)}, "
                f"new={(tuple(new.shape), new.dtype, new.device)}"
            )
    for name, old in old_buffers.items():
        new = new_buffers[name]
        if (old.shape, old.dtype, old.device) != (new.shape, new.dtype, new.device):
            raise RuntimeError(
                f"FP8 Marlin kernel layout changed for buffer {name}: "
                f"old={(tuple(old.shape), old.dtype, old.device)}, "
                f"new={(tuple(new.shape), new.dtype, new.device)}"
            )


def _restore_kernel_tensors(layer: torch.nn.Module, info: _MarlinReloadInfo) -> None:
    assert info.kernel_tensors is not None
    _delete_direct_tensors(layer)
    parameters, buffers = info.kernel_tensors
    for name, value in parameters.items():
        layer.register_parameter(name, value)
    for name in info.kernel_none_parameters:
        layer.register_parameter(name, None)
    for name, value in buffers.items():
        layer.register_buffer(
            name,
            value,
            persistent=name not in info.kernel_non_persistent_buffers,
        )
    for name in info.kernel_none_buffers:
        layer.register_buffer(
            name,
            None,
            persistent=name not in info.kernel_non_persistent_buffers,
        )


@torch.no_grad()
def _process_layer(layer: torch.nn.Module, info: _MarlinReloadInfo) -> None:
    _materialize_layer(layer, info)
    for value in _all_direct_tensors(layer).values():
        _set_weight_loader(value, _get_original_loader(value))

    for param_name, bound in info.loaded_weights:
        param = getattr(layer, param_name)
        bound.arguments["param"] = param
        _get_original_loader(param)(*bound.args, **bound.kwargs)

    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, QuantizeMethodBase):
        raise TypeError(
            f"FP8 Marlin reload expected QuantizeMethodBase, got {type(quant_method)}"
        )
    quant_method.process_weights_after_loading(layer)
    _validate_kernel_layout(layer, info)

    assert info.kernel_tensors is not None
    old_parameters, old_buffers = info.kernel_tensors
    new_parameters, new_buffers = _direct_tensors(layer)
    for name, old in old_parameters.items():
        old.data.copy_(new_parameters[name].data)
    for name, old in old_buffers.items():
        old.data.copy_(new_buffers[name].data)
    _restore_kernel_tensors(layer, info)
    info.reset_transaction()


def begin_marlin_reload(model: torch.nn.Module) -> bool:
    """Start a streaming checkpoint/model-format update transaction."""
    if not has_marlin_reload_metadata(model):
        return False
    if getattr(model, "_sglang_marlin_reload_active", False):
        return True

    for layer in model.modules():
        info = _RELOAD_INFO.get(layer)
        if info is None or not info.enabled:
            continue
        info.kernel_tensors = _direct_tensors(layer)
        (
            info.kernel_none_parameters,
            info.kernel_none_buffers,
        ) = _none_tensor_names(layer)
        info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)
        _restore_model_format_on_meta(layer, info)
        info.loaded_weights.clear()
        info.load_numel = 0
        info.load_numel_total = sum(
            value.numel() for value in _all_direct_tensors(layer).values()
        )
        info.active = True
        _wrap_weight_loaders(layer, info)

    model._sglang_marlin_reload_active = True
    return True


def abort_marlin_reload(model: torch.nn.Module) -> None:
    """Restore live kernel tensors for layers not yet processed."""
    if not getattr(model, "_sglang_marlin_reload_active", False):
        return
    for layer in model.modules():
        info = _RELOAD_INFO.get(layer)
        if info is None or not info.enabled or not info.active:
            continue
        _restore_kernel_tensors(layer, info)
        info.reset_transaction()
    model._sglang_marlin_reload_active = False


def finalize_marlin_reload(model: torch.nn.Module) -> None:
    """Finish a streaming update, rejecting incomplete Marlin layers."""
    if not getattr(model, "_sglang_marlin_reload_active", False):
        return

    incomplete = []
    for module_name, layer in model.named_modules():
        info = _RELOAD_INFO.get(layer)
        if info is None or not info.enabled or not info.active:
            continue
        if info.load_numel == 0:
            _restore_kernel_tensors(layer, info)
            info.reset_transaction()
            continue
        incomplete.append(
            f"{module_name or '<root>'}: {info.load_numel}/{info.load_numel_total}"
        )

    if incomplete:
        abort_marlin_reload(model)
        raise RuntimeError(
            "Incomplete FP8 Marlin weight update; every touched layer must receive "
            "all model-format weights and scales before repacking: "
            + ", ".join(incomplete)
        )
    model._sglang_marlin_reload_active = False
