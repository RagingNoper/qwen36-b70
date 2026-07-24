# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import os
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.ca_comm: None = None
        self._init_custom_ar()
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def _init_custom_ar(self):
        # Capture-safe custom all_reduce (L0-IPC one-shot + device-counter
        # barrier) to replace the oneCCL >=4-way collective that mis-replays
        # under FULL_DECODE XPUGraph. Opt-in VLLM_XPU_CUSTOM_AR=1.
        self._car = None
        self._car_null = os.environ.get("VLLM_XPU_CAR_NULL", "0") == "1"
        if os.environ.get("VLLM_XPU_CUSTOM_AR", "0") != "1":
            return
        ws = self.world_size
        if ws < 2 or (ws & (ws - 1)) != 0:
            return
        import sys  # os is imported at module scope (do NOT re-import locally -> shadows + UnboundLocalError)
        if "/work/ext" not in sys.path:
            sys.path.insert(0, "/work/ext")
        try:
            # VLLM_XPU_CAR_SO selects which baked all-reduce kernel to load (default = the DMA
            # copy-engine .so used by the latency configs). The int8-tp4 throughput config points
            # this at custom_ar.so.v4 (vec-reduce + reduce-scatter/all-gather). A dev mount over
            # /work/ext/custom_ar.so still works via the default path.
            _so = os.environ.get("VLLM_XPU_CAR_SO", "")
            if _so:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("custom_ar", _so)
                custom_ar = _ilu.module_from_spec(_spec); _spec.loader.exec_module(custom_ar)
            else:
                import custom_ar
            logger.info("XPU custom_ar loaded from %s", _so or "/work/ext/custom_ar.so")
        except Exception as e:
            logger.warning("XPU custom_ar import failed: %s", e)
            return
        import random
        rk = self.rank_in_group
        src = dist.get_global_rank(self.cpu_group, 0)
        obj = [f"vllm_car_{random.randint(0, 1 << 30)}"] if rk == 0 else [None]
        dist.broadcast_object_list(obj, src=src, group=self.cpu_group)
        self._car_max = int(__import__("os").environ.get("VLLM_XPU_CAR_MAX", 16 << 20))
        custom_ar.ar_init(rk, ws, obj[0], self._car_max)
        self._car = custom_ar
        logger.info("XPU CUSTOM all_reduce ENABLED (ws=%d rank=%d)", ws, rk)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # DIAGNOSTIC: VLLM_XPU_CAR_NULL=1 -> identity all_reduce (NO cross-GPU work).
        # Output is wrong but decode t/s = the collective-FREE ceiling, isolating how
        # much the all_reduce costs vs pure-compute decode.
        if getattr(self, "_car_null", False):
            return input_
        car = getattr(self, "_car", None)
        _nbytes = input_.numel() * input_.element_size()
        _route = __import__("os").environ.get("CAR_ROUTE", "size")
        if _route in ("capture", "capture+small") and car is not None and self.world_size >= 2:
            # capture: custom kernel while a graph records (graph-safe, baked
            #   into the decode graph); everything eager -> oneCCL.
            # capture+small: ALSO send small eager ARs (<= CAR_SMALL_MAX, e.g.
            #   the eager MTP drafter's per-layer reduces) through the custom
            #   kernel - oneCCL's small-message latency is what it was built to
            #   beat. Large eager ARs (prefill) stay on oneCCL.
            import torch as _tcap
            try: _capturing = _tcap.xpu.is_current_stream_capturing()
            except Exception: _capturing = False
            _use_custom = _capturing
            if not _use_custom and _route == "capture+small":
                _small = int(__import__("os").environ.get("CAR_SMALL_MAX", 1 << 20))
                _use_custom = _nbytes <= _small
            if __import__("os").environ.get("CAR_SIZE_DEBUG"):
                import sys as _s2; _s2.stderr.write(f"[CARSZ] bytes={_nbytes} ({_nbytes/1024:.0f}KB) -> {'CUSTOM' if _use_custom else 'oneCCL'}  capturing={_capturing} [route={_route}]\n"); _s2.stderr.flush()
            if _use_custom:
                return car.ar_all_reduce(input_.contiguous())
            _out = input_.clone(); __import__("torch").distributed.all_reduce(_out, group=self.device_group); return _out
        if __import__("os").environ.get("CAR_SIZE_DEBUG"):
            import sys as _sca
            try: _cap = __import__("torch").xpu.is_current_stream_capturing()
            except Exception: _cap = "?"
            _use = "CUSTOM" if (car is not None and self.world_size >= 2 and _nbytes <= self._car_max) else "oneCCL"
            _sca.stderr.write(f"[CARSZ] bytes={_nbytes} ({_nbytes/1024:.0f}KB) -> {_use}  capturing={_cap}\n"); _sca.stderr.flush()
        if (car is not None and self.world_size >= 2
                and _nbytes <= self._car_max):
            return car.ar_all_reduce(input_.contiguous())
        output = input_.clone()
        dist.all_reduce(output, group=self.device_group)
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], input_, group=self.device_group)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
