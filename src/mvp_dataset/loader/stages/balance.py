"""Distributed loader-side balance stage."""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from ...core import RuntimeContext
from ...core.resume import UnsupportedResume, callable_fingerprint, stable_fingerprint

Topology = Literal["node", "none"]


@dataclass(frozen=True, slots=True)
class RankStatus:
    """Small control-plane state shared by every rank."""

    rank: int
    node_rank: int
    local_rank: int
    available: int
    upstream_done: bool


@dataclass(frozen=True, slots=True)
class Transfer:
    """One planned sparse transfer."""

    src: int
    dst: int
    count: int


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Planned outputs and transfers for one balance chunk."""

    local_take: dict[int, int]
    transfers: list[Transfer]
    dummy_counts: dict[int, int]
    finished_after_chunk: bool


@dataclass(frozen=True, slots=True)
class _TensorRef:
    id: int


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    id: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    chunk_index: int
    offset: int
    numel: int


@dataclass(frozen=True, slots=True)
class _PackedTransfer:
    index: int
    dst: int
    control_payload: bytes
    tensors: tuple[object, ...]


class _LoaderBalanceStage:
    """TorchLoader stage configuration for distributed output balancing."""

    kind = "balance"

    def __init__(
        self,
        *,
        base_context: RuntimeContext | None,
        drop_last: bool,
        dummy_factory: Callable[[RuntimeContext], object] | None,
        buffer_size: int,
        chunk_size: int,
        max_transfer_per_round: int,
        topology: Topology,
        process_group: object | None,
        device_process_group: object | None,
    ) -> None:
        """Initialize the object."""
        if not drop_last and dummy_factory is None:
            msg = "dummy_factory is required when drop_last=False"
            raise ValueError(msg)
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if buffer_size <= 0:
            msg = f"buffer_size must be > 0, got {buffer_size}"
            raise ValueError(msg)
        if buffer_size < chunk_size:
            msg = "buffer_size must be >= chunk_size"
            raise ValueError(msg)
        if max_transfer_per_round <= 0:
            msg = f"max_transfer_per_round must be > 0, got {max_transfer_per_round}"
            raise ValueError(msg)
        if topology not in ("node", "none"):
            msg = f"topology must be 'node' or 'none', got {topology!r}"
            raise ValueError(msg)

        self.base_context = base_context
        self.drop_last = drop_last
        self.dummy_factory = dummy_factory
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.max_transfer_per_round = max_transfer_per_round
        self.topology = topology
        self.process_group = process_group
        self.device_process_group = device_process_group

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _BalanceStageIterator(
            upstream=data,
            context=RuntimeContext.from_runtime(base=self.base_context),
            drop_last=self.drop_last,
            dummy_factory=self.dummy_factory,
            buffer_size=self.buffer_size,
            chunk_size=self.chunk_size,
            max_transfer_per_round=self.max_transfer_per_round,
            topology=self.topology,
            process_group=self.process_group,
            device_process_group=self.device_process_group,
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": self.kind,
                "drop_last": self.drop_last,
                "dummy_factory": callable_fingerprint(self.dummy_factory),
                "buffer_size": self.buffer_size,
                "chunk_size": self.chunk_size,
                "max_transfer_per_round": self.max_transfer_per_round,
                "topology": self.topology,
            }
        )


class _BalanceStageIterator:
    """Iterator that balances materialized outputs across distributed ranks."""

    def __init__(
        self,
        *,
        upstream: Iterable[object],
        context: RuntimeContext,
        drop_last: bool,
        dummy_factory: Callable[[RuntimeContext], object] | None,
        buffer_size: int,
        chunk_size: int,
        max_transfer_per_round: int,
        topology: Topology,
        process_group: object | None,
        device_process_group: object | None,
    ) -> None:
        """Initialize the object."""
        self.upstream = iter(upstream)
        self.context = context
        self.drop_last = drop_last
        self.dummy_factory = dummy_factory
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.max_transfer_per_round = max_transfer_per_round
        self.topology = topology
        self.process_group = process_group
        self.device_process_group = device_process_group
        self.local_buffer: deque[object] = deque()
        self.output_chunk: deque[object] = deque()
        self.upstream_done = False
        self.tail_finished = False
        try:
            import torch.distributed as dist
        except ModuleNotFoundError:
            dist = None
        self.dist = dist

    def __iter__(self) -> Iterator[object]:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next balanced output item."""
        if self._passthrough():
            return next(self.upstream)
        if not self.output_chunk:
            self._build_next_chunk()
        if not self.output_chunk:
            raise StopIteration
        return self.output_chunk.popleft()

    def state_dict(self) -> dict[str, object]:
        """Reject resume for this stage."""
        msg = "[UnsupportedResume] loader stage kind='balance'"
        raise UnsupportedResume(msg)

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Reject resume for this stage."""
        msg = "[UnsupportedResume] loader stage kind='balance'"
        raise UnsupportedResume(msg)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": "balance",
                "drop_last": self.drop_last,
                "dummy_factory": callable_fingerprint(self.dummy_factory),
                "buffer_size": self.buffer_size,
                "chunk_size": self.chunk_size,
                "max_transfer_per_round": self.max_transfer_per_round,
                "topology": self.topology,
            }
        )

    def _passthrough(self) -> bool:
        """Return whether distributed balancing is inactive."""
        dist = self.dist
        if dist is None:
            return True
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_world_size(self.process_group) == 1

    def _build_next_chunk(self) -> None:
        """Build the next local output chunk."""
        if self.tail_finished:
            return

        dist = self.dist
        if not dist.is_available() or not dist.is_initialized():
            msg = "[DistributedUnavailable] torch.distributed must be initialized for balance()"
            raise RuntimeError(msg)
        rank = dist.get_rank()
        self._fill_local_buffer(max(self.buffer_size, dist.get_world_size(self.process_group)))

        gathered: list[RankStatus | None] = [None for _ in range(dist.get_world_size(self.process_group))]
        dist.all_gather_object(
            gathered,
            RankStatus(
                rank=rank,
                node_rank=self.context.node_rank,
                local_rank=self.context.local_rank,
                available=len(self.local_buffer),
                upstream_done=self.upstream_done,
            ),
            group=self.process_group,
        )
        plan = plan_balance_chunk(
            [status for status in gathered if status is not None],
            chunk_size=self.chunk_size,
            max_transfer_per_round=self.max_transfer_per_round,
            drop_last=self.drop_last,
            topology=self.topology,
        )

        received = _exchange_transfers(
            local_buffer=self.local_buffer,
            plan=plan,
            rank=rank,
            process_group=self.process_group,
            device_process_group=self.device_process_group,
        )

        for _ in range(plan.local_take.get(rank, 0)):
            self.output_chunk.append(self.local_buffer.popleft())
        sent = sum(transfer.count for transfer in plan.transfers if transfer.src == rank)
        for _ in range(sent):
            self.local_buffer.popleft()
        for index, transfer in enumerate(plan.transfers):
            if transfer.dst == rank:
                self.output_chunk.extend(received[index])
        for _ in range(plan.dummy_counts.get(rank, 0)):
            if self.dummy_factory is None:
                msg = "dummy_factory is required when drop_last=False"
                raise ValueError(msg)
            self.output_chunk.append(self.dummy_factory(self.context))
        self.tail_finished = plan.finished_after_chunk

    def _fill_local_buffer(self, target_size: int) -> None:
        """Prefetch materialized upstream items into the local buffer."""
        while len(self.local_buffer) < target_size and not self.upstream_done:
            try:
                item = next(self.upstream)
            except StopIteration:
                self.upstream_done = True
                break
            self.local_buffer.append(item)


def plan_balance_chunk(
    statuses: list[RankStatus],
    *,
    chunk_size: int,
    max_transfer_per_round: int,
    drop_last: bool,
    topology: Topology,
) -> ChunkPlan:
    """Return a deterministic balance plan from gathered rank statuses."""
    ordered = sorted(statuses, key=lambda status: status.rank)
    ranks = [status.rank for status in ordered]
    by_rank = {status.rank: status for status in ordered}
    remaining = {status.rank: status.available for status in ordered}
    local_take = {rank: 0 for rank in ranks}
    dummy_counts = {rank: 0 for rank in ranks}
    transfers: list[Transfer] = []
    sent_by_src = {rank: 0 for rank in ranks}
    finished = False

    for _ in range(chunk_size):
        total_remaining = sum(remaining.values())
        if total_remaining == 0:
            finished = all(status.upstream_done for status in ordered)
            break
        if total_remaining < len(ranks):
            if drop_last:
                finished = all(status.upstream_done for status in ordered)
                break
            for rank in ranks:
                if remaining[rank] > 0:
                    local_take[rank] += 1
                    remaining[rank] -= 1
                else:
                    dummy_counts[rank] += 1
            finished = all(status.upstream_done for status in ordered)
            break

        deficits: list[int] = []
        for rank in ranks:
            if remaining[rank] > 0:
                local_take[rank] += 1
                remaining[rank] -= 1
            else:
                deficits.append(rank)

        planned: list[tuple[int, int]] = []
        for dst in deficits:
            donor = _choose_donor(
                dst=dst,
                statuses=by_rank,
                remaining=remaining,
                sent_by_src=sent_by_src,
                max_transfer_per_round=max_transfer_per_round,
                topology=topology,
            )
            if donor is None:
                for rank in ranks:
                    local_take[rank] -= int(rank not in deficits)
                    remaining[rank] += int(rank not in deficits)
                for src, _ in planned:
                    remaining[src] += 1
                    sent_by_src[src] -= 1
                if not any(local_take.values()) and not any(dummy_counts.values()) and not transfers:
                    msg = "max_transfer_per_round is too small to build one balanced step"
                    raise ValueError(msg)
                return ChunkPlan(
                    local_take=local_take,
                    transfers=_merge_transfers(transfers),
                    dummy_counts=dummy_counts,
                    finished_after_chunk=False,
                )
            planned.append((donor, dst))
            remaining[donor] -= 1
            sent_by_src[donor] += 1

        for src, dst in planned:
            transfers.append(Transfer(src=src, dst=dst, count=1))

    return ChunkPlan(
        local_take=local_take,
        transfers=_merge_transfers(transfers),
        dummy_counts=dummy_counts,
        finished_after_chunk=finished,
    )


def _choose_donor(
    *,
    dst: int,
    statuses: dict[int, RankStatus],
    remaining: dict[int, int],
    sent_by_src: dict[int, int],
    max_transfer_per_round: int,
    topology: Topology,
) -> int | None:
    """Return the best donor rank for one deficit rank."""
    candidates = [
        rank
        for rank, available in remaining.items()
        if available > 0 and sent_by_src[rank] < max_transfer_per_round and rank != dst
    ]
    if not candidates:
        return None
    if topology == "node":
        dst_node = statuses[dst].node_rank
        candidates.sort(key=lambda rank: (statuses[rank].node_rank != dst_node, -remaining[rank], rank))
    else:
        candidates.sort(key=lambda rank: (-remaining[rank], rank))
    return candidates[0]


def _merge_transfers(transfers: list[Transfer]) -> list[Transfer]:
    """Merge adjacent transfer units with the same source and destination."""
    merged: list[Transfer] = []
    for transfer in transfers:
        if merged and merged[-1].src == transfer.src and merged[-1].dst == transfer.dst:
            previous = merged.pop()
            merged.append(Transfer(src=previous.src, dst=previous.dst, count=previous.count + transfer.count))
        else:
            merged.append(transfer)
    return merged


def _exchange_transfers(
    *,
    local_buffer: deque[object],
    plan: ChunkPlan,
    rank: int,
    process_group: object | None,
    device_process_group: object | None,
) -> dict[int, list[object]]:
    """Exchange planned transfer payloads and return received items by transfer index."""
    outgoing: list[_PackedTransfer] = []
    send_offset = plan.local_take.get(rank, 0)
    buffered = list(local_buffer)
    for index, transfer in enumerate(plan.transfers):
        if transfer.src != rank:
            continue
        items = buffered[send_offset : send_offset + transfer.count]
        send_offset += transfer.count
        control_payload, tensors = _pack_transfer_items(items)
        outgoing.append(
            _PackedTransfer(
                index=index,
                dst=transfer.dst,
                control_payload=control_payload,
                tensors=tuple(tensors),
            )
        )

    incoming = [(index, transfer.src) for index, transfer in enumerate(plan.transfers) if transfer.dst == rank]
    if not outgoing and not incoming:
        return {}

    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        msg = "[DistributedUnavailable] torch.distributed must be initialized for balance()"
        raise RuntimeError(msg)
    received: dict[int, list[object]] = {}
    controls: dict[int, dict[str, object]] = {}
    recv_tensors: dict[int, list[object]] = {}
    live_tensors: list[object] = []
    size_recvs: list[tuple[int, int, object, object]] = []
    data_recvs: list[tuple[int, int, object, object]] = []
    size_send_reqs: list[object] = []
    data_send_reqs: list[object] = []

    for index, src in incoming:
        tensor = torch.empty(1, dtype=torch.long)
        request = dist.irecv(tensor, src=src, group=process_group, tag=29300 + index * 2)
        size_recvs.append((index, src, tensor, request))

    for packed in outgoing:
        tensor = torch.tensor([len(packed.control_payload)], dtype=torch.long)
        live_tensors.append(tensor)
        size_send_reqs.append(dist.isend(tensor, dst=packed.dst, group=process_group, tag=29300 + packed.index * 2))

    for _, _, _, request in size_recvs:
        request.wait() if hasattr(request, "wait") else None
    for request in size_send_reqs:
        request.wait()

    for index, src, size_tensor, _ in size_recvs:
        tensor = torch.empty(int(size_tensor.item()), dtype=torch.uint8)
        request = dist.irecv(tensor, src=src, group=process_group, tag=29301 + index * 2)
        data_recvs.append((index, src, tensor, request))

    for packed in outgoing:
        tensor = torch.tensor(list(packed.control_payload), dtype=torch.uint8)
        live_tensors.append(tensor)
        data_send_reqs.append(dist.isend(tensor, dst=packed.dst, group=process_group, tag=29301 + packed.index * 2))

    for index, _, tensor, request in data_recvs:
        request.wait()
        control = pickle.loads(bytes(tensor.tolist()))
        if not isinstance(control, dict):
            msg = "[InvalidBalanceTransfer] transfer control payload must decode to a dict"
            raise RuntimeError(msg)
        controls[index] = control
    for request in data_send_reqs:
        request.wait()

    _exchange_tensors(
        incoming=incoming,
        outgoing=outgoing,
        controls=controls,
        recv_tensors=recv_tensors,
        rank=rank,
        plan=plan,
        process_group=process_group,
        device_process_group=device_process_group,
    )

    for index, _ in incoming:
        items = _restore_transfer_items(controls[index], recv_tensors.get(index, []))
        if not isinstance(items, list):
            msg = "[InvalidBalanceTransfer] transfer payload must decode to a list"
            raise RuntimeError(msg)
        received[index] = items
    _ = live_tensors
    return received


def _pack_transfer_items(items: list[object]) -> tuple[bytes, list[object]]:
    """Return a pickled control payload plus merged tensor payloads."""
    import torch

    tensors: list[tuple[int, object]] = []

    def split(value: object) -> object:
        if isinstance(value, torch.Tensor):
            ref_id = len(tensors)
            tensors.append((ref_id, value))
            return _TensorRef(ref_id)
        if isinstance(value, dict):
            return {key: split(item) for key, item in value.items()}
        if isinstance(value, list):
            return [split(item) for item in value]
        if isinstance(value, tuple):
            return tuple(split(item) for item in value)
        return value

    skeleton = split(items)
    specs: list[_TensorSpec | None] = [None for _ in tensors]
    chunk_tensors: list[object] = []
    by_key: dict[tuple[str, str], list[tuple[int, object]]] = {}
    for ref_id, tensor in tensors:
        by_key.setdefault((str(tensor.device), str(tensor.dtype).removeprefix("torch.")), []).append((ref_id, tensor))

    for chunk_index, (device, dtype) in enumerate(sorted(by_key)):
        parts: list[object] = []
        offset = 0
        for ref_id, tensor in by_key[(device, dtype)]:
            flat = tensor.contiguous().reshape(-1)
            numel = int(flat.numel())
            parts.append(flat)
            specs[ref_id] = _TensorSpec(
                id=ref_id,
                shape=tuple(int(size) for size in tensor.shape),
                dtype=dtype,
                device=device,
                chunk_index=chunk_index,
                offset=offset,
                numel=numel,
            )
            offset += numel
        chunk_tensors.append(torch.cat(parts) if len(parts) > 1 else parts[0])

    control = {
        "items": skeleton,
        "tensor_specs": [spec for spec in specs if spec is not None],
    }
    return pickle.dumps(control, protocol=pickle.HIGHEST_PROTOCOL), chunk_tensors


def _exchange_tensors(
    *,
    incoming: list[tuple[int, int]],
    outgoing: list[_PackedTransfer],
    controls: dict[int, dict[str, object]],
    recv_tensors: dict[int, list[object]],
    rank: int,
    plan: ChunkPlan,
    process_group: object | None,
    device_process_group: object | None,
) -> None:
    """Exchange merged tensor payloads."""
    import torch
    import torch.distributed as dist

    packed_by_index = {packed.index: packed for packed in outgoing}
    incoming_by_index = dict(incoming)
    cpu_ops: list[object] = []
    device_ops: list[object] = []

    for index, transfer in enumerate(plan.transfers):
        if transfer.dst == rank:
            control = controls[index]
            tensor_specs = control.get("tensor_specs")
            if not isinstance(tensor_specs, list) or not all(isinstance(spec, _TensorSpec) for spec in tensor_specs):
                msg = "[InvalidBalanceTransfer] tensor_specs must be a list of tensor specs"
                raise RuntimeError(msg)
            chunk_specs: dict[int, tuple[str, str, int]] = {}
            for spec in tensor_specs:
                current = chunk_specs.get(spec.chunk_index)
                numel = max(spec.offset + spec.numel, current[2] if current is not None else 0)
                if current is not None and current[:2] != (spec.dtype, spec.device):
                    msg = "[InvalidBalanceTransfer] inconsistent tensor chunk metadata"
                    raise RuntimeError(msg)
                chunk_specs[spec.chunk_index] = (spec.dtype, spec.device, numel)
            if sorted(chunk_specs) != list(range(len(chunk_specs))):
                msg = "[InvalidBalanceTransfer] tensor chunk indexes must be contiguous"
                raise RuntimeError(msg)
            tensors = []
            for _, (dtype_name, device, numel) in sorted(chunk_specs.items()):
                dtype = getattr(torch, dtype_name, None)
                if dtype is None:
                    msg = f"[InvalidBalanceTransfer] unknown tensor dtype {dtype_name!r}"
                    raise RuntimeError(msg)
                tensors.append(torch.empty(numel, dtype=dtype, device=device))
            recv_tensors[index] = tensors
            src = incoming_by_index[index]
            for tensor in tensors:
                group = process_group if tensor.device.type == "cpu" else device_process_group or process_group
                op = dist.P2POp(dist.irecv, tensor, src, group=group)
                if tensor.device.type == "cpu":
                    cpu_ops.append(op)
                else:
                    device_ops.append(op)
        if transfer.src == rank:
            packed = packed_by_index[index]
            for tensor in packed.tensors:
                group = process_group if tensor.device.type == "cpu" else device_process_group or process_group
                op = dist.P2POp(dist.isend, tensor, packed.dst, group=group)
                if tensor.device.type == "cpu":
                    cpu_ops.append(op)
                else:
                    device_ops.append(op)

    for ops in (cpu_ops, device_ops):
        if not ops:
            continue
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()


def _restore_transfer_items(control: dict[str, object], tensors: list[object]) -> object:
    """Rebuild transfer items from a skeleton and received tensors."""
    specs_raw = control.get("tensor_specs")
    if not isinstance(specs_raw, list):
        msg = "[InvalidBalanceTransfer] tensor_specs must be a list"
        raise RuntimeError(msg)
    specs: dict[int, _TensorSpec] = {}
    for spec in specs_raw:
        if not isinstance(spec, _TensorSpec):
            msg = "[InvalidBalanceTransfer] invalid tensor spec"
            raise RuntimeError(msg)
        specs[spec.id] = spec

    def restore(value: object) -> object:
        if isinstance(value, _TensorRef):
            spec = specs[value.id]
            tensor = tensors[spec.chunk_index]
            return tensor.narrow(0, spec.offset, spec.numel).reshape(spec.shape)
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, tuple):
            return tuple(restore(item) for item in value)
        return value

    return restore(control.get("items"))
