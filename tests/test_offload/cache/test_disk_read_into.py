# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.cache import disk_utils
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.disk_utils import _opened_raw, disk_load_context
from compressed_tensors.offload.cache.utils import load_disk_tensor_from_offload
from compressed_tensors.offload.module import subgraph_stage_modules
from loguru import logger
from safetensors import SafetensorError
from safetensors.torch import save_file
from tests.testing_utils import requires_gpu


@pytest.fixture(autouse=True)
def isolated_disk_index(monkeypatch):
    monkeypatch.setattr(DiskCache, "index", {})


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(), (0, 3), (11,), (4, 7)])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
        torch.int64,
        torch.uint64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
def test_raw_read_preserves_bytes(tmp_path, monkeypatch, shape, dtype):
    tensor = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
    shard = tmp_path / "shard.safetensors"
    save_file({"a": torch.ones(9), "weight": tensor, "z": torch.zeros(5)}, shard)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("Same-dtype disk reads must not use Tensor.copy_")

    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    with _opened_raw(str(shard)) as file:
        result = file.get_tensor("weight")

    assert result.shape == tensor.shape
    assert result.dtype == dtype
    assert torch.equal(
        result.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8)
    )


@pytest.fixture
def emulate_pinned_allocation(monkeypatch):
    """Record pinning requests without requiring an accelerator."""
    allocations = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        allocations.append(kwargs.get("pin_memory", False))
        kwargs["pin_memory"] = False
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("cuda")
    )
    monkeypatch.setattr(torch, "empty", empty)
    return allocations


@pytest.mark.unit
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float32, torch.int8])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_pinned_stage_preserves_parameter_and_cast(
    tmp_path, emulate_pinned_allocation, target_dtype, requires_grad
):
    parameter = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        requires_grad=requires_grad,
    )
    parameter.test_metadata = "preserve"
    cache = DiskCache("cuda", offload_dir=tmp_path)
    offloaded = cache.offload(parameter)
    offloaded.requires_grad_(requires_grad)
    entry = DiskCache.index[offloaded]
    save_file({"weight": parameter.data.float()}, entry["safetensors_file"])
    entry["dtype"] = str(target_dtype).removeprefix("torch.")
    emulate_pinned_allocation.clear()

    staged = cache.stage(offloaded, pin_memory=True)

    assert isinstance(staged, torch.nn.Parameter)
    assert staged.test_metadata == parameter.test_metadata
    assert staged.requires_grad == (requires_grad and target_dtype.is_floating_point)
    assert staged.dtype == target_dtype
    assert torch.equal(staged, parameter.detach().to(target_dtype))
    expected_allocations = [target_dtype == torch.float32]
    if target_dtype != torch.float32:
        expected_allocations.append(True)
    assert emulate_pinned_allocation == expected_allocations


@pytest.mark.unit
def test_subgraph_stages_parameters_and_buffers_directly(
    tmp_path, emulate_pinned_allocation
):
    module = torch.nn.Linear(4, 4)
    module.register_buffer("scale", torch.ones(4))
    reference = {name: value.clone() for name, value in module.state_dict().items()}
    offload_module(
        module, onload_device="cuda", offload_device="disk", offload_dir=tmp_path
    )
    emulate_pinned_allocation.clear()

    subgraph_stage_modules({"linear": module}, pin_memory=True)

    assert module._parameters.is_staged
    assert module._buffers.is_staged
    assert not hasattr(module, "_original_forward_func")
    assert isinstance(module._parameters.offloaded_values["weight"], torch.nn.Parameter)
    for cache in (module._parameters, module._buffers):
        for name, tensor in cache.offloaded_values.items():
            assert torch.equal(tensor, reference[name])
    assert emulate_pinned_allocation == [True, True, True]


@pytest.mark.unit
@pytest.mark.parametrize("onload_device", ["cpu", torch.device("cpu"), "cuda"])
@pytest.mark.parametrize("accelerator_available", [False, True])
def test_stage_without_pinning_uses_existing_loader(
    tmp_path, monkeypatch, onload_device, accelerator_available
):
    cache = DiskCache(onload_device, offload_dir=tmp_path)
    tensor = torch.arange(10)
    offloaded = cache.offload(tensor)
    monkeypatch.setattr(
        torch.accelerator, "is_available", lambda: accelerator_available
    )

    def unexpected_raw_open(*args, **kwargs):
        pytest.fail("Unpinned staging must use the existing safetensors loader")

    monkeypatch.setattr(disk_utils, "_RawSafetensorsFile", unexpected_raw_open)
    assert torch.equal(cache.stage(offloaded), tensor)
    assert cache.stage(None, pin_memory=True) is None
    if not accelerator_available or torch.device(onload_device).type == "cpu":
        assert torch.equal(cache.stage(offloaded, pin_memory=True), tensor)


@pytest.mark.unit
def test_raw_reader_reused_and_closed(tmp_path, monkeypatch):
    shard = tmp_path / "shard.safetensors"
    save_file({"a": torch.ones(3), "b": torch.zeros(7)}, shard)
    real_safe_open = disk_utils.safe_open
    opens = []

    def counted(*args, **kwargs):
        opens.append(args[0])
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(disk_utils, "safe_open", counted)
    with disk_load_context():
        with _opened_raw(str(shard)) as first:
            assert torch.equal(first.get_tensor("a"), torch.ones(3))
        with disk_load_context(), _opened_raw(str(shard)) as second:
            assert first is second
            assert torch.equal(second.get_tensor("b"), torch.zeros(7))
        assert not first.file.closed
    assert first.file.closed
    assert len(opens) == 1


@pytest.mark.unit
def test_raw_reader_reuses_checkpoint_symlinks(tmp_path, emulate_pinned_allocation):
    from tests.test_offload.cache.test_disk import _symlinked_shard

    try:
        cache, offloaded = _symlinked_shard(tmp_path, count=3)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    cache.onload_device = "cuda"
    shard_reader = None
    with disk_load_context():
        for index, meta in enumerate(offloaded):
            assert torch.equal(
                cache.stage(meta, pin_memory=True), torch.full((4,), float(index))
            )
            with _opened_raw(DiskCache.index[meta]["safetensors_file"]) as reader:
                if shard_reader is None:
                    shard_reader = reader
                assert reader is shard_reader
        cache.update_offload(offloaded[0], torch.full((4,), 99.0))
        assert torch.equal(
            cache.stage(offloaded[0], pin_memory=True), torch.full((4,), 99.0)
        )
        assert not shard_reader.file.closed
        assert torch.equal(cache.stage(offloaded[1], pin_memory=True), torch.ones(4))
    assert shard_reader.file.closed


@pytest.mark.unit
def test_raw_reader_bounds_handles_and_closes_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_utils, "_MAX_OPEN_FILES", 2)
    readers = []
    with pytest.raises(RuntimeError, match="stop"):
        with disk_load_context():
            for index in range(3):
                shard = tmp_path / f"{index}.safetensors"
                save_file({"weight": torch.ones(3)}, shard)
                with _opened_raw(str(shard)) as reader:
                    readers.append(reader)
            assert readers[0].file.closed
            raise RuntimeError("stop")
    assert all(reader.file.closed for reader in readers)


@pytest.mark.unit
def test_raw_readers_are_thread_local(tmp_path):
    shard = tmp_path / "shard.safetensors"
    tensors = {"a": torch.ones(4096), "b": torch.zeros(4096)}
    save_file(tensors, shard)
    barrier = Barrier(2)

    def read(name):
        with disk_load_context(), _opened_raw(str(shard)) as reader:
            barrier.wait(timeout=10)
            for _ in range(8):
                assert torch.equal(reader.get_tensor(name), tensors[name])
            return reader

    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = list(pool.map(read, tensors))
    assert readers[0] is not readers[1]
    assert all(reader.file.closed for reader in readers)


@pytest.mark.unit
def test_raw_reader_handles_short_reads_and_eof(tmp_path):
    shard = tmp_path / "shard.safetensors"
    tensor = torch.arange(11)
    save_file({"weight": tensor}, shard)

    class ShortReads(io.BytesIO):
        def readinto(self, buffer):
            return super().readinto(buffer[:3])

    with _opened_raw(str(shard)) as reader:
        reader.file.close()
        reader.file = ShortReads(shard.read_bytes())
        assert torch.equal(reader.get_tensor("weight"), tensor)
        reader.file.close()
        reader.file = ShortReads(shard.read_bytes()[:-1])
        with pytest.raises(EOFError, match="weight"):
            reader.get_tensor("weight")


@pytest.mark.unit
def test_raw_reader_validates_safetensors(tmp_path):
    shard = tmp_path / "invalid.safetensors"
    shard.write_bytes(b"invalid file")
    with pytest.raises(SafetensorError):
        with _opened_raw(str(shard)):
            pytest.fail("Invalid safetensors files must be rejected")


@pytest.mark.unit
def test_raw_reader_falls_back_for_unmapped_dtype(tmp_path, monkeypatch):
    shard = tmp_path / "shard.safetensors"
    tensor = torch.arange(11, dtype=torch.float32)
    save_file({"weight": tensor}, shard)
    monkeypatch.delitem(disk_utils.str_to_torch_dtype, "F32")
    with _opened_raw(str(shard)) as reader:
        result = reader.get_tensor("weight", dtype=torch.bfloat16)
    assert torch.equal(result, tensor.bfloat16())


@pytest.mark.unit
def test_pinned_allocation_failure_logs_hint(tmp_path, monkeypatch):
    cache = DiskCache("cuda", offload_dir=tmp_path)
    offloaded = cache.offload(torch.ones(3))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)

    def allocation_failure(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")

    monkeypatch.setattr(torch, "empty", allocation_failure)
    messages = []
    handler = logger.add(lambda msg: messages.append(msg.record["message"]))
    try:
        with pytest.raises(RuntimeError, match="Tried to allocate"):
            cache.stage(offloaded, pin_memory=True)
    finally:
        logger.remove(handler)
    assert any("Pinned-memory staging ran out of host RAM" in msg for msg in messages)


@pytest.mark.unit
def test_raw_reader_evicted_before_update(tmp_path, emulate_pinned_allocation):
    cache = DiskCache("cuda", offload_dir=tmp_path)
    offloaded = cache.offload(torch.ones(3))
    with disk_load_context():
        assert torch.equal(cache.stage(offloaded, pin_memory=True), torch.ones(3))
        cache.update_offload(offloaded, torch.full((3,), 5.0))
        assert torch.equal(
            cache.stage(offloaded, pin_memory=True), torch.full((3,), 5.0)
        )


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_direct_read_is_pinned_after_dtype_cast(tmp_path, monkeypatch, target_dtype):
    shard = tmp_path / "shard.safetensors"
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    save_file({"weight": tensor}, shard)
    if target_dtype == tensor.dtype:

        def unexpected_copy(*args, **kwargs):
            pytest.fail("Same-dtype pinned reads must not use Tensor.copy_")

        monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    result = load_disk_tensor_from_offload(
        {
            "safetensors_file": str(shard),
            "weight_name": "weight",
            "dtype": str(target_dtype).removeprefix("torch."),
        },
        device="cpu",
        pin_memory=True,
    )
    assert result.is_pinned()
    assert torch.equal(result, tensor.to(target_dtype))
