# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark warm-cache mmap + copy against direct disk-to-pinned reads.

    python benchmarks/benchmark_disk_pinned_staging.py --mib 16 --tensors 16

Use --pageable for CPU-only I/O. Wall time is the median; CPU time is the mean,
including a 300 ms window for copy workers to settle between samples.
"""

import argparse
import gc
import statistics
import tempfile
import time
from pathlib import Path

import torch
from compressed_tensors.offload.cache.disk_utils import _opened_raw, disk_load_context
from compressed_tensors.offload.cache.utils import load_disk_tensor_from_offload
from compressed_tensors.offload.utils import to_empty
from safetensors.torch import save_file


def run(
    mib: int, count: int, repeats: int, pin_memory: bool
) -> tuple[float, float, float, float]:
    with tempfile.TemporaryDirectory() as directory:
        shard = Path(directory) / "shard.safetensors"
        tensor = torch.arange(mib * 1024 * 1024 // 2, dtype=torch.float32).to(
            torch.bfloat16
        )
        save_file({f"w{i}": tensor.clone() for i in range(count)}, shard)
        entries = [
            {
                "safetensors_file": str(shard),
                "weight_name": f"w{i}",
                "dtype": "bfloat16",
            }
            for i in range(count)
        ]

        def legacy():
            with disk_load_context():
                for entry in entries:
                    source = load_disk_tensor_from_offload(entry, device="cpu")
                    result = to_empty(source, device="cpu", pin_memory=pin_memory)
                    with torch.no_grad():
                        result.copy_(source)
                    del source, result

        def direct():
            with disk_load_context(), _opened_raw(str(shard)) as file:
                for entry in entries:
                    file.get_tensor(entry["weight_name"], pin_memory=pin_memory)

        with _opened_raw(str(shard)) as file:
            result = file.get_tensor(f"w{count - 1}", pin_memory=pin_memory)
        assert torch.equal(result, tensor)
        assert result.is_pinned() == pin_memory
        del result, tensor
        legacy()
        direct()
        time.sleep(0.3)
        times = [[], []]
        cpu_times = [[], []]
        for repeat in range(repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                gc.collect()
                cpu_start = time.process_time()
                start = time.perf_counter()
                (legacy, direct)[index]()
                times[index].append(time.perf_counter() - start)
                time.sleep(0.3)
                cpu_times[index].append(time.process_time() - cpu_start)
        return (
            statistics.median(times[0]),
            statistics.median(times[1]),
            statistics.mean(cpu_times[0]),
            statistics.mean(cpu_times[1]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--tensors", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--pageable", action="store_true")
    args = parser.parse_args()
    if min(*args.mib, args.tensors, args.repeats) <= 0:
        parser.error("Sizes, tensor count and repetitions must be positive")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if not args.pageable and not torch.accelerator.is_available():
        parser.error(
            "Pinned staging requires an accelerator; use --pageable for I/O only"
        )
    memory = "pageable (I/O only)" if args.pageable else "pinned"
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, {memory}")
    print(
        f"Warm page cache; median of {args.repeats}; {args.tensors} tensor(s) per shard"
    )
    print("CPU time includes 300 ms post-call settling; wall time excludes it")
    print("MiB/tensor  mmap+copy ms  readinto ms  speedup  old CPU ms  new CPU ms")
    for mib in args.mib:
        old, new, old_cpu, new_cpu = run(
            mib, args.tensors, args.repeats, not args.pageable
        )
        print(
            f"{mib:10d}  {old * 1e3:12.3f}  {new * 1e3:11.3f}  {old / new:7.2f}x"
            f"  {old_cpu * 1e3:10.3f}  {new_cpu * 1e3:10.3f}"
        )


if __name__ == "__main__":
    main()
