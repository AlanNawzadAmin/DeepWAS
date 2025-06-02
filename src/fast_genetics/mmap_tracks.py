import json
from pathlib import Path

import numpy as np
from numba import jit
from tqdm import tqdm

# from .hdf5_tracks import fill_windows_chunk_fast


def create_chunked_mmap(bw_list, chroms, out_dir, chunk_size=1_000_000):
    """Creates memory-mapped arrays optimized for random access."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        # Don't use rmtree - could be dangerous. Instead, raise error if dir exists
        raise ValueError(f"Output directory {out_dir} already exists")
    out_dir.mkdir(parents=True)

    metadata = {"chunk_size": chunk_size, "chromosomes": {}}

    for chrom in chroms:
        chrom_dir = out_dir / f"chr{chrom}"
        chrom_dir.mkdir()
        chrom_metadata = {}

        for track_idx, bw in tqdm(enumerate(bw_list), desc=f"Processing chr{chrom}"):
            intervals = bw.intervals(f"chr{chrom}")
            if not intervals:
                continue

            intervals = bw.intervals(f"chr{chrom}")
            if not intervals:
                continue

            # Create a structured array combining all three values
            dtype = np.dtype([("start", np.int32), ("end", np.int32), ("value", np.float32)])
            data = np.array([(x[0], x[1], x[2]) for x in intervals], dtype=dtype)

            # Compute chunk indices
            max_pos = int(data["end"].max())
            chunk_bounds = np.arange(0, max_pos + chunk_size, chunk_size, dtype=np.int32)
            chunk_start_idx = np.searchsorted(data["start"], chunk_bounds[:-1])
            chunk_end_idx = np.searchsorted(data["start"], chunk_bounds[1:])

            # Save as a single structured array
            np.save(chrom_dir / f"track_{track_idx}.npy", data)

            # Store minimal metadata
            chrom_metadata[f"track_{track_idx}"] = {
                "length": len(data),
                "chunk_start_idx": chunk_start_idx.tolist(),
                "chunk_end_idx": chunk_end_idx.tolist(),
            }

        metadata["chromosomes"][f"chr{chrom}"] = chrom_metadata

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def extract_windows(positions, chrom, mmap_dir, n_tracks, mmapped_files={}, window_size=100, nan_value=0, print_=False):
    """Extract windows using direct memory mapping."""
    # t0 = time.time()
    mmap_dir = Path(mmap_dir)
    pos_np = np.asarray(positions, dtype=np.int32)  # Ensure int32
    sort_inds = np.argsort(pos_np)
    pos_np = pos_np[sort_inds]
    mean_pos = int(np.median(pos_np))
    pos_np = np.where(np.abs(pos_np - mean_pos) < 3e6, pos_np, mean_pos)

    # Initialize output with float32
    out = np.full((n_tracks, len(positions), window_size), nan_value, dtype=np.float32)
    half = window_size // 2

    # Load metadata once
    with open(mmap_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    chunk_size = metadata["chunk_size"]
    chrom_key = f"chr{chrom}"
    if chrom_key not in metadata["chromosomes"]:
        return out

    chrom_metadata = metadata["chromosomes"][chrom_key]

    # Pre-calculate chunk range
    min_chunk = (pos_np.min() - half) // chunk_size
    max_chunk = (pos_np.max() + half) // chunk_size + 1

    for track in range(n_tracks):
        track_key = f"track_{track}"
        if track_key not in chrom_metadata:
            continue

        track_meta = chrom_metadata[track_key]

        # Convert indices to int32
        chunk_start_idx = np.array(track_meta["chunk_start_idx"], dtype=np.int32)
        chunk_end_idx = np.array(track_meta["chunk_end_idx"], dtype=np.int32)

        min_chunk = max(0, min(len(chunk_start_idx) - 1, min_chunk))
        max_chunk = max(0, min(len(chunk_end_idx), max_chunk))
        total_start = chunk_start_idx[min_chunk]
        total_end = chunk_end_idx[max_chunk - 1]

        # Create memory maps with exact offset and length
        # t0 = time.time()
        data = mmapped_files[f"track_{track}"][total_start:total_end]
        starts = data["start"]
        ends = data["end"]
        values = data["value"]

        # process
        # if print_:
        #     print("load", time.time() - t0); t0 = time.time()
        fill_windows_chunk_fast(
            pos_np,
            starts,
            ends,
            values,
            out[track],
            track,
            window_size,
        )
        # if print_:
        #     print("chunk", time.time() - t0, total_end - total_start)
        #     t0 = time.time()
    return np.swapaxes(out, 0, 1)[np.argsort(sort_inds)]


@jit(nopython=True)
def reverse_search(arr, target):
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] <= target:
            return i
    return -1


@jit(nopython=True)
def fwd_search(arr, target):
    for i in range(len(arr)):
        if arr[i] >= target:
            return i - 1
    return -1


@jit(nopython=True, fastmath=True, cache=True)
def fill_windows_chunk_fast(positions, starts, ends, values, out, track_idx, window_size):
    """Note: Now using int32 for positions/indices and float32 for values/output"""
    half = window_size // 2
    last_start_idx = 0
    last_end_idx = 0
    last_pos = -10000000

    _ = starts[0], ends[0], values[0]

    for i in np.arange(len(positions)):
        pos = positions[i]
        window_start = pos - half
        window_end = pos + half
        pos_diff = pos - last_pos

        relative_idx = np.searchsorted(ends[last_start_idx : last_start_idx + pos_diff + 1], window_start, side="right")
        idx_start = last_start_idx + relative_idx
        search_s = max(last_end_idx, idx_start)
        search_e = min(last_end_idx + pos_diff + 1, idx_start + window_size + 1)
        if search_s < search_e:
            relative_idx = np.searchsorted(starts[search_s:search_e], window_end)
            idx_end = search_s + relative_idx

            last_start_idx = idx_start
            last_end_idx = idx_end
            last_pos = pos

            for j in range(idx_start, idx_end):
                s_pos = starts[j] - window_start
                e_pos = ends[j] - window_start
                s_idx = np.maximum(0, s_pos)
                e_idx = np.minimum(window_size, e_pos)
                out[i, s_idx:e_idx] = values[j]
