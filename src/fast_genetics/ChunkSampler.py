import random
import time
from collections import defaultdict

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


def split_chroms(chrom_to_indices, unique_chroms, n_workers):
    worker_chroms = [[] for _ in range(n_workers)]
    worker_indices = [[] for _ in range(n_workers)]
    lens = [0] * n_workers
    for chrom in unique_chroms:
        min_ind = np.argmin(lens)
        new_inds = chrom_to_indices[chrom]
        worker_chroms[min_ind].append(chrom)
        worker_indices[min_ind].extend(new_inds)
        lens[min_ind] += len(new_inds)
    return worker_chroms, worker_indices


def split_indices(indicies, n_workers):
    indicies = np.sort(indicies)
    ind_per_worker = int(np.ceil(len(indicies) / n_workers))
    worker_indices = [indicies[i * ind_per_worker : (i + 1) * ind_per_worker] for i in range(n_workers)]
    return worker_indices


class ChunkedDataset(IterableDataset):
    """Dataset that gives each worker sequential chunks to process.

    Each worker:
    1. Gets its own subset of chunks based on worker_id
    2. Processes each chunk sequentially for cache efficiency
    3. Moves to another random chunk when done
    """

    def __init__(self, dataset, chunk_size=10, shuffle=True, seed=None, locs=[], collate_fn=None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.size = len(dataset)
        self.num_chunks = (self.size + chunk_size - 1) // chunk_size
        self.indicies = np.arange(self.size)
        self.collator = collate_fn
        self.print = False

        # for chrom split
        self.locs = locs
        self.chroms = [loc[0] for loc in self.locs]
        self.starts = [loc[1] for loc in self.locs]
        self.ends = [loc[2] for loc in self.locs]
        self.chrom_to_indices = defaultdict(list)
        for idx, chrom in enumerate(self.chroms):
            self.chrom_to_indices[chrom].append(idx)

    def __iter__(self):
        # Get worker info
        worker_info = get_worker_info()

        # If no workers, process everything in order
        if worker_info is None:
            pass
        else:
            # Each worker handles a different set of indices
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # unique_chroms = sorted(self.chrom_to_indices.keys())
            # worker_chroms, worker_indices = split_chroms(self.chrom_to_indices, unique_chroms, num_workers)
            worker_indices = split_indices(self.indicies, num_workers)
            worker_indices = worker_indices[worker_id]

            # Set worker-specific seed if provided
            if self.seed is not None:
                random.seed(self.seed + worker_id)

        # Break worker's section into chunks
        worker_indices.sort()  # Sort to keep sequential access within chromosomes

        chunks = []
        for i in range(0, len(worker_indices), self.chunk_size):
            chunk_indices = worker_indices[i : i + self.chunk_size]
            chunks.append(chunk_indices)

        # Shuffle the order of chunks if requested
        if self.shuffle:
            random.shuffle(chunks)

        # Process each chunk sequentially
        mean_time = 0
        n_steps = 0.01
        for chunk_inds in chunks:
            for idx in chunk_inds:
                if self.print:
                    print("Starting worker loading:", worker_id, idx)
                    tik = time.time()
                if self.collator is None:
                    dat = self.dataset[idx]
                else:
                    dat = self.collator([self.dataset[idx]])
                if self.print:
                    mem = __import__("psutil").Process().memory_info().rss / 1024 / 1024 // 1024
                    time0 = time.time() - tik
                    mean_time, n_steps = (mean_time * n_steps + time0) / (n_steps + 1), n_steps + 1
                    print(
                        "worker loading:",
                        worker_id,
                        idx,
                        int(time0),
                        "s,",
                        int(mean_time),
                        "s,",
                        mem,
                        "GB",
                    )
                yield dat

    def __len__(self):
        return self.size
