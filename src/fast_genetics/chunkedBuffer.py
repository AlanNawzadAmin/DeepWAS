from collections import OrderedDict

import numpy as np


class SingleIOChunkedBuffer:
    """Buffer for .npy files that loads data in chunks with exactly one I/O operation per chunk."""

    def __init__(self, file_path, chunk_size=5_000_000, max_cached_chunks=3):
        self.file_path = file_path
        self.chunk_size = int(chunk_size)
        self.max_cached_chunks = max_cached_chunks

        # Load array info without loading all data
        temp_mmap = np.load(file_path, mmap_mode="r")
        self.dtype = temp_mmap.dtype
        self.shape = temp_mmap.shape

        # Get file details
        self.header_size = temp_mmap.offset
        self.itemsize = temp_mmap.itemsize

        # Close the temporary mapping
        del temp_mmap

        # LRU cache for chunks
        self.chunk_cache = OrderedDict()

        # self.file = open(self.file_path, "rb")

    def __getitem__(self, idx):
        """Get data, loading chunks as needed."""
        # Handle field access for structured arrays
        if isinstance(idx, str):
            if idx not in self.dtype.names:
                raise ValueError(f"Field {idx} not found in array")
            return FieldView(self, idx)

        # Handle slice indices
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.shape[0]

            # Calculate which chunks we need
            start_chunk = start // self.chunk_size
            end_chunk = (stop - 1) // self.chunk_size

            # If all data is in one chunk, optimize the access
            if start_chunk == end_chunk:
                chunk = self._get_chunk(start_chunk)
                local_start = start - (start_chunk * self.chunk_size)
                local_end = stop - (start_chunk * self.chunk_size)
                return chunk[local_start:local_end]

            # Multiple chunks needed - load each with a single I/O operation
            chunks_to_concat = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                chunk = self._get_chunk(chunk_idx)
                chunk_start = chunk_idx * self.chunk_size

                local_start = max(0, start - chunk_start)
                local_end = min(len(chunk), stop - chunk_start)

                chunks_to_concat.append(chunk[local_start:local_end])

            return np.concatenate(chunks_to_concat)

        # Handle integer indices
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        chunk = self._get_chunk(chunk_idx)
        return chunk[local_idx]

    def _get_chunk(self, chunk_idx):
        """Get a chunk with a SINGLE I/O operation."""
        if chunk_idx in self.chunk_cache:
            # Cache hit - return from cache
            chunk = self.chunk_cache.pop(chunk_idx)
            self.chunk_cache[chunk_idx] = chunk
            return chunk

        # Calculate chunk range
        start_idx = int(chunk_idx) * self.chunk_size
        end_idx = min((chunk_idx + 1) * self.chunk_size, self.shape[0])
        chunk_size = end_idx - start_idx

        # Calculate file offsets
        data_offset = self.header_size + (start_idx * self.itemsize)
        bytes_to_read = chunk_size * self.itemsize

        # Read the chunk with a SINGLE I/O operation
        with open(self.file_path, "rb") as f:
            f.seek(data_offset)
            data_bytes = f.read(bytes_to_read)

        # Convert bytes to NumPy array
        chunk = np.frombuffer(data_bytes, dtype=self.dtype)

        # Manage cache
        if len(self.chunk_cache) >= self.max_cached_chunks:
            self.chunk_cache.popitem(last=False)  # Remove oldest

        self.chunk_cache[chunk_idx] = chunk
        return chunk

    # def close(self):
    #     self.file.close()

    # def __del__(self):
    #     self.close()


class FieldView:
    """View of a single field in a structured array."""

    def __init__(self, parent, field_name):
        self.parent = parent
        self.field_name = field_name

    def __getitem__(self, idx):
        data = self.parent[idx]
        return data[self.field_name]
