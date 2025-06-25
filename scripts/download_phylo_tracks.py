import os
import time
from pathlib import Path
from threading import Lock

import numpy as np
import requests
import tqdm

# Create a lock for progress bar updates to prevent garbled output
print_lock = Lock()


def download_bigwig(url: str, output_dir: str, filename: str = None) -> str:
    """
    Download a BigWig file from a URL using smaller range requests with robust error handling.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]
        if not filename.endswith(".bw") and not filename.endswith(".bigWig"):
            filename += ".bw"

    output_path = os.path.join(output_dir, filename)

    # If file exists and is complete, skip it
    if os.path.exists(output_path):
        response = requests.head(url)
        expected_size = int(response.headers.get("content-length", 0))
        if os.path.getsize(output_path) == expected_size:
            print(f"File already exists and is complete: {output_path}")
            return output_path

    # Use 1GB chunks instead of 2GB to stay well under server limits
    chunk_size = 1024 * 1024 * 1024  # 1GB in bytes

    # try:
    response = requests.head(url)
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            bytes_downloaded = 0

            while bytes_downloaded < total_size:
                # Calculate chunk range
                start = bytes_downloaded
                end = min(start + chunk_size - 1, total_size - 1)

                # Try to download this chunk with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        headers = {"Range": f"bytes={start}-{end}"}
                        response = requests.get(url, headers=headers, stream=True)
                        response.raise_for_status()

                        # Download chunk data
                        chunk_size_downloaded = 0
                        for data in response.iter_content(chunk_size=8192):
                            if data:
                                chunk_size_downloaded += len(data)
                                f.write(data)
                                pbar.update(len(data))

                        # Verify chunk size
                        expected_chunk_size = end - start + 1
                        if chunk_size_downloaded != expected_chunk_size:
                            if attempt == max_retries - 1:
                                raise Exception(
                                    f"Chunk download incomplete after {max_retries} attempts. "
                                    f"Expected {expected_chunk_size} bytes, got {chunk_size_downloaded}"
                                )
                            continue

                        # Chunk downloaded successfully
                        bytes_downloaded += chunk_size_downloaded
                        break

                    except (requests.exceptions.RequestException, Exception) as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Failed to download chunk after {max_retries} attempts: {e}")
                        time.sleep(2**attempt)  # Exponential backoff

            # Verify final file size with tolerance
            final_size = os.path.getsize(output_path)
            size_difference = abs(final_size - total_size)
            size_difference_percent = (size_difference / total_size) * 100

            # Allow up to 0.5% difference in file size
            if size_difference_percent > 0.5:
                raise Exception(
                    f"File size mismatch too large. Expected {total_size}, "
                    f"got {final_size} (difference of {size_difference_percent:.2f}%)"
                )
            elif size_difference > 0:
                print(
                    f"Note: Final file size differs by {size_difference_percent:.2f}% "
                    f"({size_difference} bytes). This is within acceptable range."
                )

    return output_path

    # except Exception as e:
    #     print(f"Error downloading {filename}: {str(e)}")
    #     # Don't delete the file if it's just a small size mismatch
    #     if "File size mismatch too large" not in str(e):
    #         if os.path.exists(output_path):
    #             os.remove(output_path)
    #     return None


def download_conservation_scores(output_dir: str):
    """Download common conservation score tracks.
    These are derived from alignments which themselves can be lifted from
    http://hgdownload.soe.ucsc.edu/goldenPath/hg38/.
    Can do these later!"""
    conservation_urls = {
        # Most recent comprehensive alignments
        "phyloP447way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw",
        "phyloP470way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/hg38.phyloP470way.bw",
        "phastCons470way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw",
        # Historical alignments with different evolutionary depths
        "phyloP100way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
        "phyloP30way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP30way/hg38.phyloP30way.bw",
        "phyloP20way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP20way/hg38.phyloP20way.bw",
        "phyloP17way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP17way/hg38.phyloP17way.bw",
        "phyloP7way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP7way/hg38.phyloP7way.bw",
        "phyloP4way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP4way/hg38.phyloP4way.bw",
        # PhastCons scores at matching depths
        "phastCons100way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
        "phastCons30way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons30way/hg38.phastCons30way.bw",
        "phastCons20way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons20way/hg38.phastCons20way.bw",
        "phastCons17way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons17way/hg38.phastCons17way.bw",
        "phastCons7way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons7way/hg38.phastCons7way.bw",
        "phastCons4way": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons4way/hg38.phastCons4way.bw",
    }

    cons_dir = output_dir
    paths = {}
    items = list(conservation_urls.items())
    np.random.shuffle(items)

    for name, url in items:
        print(f"Downloading {name} conservation scores...")
        path = download_bigwig(url, cons_dir, f"{name}.bw")
        paths[name] = path

    return paths


# Main download function
def download_all_features(base_dir: str):
    """Download all feature tracks and organize them"""
    print("Starting download of genomic features...")

    # Create base directory
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Download conservation scores
    print("\nDownloading conservation scores...")
    conservation_paths = download_conservation_scores(base_dir)

    print("\nAll downloads complete! Data organized in:", base_dir)
    return {"conservation": conservation_paths}


# Usage example:
if __name__ == "__main__":
    output_dir = "data/tracks/phylo"
    # Download everything to a 'genomic_features' directory
    paths = download_all_features(output_dir)
