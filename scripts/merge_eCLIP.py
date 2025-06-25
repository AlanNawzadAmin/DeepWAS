#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import pyBigWig


@dataclass
class Interval:
    start: int  # Start position of the interval
    end: int  # End position of the interval
    value: float  # Value associated with this interval


def merge_intervals(intervals: List[Interval]) -> Iterator[Tuple[int, int, float]]:
    """
    Merge overlapping intervals and compute the sum of values.
    """
    if not intervals:
        return

    # Create change points
    changes = []
    for interval in intervals:
        changes.append((interval.start, 1, interval.value))  # start point
        changes.append((interval.end, -1, interval.value))  # end point

    # Sort changes by position
    changes.sort(key=lambda x: (x[0], -x[1]))

    current_sum = 0
    active_values = 0
    last_pos = changes[0][0]

    for pos, change_type, value in changes:
        # Output interval if we have moved
        if pos > last_pos and active_values > 0:
            yield (last_pos, pos, current_sum)

        # Update running sum
        if change_type == 1:  # interval start
            current_sum += value
            active_values += 1
        else:  # interval end
            current_sum -= value
            active_values -= 1

        last_pos = pos


def sum_bigwigs(input_files: List[str], output_file: str, chromosomes: List[str] = None):
    """
    Sum multiple bigWig files.
    """
    # Open all input files
    bw_files = [pyBigWig.open(f) for f in input_files]

    # Get chromosome information from first file
    first_bw = bw_files[0]
    if chromosomes:
        chrom_sizes = [(chrom, first_bw.chroms()[chrom]) for chrom in chromosomes if chrom in first_bw.chroms()]
    else:
        chrom_sizes = list(first_bw.chroms().items())

    # Create output file
    output_bw = pyBigWig.open(output_file, "w")
    output_bw.addHeader(list(chrom_sizes))

    from tqdm import tqdm

    # Process each chromosome
    for chrom, size in tqdm(chrom_sizes, desc="Processing chromosomes"):
        try:
            all_intervals = []
            pbar = tqdm(bw_files, desc=f"{chrom}", leave=False)

            # Collect intervals from all files...

            # Merge intervals
            if all_intervals:
                merged = list(merge_intervals(all_intervals))

                if merged:
                    # Write to output file
                    output_bw.addEntries(
                        [chrom] * len(merged),
                        [m[0] for m in merged],  # starts
                        ends=[m[1] for m in merged],
                        values=[m[2] for m in merged],
                    )
        except Exception as e:
            print(f"\nError processing {chrom}: {str(e)}")
            print("Continuing with next chromosome...")

        # Collect intervals from all files
        for bw in pbar:
            if not bw.chroms().get(chrom):
                pbar.update(1)
                continue

            try:
                intervals = bw.intervals(chrom)
                if intervals:
                    all_intervals.extend(Interval(start, end, value) for start, end, value in intervals)
            except RuntimeError as e:
                print(f"\nWarning: Error reading {chrom} from {bw.path}: {str(e)}")
                continue
            except Exception as e:
                print(f"\nError: Unexpected error reading {chrom} from {bw.path}: {str(e)}")
                continue

        # Merge intervals
        merged = list(merge_intervals(all_intervals))

        if merged:
            # Write to output file
            output_bw.addEntries(
                [chrom] * len(merged),
                [m[0] for m in merged],  # starts
                ends=[m[1] for m in merged],
                values=[m[2] for m in merged],
            )

    # Cleanup
    for bw in bw_files:
        bw.close()
    output_bw.close()

from pathlib import Path

def main(output_dir):
    # Define base directory
    base_dir = output_dir

    # Find all eCLIP directories
    eclip_dirs = [d for d in Path(base_dir).glob("eCLIP*") if d.is_dir()]

    # if not eclip_dirs:
    #     print(f"No directories starting with 'eCLIP' found in {base_dir}")
    #     return

    # Collect all bigWig files from these directories
    input_files = []
    for directory in eclip_dirs:
        bigwigs = list(directory.glob("*.bigWig"))
        input_files.extend(str(bw) for bw in bigwigs)

    # if not input_files:
    #     print(f"No bigWig files found in eCLIP directories")
    #     return

    print(f"Found {len(input_files)} files to process:")
    for f in input_files:
        print(f"  {Path(f).name}")

    # Define autosomes
    autosomes = [f"chr{i}" for i in range(1, 23)]

    # Set output file
    output_file = os.path.join(base_dir, "eclip_sum.bigWig")

    # Run summation
    print(f"\nProcessing autosomes: {', '.join(autosomes)}")
    sum_bigwigs(input_files, output_file, autosomes)


if __name__ == "__main__":
    output_dir = "data/tracks/encode"
    main(output_dir)
