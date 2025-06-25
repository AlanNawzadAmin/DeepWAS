import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
import requests


class EncodeDataCollector:
    def __init__(self, small_set, out_dir):
        self.experiments_by_type = defaultdict(list)
        self.unique_experiments = defaultdict(dict)
        self.small_set = small_set
        self.out_dir = out_dir

    def get_encode_files(self, assay_type: str = None, target: str = None) -> List[dict]:
        """Get list of ENCODE experiments meeting criteria"""
        base_url = (
            # "https://www.encodeproject.org/search/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&control_type%21=%2A&perturbed=false&status=released&audit.ERROR.category%21=extremely+low+read+depth&audit.ERROR.category%21=missing+control+alignments&audit.ERROR.category%21=control+extremely+low+read+depth&audit.ERROR.category%21=not+compliant+biosample+characterization&audit.ERROR.category%21=extremely+low+read+length&audit.NOT_COMPLIANT.category%21=insufficient+read+depth&audit.NOT_COMPLIANT.category%21=partially+characterized+antibody&audit.NOT_COMPLIANT.category%21=severe+bottlenecking&audit.NOT_COMPLIANT.category%21=poor+library+complexity&audit.NOT_COMPLIANT.category%21=insufficient+read+length&audit.NOT_COMPLIANT.category%21=insufficient+replicate+concordance&audit.NOT_COMPLIANT.category%21=unreplicated+experiment&audit.NOT_COMPLIANT.category%21=control+insufficient+read+depth&audit.NOT_COMPLIANT.category%21=missing+input+control&audit.WARNING.category%21=mild+to+moderate+bottlenecking&audit.WARNING.category%21=missing+genetic+modification+reagents&audit.WARNING.category%21=low+read+depth&audit.WARNING.category%21=borderline+replicate+concordance&audit.WARNING.category%21=moderate+library+complexity&audit.WARNING.category%21=antibody+characterized+with+exemption&audit.WARNING.category%21=low+read+length&audit.WARNING.category%21=missing+genetic+modification+characterization&audit.WARNING.category%21=missing+controlled_by&audit.WARNING.category%21=missing+biosample+characterization&audit.WARNING.category%21=inconsistent+platforms&audit.WARNING.category%21=inconsistent+control+read+length&audit.WARNING.category%21=improper+control_type+of+control+experiment&audit.WARNING.category%21=missing+compliant+biosample+characterization&audit.WARNING.category%21=control+low+read+depth&audit.WARNING.category%21=inconsistent+control+run_type&audit.WARNING.category%21=mixed+read+lengths&audit.WARNING.category%21=mixed+run+types&audit.WARNING.category%21=matching+md5+sums&assay_title=TF+ChIP-seq&target.investigated_as=transcription+factor"
            "https://www.encodeproject.org/search/"
            "?type=Experiment"
            "&control_type!=*"
            "&status=released"
            "&perturbed=false"
            "&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens"
            "&perturbed=true"
            "&audit.ERROR.category!=extremely+low+read+depth"
            "&audit.ERROR.category!=file+validation+error"
            "&audit.ERROR.category!=missing+antibody"
            "&audit.ERROR.category!=extremely+low+spot+score"
            "&audit.ERROR.category!=missing+control+alignments"
            "&audit.ERROR.category!=control+extremely+low+read+depth"
            "&audit.ERROR.category!=missing+lambda+C+conversion+rate"
            "&audit.ERROR.category!=extremely+low+coverage"
            "&audit.ERROR.category!=not+compliant+biosample+characterization"
            "&audit.ERROR.category!=extremely+low+read+length"
            "&audit.ERROR.category!=inconsistent+ontology+term"
            "&audit.ERROR.category!=missing+footprints"
            "&audit.NOT_COMPLIANT.category!=insufficient+read+depth"
            "&audit.NOT_COMPLIANT.category!=control+insufficient+read+depth"
            "&audit.NOT_COMPLIANT.category!=unreplicated+experiment"
            "&audit.NOT_COMPLIANT.category!=insufficient+read+length"
            "&audit.NOT_COMPLIANT.category!=poor+library+complexity"
            "&audit.NOT_COMPLIANT.category!=severe+bottlenecking"
            "&audit.NOT_COMPLIANT.category!=partially+characterized+antibody"
            "&audit.NOT_COMPLIANT.category!=missing+spikeins"
            "&audit.NOT_COMPLIANT.category!=low+FRiP+score"
            "&audit.NOT_COMPLIANT.category!=insufficient+replicate+concordance"
            "&audit.NOT_COMPLIANT.category!=insufficient+coverage"
            "&audit.NOT_COMPLIANT.category!=insufficient+number+of+reproducible+peaks"
            "&audit.NOT_COMPLIANT.category!=low+non-redundant+PET"
            "&audit.NOT_COMPLIANT.category!=extremely+low+pct_unique_long_range_greater_than_20kb"
            "&audit.NOT_COMPLIANT.category!=missing+documents"
            "&audit.NOT_COMPLIANT.category!=missing+possible_controls"
            "&audit.NOT_COMPLIANT.category!=extremely+low+pct_ligation_motif_present"
            "&audit.NOT_COMPLIANT.category!=uncharacterized+antibody"
            "&audit.NOT_COMPLIANT.category!=insufficient+sequencing+depth"
            "&audit.NOT_COMPLIANT.category!=missing+input+control"
            "&audit.NOT_COMPLIANT.category!=low+TSS+enrichment"
            "&audit.WARNING.category!=low+read+depth"
            "&audit.WARNING.category!=missing+controlled_by"
            "&assembly=GRCh38"
            "&files.file_type=bigWig"
            "&audit.WARNING.category!=mild+to+moderate+bottlenecking"
            "&audit.WARNING.category!=borderline+replicate+concordance"
            "&audit.WARNING.category!=low+spot+score"
            "&audit.WARNING.category!=inconsistent+control+read+length"
            "&audit.WARNING.category!=control+low+read+depth"
            "&audit.WARNING.category!=matching+md5+sums"
            "&audit.WARNING.category!=improper+control_type+of+control+experiment"
            "&audit.WARNING.category!=inconsistent+control+run_type"
            "&audit.WARNING.category!=low+replicate+concordance"
            "&audit.WARNING.category!=low+sequenced_read_pairs"
            "&audit.WARNING.category!=missing+analysis_step_run"
            "&audit.WARNING.category!=borderline+number+of+aligned+reads"
            "&audit.WARNING.category!=low+total_unique_reads"
            "&audit.WARNING.category!=missing+control_type+of+control_experiment"
            "&audit.WARNING.category!=unexpected+target+of+control+experiment"
            "&audit.WARNING.category!=inconsistent+assembly"
            "&audit.WARNING.category!=moderate+number+of+reproducible+peaks"
            "&audit.WARNING.category!=low+lambda+C+conversion+rate"
            "&audit.WARNING.category!=low+pct_ligation_motif_present"
            "&audit.WARNING.category!=missing+lambda+C+conversion+rate"
            "&audit.WARNING.category!=low+coverage"
            "&audit.WARNING.category!=low+intra/inter-chr+PET+ratio"
            "&audit.WARNING.category!=high+pct_unique_total_duplicates"
            "&audit.WARNING.category!=low+total+read+pairs"
            "&audit.WARNING.category!=low+pct_unique_long_range_greater_than_20kb"
            "&audit.WARNING.category!=borderline+microRNAs+expressed"
            "&audit.WARNING.category!=missing+footprints"
            "&audit.WARNING.category!=missing+spikeins"
            "&audit.WARNING.category!=mixed+run+types"
            "&audit.WARNING.category!=inconsistent+platforms"
        )

        if assay_type:
            # Convert common assay names to ENCODE terms
            base_url += f"&assay_title={assay_type}"

        if target:
            base_url += f"&target.label={target}"

        base_url += "&format=json&limit=all"

        try:
            response = requests.get(base_url, headers={"Accept": "application/json"})
            response.raise_for_status()
            return response.json()["@graph"]
        except Exception as e:
            print(f"Error getting {assay_type} data: {e}")
            print(f"URL was: {base_url}")
            return []

    def get_experiment_quality_score(self, exp: dict) -> float:
        """Calculate a quality score for an experiment based on metrics"""
        score = 0.0

        # Prefer newer experiments
        date = exp.get("date_released", "2000-01-01")
        score += pd.to_datetime(date).timestamp() / 1e10

        # Prefer higher read depths
        read_depth = exp.get("read_depth", 0)
        score += min(read_depth / 1e8, 1.0)  # Cap influence of read depth

        # Prefer experiments with replicates
        if exp.get("replication_type") == "isogenic":
            score += 0.5

        # Penalize experiments with warnings
        if exp.get("audit"):
            score -= len(exp["audit"]) * 0.1

        return score

    def get_experiment_key(self, exp: dict) -> str:
        """Generate a unique key for experiment type"""
        assay = exp["assay_title"]
        biosample = exp.get("biosample_summary", "unknown") if not self.small_set else ""
        target = exp.get("target", {}).get("label", "none")

        return f"{assay}_{target}_{biosample}"

    def collect_data(self):
        """Collect and organize all experiments"""

        for assay in assay_types:
            print(f"\nFetching {assay} experiments...")
            experiments = self.get_encode_files(assay)
            self.experiments_by_type[assay] = experiments

            temp_unique_experiments = {}
            # Process each experiment
            for exp in experiments:
                key = self.get_experiment_key(exp)
                quality_score = self.get_experiment_quality_score(exp)

                # Keep only the highest quality experiment for each key
                if key in temp_unique_experiments and key not in self.unique_experiments[assay]:
                    self.unique_experiments[assay][key] = exp
                if key not in temp_unique_experiments or quality_score > self.get_experiment_quality_score(
                    temp_unique_experiments[key]
                ):
                    temp_unique_experiments[key] = exp
            self.unique_experiments[assay] = temp_unique_experiments  # if not filtering
            print(len(experiments), self.unique_experiments[assay])

    def print_summary(self):
        """Print summary of unique experiments found"""
        print("\n=== ENCODE Dataset Summary ===")

        for assay_type, experiments in self.unique_experiments.items():
            print(f"\n{assay_type}:")
            print(f"Total unique experiments: {len(experiments)}")

            # Count by biosample
            biosamples = defaultdict(int)
            for exp in experiments.values():
                biosample = exp.get("biosample_summary", "unknown")
                biosamples[biosample] += 1
            print(f"Unique biosamples: {len(biosamples)}")

            # For ChIP-seq, count unique targets
            if assay_type in ["TF ChIP-seq", "Histone ChIP-seq", "eCLIP"]:
                targets = defaultdict(int)
                for exp in experiments.values():
                    target = exp.get("target", {}).get("label", "unknown")
                    targets[target] += 1
                print(f"Unique targets: {len(targets)}")
                print("\nTop targets:")
                for target, count in sorted(targets.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {target}: {count} experiments")

    def download_unique_experiments(self):
        """Download the unique set of experiments"""
        os.makedirs(self.out_dir, exist_ok=True)
        os.chdir(self.out_dir)

        # Create metadata file
        metadata = {"assay_counts": {k: len(v) for k, v in self.unique_experiments.items()}, "experiments": []}

        # Download each unique experiment
        with ThreadPoolExecutor(max_workers=15) as executor:
            for assay_type, experiments in self.unique_experiments.items():
                for exp in experiments.values():
                    metadata["experiments"].append(
                        {
                            "accession": exp["accession"],
                            "assay": assay_type,
                            "biosample": exp.get("biosample_summary", "unknown"),
                            "target": exp.get("target", {}).get("label", "none"),
                            "date": exp.get("date_released", "unknown"),
                            "quality_score": self.get_experiment_quality_score(exp),
                        }
                    )
                    executor.submit(self.download_bigwig, exp, assay_type)

        # Save metadata
        with open("metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def download_bigwig(self, exp: dict, assay_type: str, download_one=True):
        """Download bigWig file for an experiment with existence checking"""
        try:
            # Get experiment accession
            exp_name = f"{assay_type.replace(' ', '_')}_{exp.get('target', {}).get('label', 'none')}_{exp['accession']}"

            # Create directory structure
            exp_dir = os.path.join(self.out_dir, exp_name)

            # Check if experiment is already fully downloaded
            if os.path.exists(exp_dir):
                metadata_path = os.path.join(exp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        existing_metadata = json.load(f)
                    # If metadata exists and matches, skip this experiment
                    if existing_metadata["accession"] == exp["accession"]:
                        print(f"Experiment {exp_name} already downloaded, skipping...")
                        return

            os.makedirs(exp_dir, exist_ok=True)

            # Save experiment metadata
            metadata = {
                "accession": exp["accession"],
                "assay": exp["assay_title"],
                "biosample": exp.get("biosample_term_name", "unknown"),
                "target": exp.get("target", {}).get("label", "none"),
                "date": exp.get("date_released", "unknown"),
                "quality_metrics": exp.get("quality_metrics", {}),
                "lab": exp.get("lab", {}).get("title", "unknown"),
            }

            metadata_path = os.path.join(exp_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            # Get and download bigWig files
            files_url = f"https://www.encodeproject.org{exp['@id']}/?format=json"
            files_response = requests.get(files_url)
            files = files_response.json()["files"]

            bigwigs = [f for f in files if f.get("file_type") == "bigWig" and f.get("assembly") == "GRCh38"]
            if download_one:
                bigwigs_fold = [
                    bw for bw in bigwigs if "fold" in bw.get("output_type") or "peaks" in bw.get("output_type")
                ]
                if len(bigwigs_fold) == 0:
                    print("Could not find fold change for", exp_name)
                    bigwigs = [bw for bw in bigwigs if "p-value" not in bw.get("output_type")]
                else:
                    bigwigs = bigwigs_fold

            for bw in bigwigs if not download_one else [bigwigs[0]]:
                output_file = os.path.join(exp_dir, f"{bw['accession']}.bigWig")
                metadata_file = os.path.join(exp_dir, f"{bw['accession']}.metadata.json")

                # Check if both file and its metadata exist
                if os.path.exists(output_file) and os.path.exists(metadata_file):
                    print(f"File {bw['accession']}.bigWig already exists, skipping...")
                    continue

                url = f"https://www.encodeproject.org{bw['href']}"

                print(f"Downloading {output_file}...")
                cmd = f"wget -O {output_file} {url}"
                subprocess.run(cmd, shell=True)

                # Save file-specific metadata
                file_metadata = {
                    "accession": bw["accession"],
                    "output_type": bw.get("output_type"),  # e.g., "signal p-value", "fold change"
                    "biological_replicates": bw.get("biological_replicates", []),  # which replicates
                    "technical_replicates": bw.get("technical_replicates", []),
                    "file_size": bw.get("file_size"),
                    "date_created": bw.get("date_created"),
                    "derived_from": bw.get("derived_from", []),  # upstream files
                    "file_format_type": bw.get("file_format_type"),
                    "quality_metrics": bw.get("quality_metrics", []),
                    "status": bw.get("status"),
                    "step_run": bw.get("step_run"),  # processing pipeline info
                }

                with open(metadata_file, "w") as out_f:
                    json.dump(file_metadata, out_f, indent=2)

        except Exception as e:
            print(f"Error downloading {exp.get('accession')}: {e}")
            # If something goes wrong, try to clean up partial downloads
            if os.path.exists(exp_dir) and not os.listdir(exp_dir):
                os.rmdir(exp_dir)


def main(small_set):
    collector = EncodeDataCollector(small_set)
    collector.collect_data()
    collector.print_summary()

    print("\nReady to download unique experiments.")
    # response = input("Proceed with download? (y/n): ")
    # if response.lower() == 'y':
    collector.download_unique_experiments()


assay_types = [
    # "TF ChIP-seq",
    # "Histone ChIP-seq",  # histone modifications
    "eCLIP",  # RNABP"total RNA-seq", "polyA plus RNA-seq", # mRNA
    # "polyA minus RNA-seq",
    # "small RNA-seq",
    # "microRNA-seq",  # ncRNA
    # "ChIA-PET",  # high res Hi-C
    # "WGBS",  # dna methylation
    # "DNase-seq",
    # "ATAC-seq",  # accessibility
    # "PRO-cap",
    # "PRO-seq",  # TSS
    # "Bru-seq",
    # "BruChase-seq",
    # "RAMPAGE",
    # "PAS-seq",  # 3 and 5' mapping
]

if __name__ == "__main__":
    output_dir = "data/tracks/encode"
    small_set = True  # ignore biosample!
    main(small_set, output_dir)


# few: "scRNA-seq" (4), "GM DNase-seq" (10), "BruUV-seq"(16),
# multiplex chip (226): "Mint-ChIP-seq"
# Hi-C: "in situ Hi-C", "intact Hi-C",
# RNA-seq perturbations (~800): "shRNA RNA-seq", "CRISPR RNA-seq",
