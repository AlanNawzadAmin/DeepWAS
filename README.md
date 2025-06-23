# Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra



[Alan N. Amin*](https://alannawzadamin.github.io), [Andres Potapczynski*](https://www.andpotap.com), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). * equal contribution

[Paper](https://openreview.net/pdf?id=oOtdWiLb1e)

<p align="center">
  <img width="451" alt="concept" src="https://github.com/user-attachments/assets/35e4c0d7-4ae1-4cc3-aa31-98e53f6d1263" />
</p>

DeepWAS trains deep functionally informed priors on large public GWAS data. It does so efficiently by using an itereative algorithm to calculate the likelihood and its gradient.

## Processing the track data

### Downloading

Note these downloads may take up to several hours on a standard cpu and make up several terabytes of data.

#### ENCODE

Run ```python scripts/download_func_tracks.py```, ```python scripts/download_func_tracks_tfs.py```, and ```python scripts/download_func_tracks_chip_not_tf.py``` to download the data.
Afterwards, run ```python scripts/merge_eCLIP.py``` to merge the eCLIP tracks.

#### Phylogeny

Run ```python scripts/download_phylo_tracks.py```.

#### FANTOM

Run ```python scripts/download_fantom_tracks.py```.

#### dbNSFP

Visit ```https://www.dbnsfp.org/download``` to request to download dbNSFP.
Note the code assumes version 5.1 -- some features may be missing in earlier or later versions.
Move ```dbNSFP5.1a.zip``` into a folder called ```data/tracks/dbNSFP/```.
Then uncompress the files with

```
cd data/tracks/dbNSFP

unzip dbNSFP5.1a.zip
cd dbNSFP5.1a

for file in dbNSFP5.1a_variant.chr*.gz; do
    echo "Decompressing $file..."
    gunzip -v "$file"
done

cd ../../../..
python scripts/filter_dbNSFP.py
```

### Preprocessing

Run these scripts after downloading all the data above.
Note these preprocessing jobs can take up to several hours on a standard cpu.

#### Extracting BigWigs into .npy files

You need to run ```python make_hdf5s.py r t``` for all chromosomes ```r=0, ..., 21``` and your desired subset of features ```t```.
```t=0``` is ```["phylo", "big_encode", "fantom"]```, ```t=1``` is ```["phylo"]```, and ```t=2``` is ```["phylo", "fantom"]```.
These can all be run in parallel.

#### Getting summary statistics

To compute the mean and standard deviation of each track (so that we can standardize tracks during training), run ```python experiments/compute_track_stats.py```.

## Running semi-synthetic simulations
To run a semi-sythetic simulation you can run the following command:
```shell
py train.py data.name=simRandInit data.max_n_snps=1000 data.n_workers=3 train.n_epoch=10 architecture.model=enformer model.loss=wasp train.lr=0.0002 data.other_args.model=enrich
```
Above we are using an Enformer model to approximate a hard enrichment fuction.
