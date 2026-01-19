# Mugi Profiling
A profiling script which compares perplexity and value distributions of multiple transformer workloads across different nonlinear implementations.

## Installation
Make sure you have [Anaconda](https://www.anaconda.com/) installed before the steps below.

### AE setup
1. ```git clone``` [this repo](https://github.com/UnaryLab/mugi_profiling) and ```cd mugi_profiling``` to the repo dir.
2. ```git fetch --all` to retrieve branches.
3. ```git checkout -b asplos_2026_ae origin/asplos_2026_ae``` to switch to the AE branch
4. ```conda env create -f environment.yaml```
   - The ```name: mugi_profiling``` in ```evironment.yaml``` can be updated to a preferred one.
5. ```conda activate mugi_profiling```
6. ```bash mugi_profiling.sh``` to run the simulation workflow.
7. Output figures can be found in ```figures/output/```.

## Zenodo
A zenodo submission exists at https://zenodo.org/records/18063514

## Citation
If Mugi has been useful in your own research, please cite us using the following bibtex citation:

```
@inproceedings{price2026asplos,
  title     = {Mugi: Value Level Parallelism For Efficient LLMs},
  author    = {Daniel Price and Prabhu Vellaisamy and John Paul Shen and Di Wu},
  booktitle = {International Conference on Architectural Support for Programming Languages and Operating Systems},
  year      = {2026}
}
```
