# ApexGPT

This repository is the lightweight public code-and-results release for the ApexGPT manuscript.

Pretrained ApexGPT checkpoints are distributed separately on Zenodo:

- https://zenodo.org/records/19597375

This GitLab repository keeps the public codebase small and focuses on:

- ApexGPT training and generation scripts in `code/apexgpt/`
- manuscript-linked processed training data in `data/training/`
- manuscript-linked result tables in `data/results/`
- tokenizer resources in `code/apexgpt/checkpoints/` and `resources/progen2_tokenizer/`
- a verified small generation smoke test in `data/demo/smoke_generation/`

## Repository layout

- `code/apexgpt/`
  ApexGPT training and generation code
- `data/training/`
  processed AMP and MDH training/test sequence files
- `data/results/`
  manuscript-linked result tables retained from the source project
- `data/demo/smoke_generation/`
  expected CSV outputs from the verified small generation use case
- `env/environment.yml`
  pinned Conda environment exported from the manuscript code directory

## Installation

Create the manuscript environment:

```sh
conda env create -f env/environment.yml
conda activate Bert
```

If you want to retrain from the public ProGen2 base model, download it with:

```sh
bash scripts/download_progen2_large.sh
```

## Pretrained models

Download the pretrained ApexGPT checkpoints from Zenodo and extract them anywhere on your machine:

- https://zenodo.org/records/19597375

For generation, pass the extracted checkpoint directory explicitly with `--checkpoint-dir`.

Example checkpoint directories after extraction might look like:

- `/path/to/ApexGPT_EMA_lambda10_amp`
- `/path/to/ApexGPT_EMA_lambda10_mdh`

The scripts now fail with a clear message if no checkpoint directory is available locally.

## Small generation use case

The repository includes a verified small-sample generation smoke test for both AMP and MDH. These commands are intended for code validation and review, not for full-scale manuscript generation runs.

Run the commands from `code/apexgpt/`.

### AMP smoke test

```sh
python generated_AMP_EMA.py \
  --device cpu \
  --checkpoint-dir /path/to/ApexGPT_EMA_lambda10_amp \
  --max-length 20 \
  --smoke-test true \
  --smoke-draws-primary 2 \
  --smoke-draws-combined 1 \
  --smoke-max-contexts 1 \
  --output-dir ../../data/demo/smoke_generation/amp_cpu
```

### MDH smoke test

```sh
python generated_MDH_EMA.py \
  --device cpu \
  --checkpoint-dir /path/to/ApexGPT_EMA_lambda10_mdh \
  --max-length 40 \
  --smoke-test true \
  --smoke-draws-primary 2 \
  --smoke-draws-combined 1 \
  --smoke-max-contexts 1 \
  --output-dir ../../data/demo/smoke_generation/mdh_cpu
```

Expected output files are already included in:

- `data/demo/smoke_generation/amp_cpu/`
- `data/demo/smoke_generation/mdh_cpu/`

These smoke tests write:

- `*_generated_RL_statistic_p_t_1_aa.csv`
- `*_generated_RL_statistic_p_t_2_aa.csv`
- `*_generated_RL_statistic_p_t_1_2_aa.csv`

## Full generation

For larger runs, use the same scripts without `--smoke-test true` and choose a CUDA device if available:

```sh
python generated_AMP_EMA.py --device cuda:0 --checkpoint-dir /path/to/ApexGPT_EMA_lambda10_amp
python generated_MDH_EMA.py --device cuda:0 --checkpoint-dir /path/to/ApexGPT_EMA_lambda10_mdh
```

Note that the original scripts generate many more sequences than the smoke test and are intended for full inference workflows.

## Training on your own data

Prepare a plain-text file with one sequence per line, following the same format as:

- `code/apexgpt/data/AMP_train.txt`
- `code/apexgpt/data/MDH_train.txt`
- `code/apexgpt/data/MDH_test.txt`

Then edit a copied config file in `code/apexgpt/`:

- `config_AMP_EMA.json`
- `config_MDH_EMA.json`
- `config_AMP.json`
- `config_MDH.json`

Update at least:

- `train_file`
- `validation_file` if applicable
- `max_seq_length`
- `output_dir`

Example training commands:

```sh
python main_AMP_EMA.py config_AMP_EMA.json
python main_MDH_EMA.py config_MDH_EMA.json
```

## Notes

- This repository intentionally does not bundle the large pretrained checkpoints
- The public pretrained model release is hosted on Zenodo rather than in GitLab
- The included smoke-test outputs are meant to provide a lightweight validation target for reviewers and users

## Reproducing the benchmarking results

See [REPRODUCING.md](REPRODUCING.md) for a step-by-step walkthrough from a
fresh clone to every metric reported in Fig. 1d and Supplementary Tables 1
and 2.
