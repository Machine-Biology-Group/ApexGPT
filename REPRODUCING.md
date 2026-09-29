# Reproducing the benchmarking results

This walks from a fresh clone to every number reported in Fig. 1d and
Supplementary Tables 1 and 2 of *Foundation model-guided latent spaces enable
functional protein design*.

Two artefacts are involved and they hold different things:

| | this repository | the archived snapshot on Zenodo |
|---|---|---|
| training and generation code | yes | yes (copy) |
| environment specification | yes | yes (plus resolved tool versions) |
| model configuration files | yes | yes (copy) |
| per-sequence predictor output behind Fig. 1d | yes | yes |
| **evaluation scripts** that turn sequences into table cells | no | **yes**, in `05_code/` |
| generated sequence files for every arm | no | **yes**, in `01_generated_sequences/` |
| reference sets (training FASTAs, Pfam HMMs) | no | **yes**, in `02_reference_sets/` |
| derived values (alignments, entropies, similarities, sweeps) | no | **yes**, in `04_derived_values/` |
| figure source data (t-SNE, effect sizes) | no | **yes**, in `06_figure_source_data/` |

Trained model weights are a separate Zenodo record:
[10.5281/zenodo.19597375](https://doi.org/10.5281/zenodo.19597375) (`models.zip`, 20.3 GB, CC-BY).

> **Data and analysis code:**
> [10.5281/zenodo.23045974](https://doi.org/10.5281/zenodo.23045974).
> Download `ApexGPT_data_and_code.zip` and unpack it; that is `SNAPSHOT/`
> throughout. Every step below can also be run against files you regenerate
> yourself, and the commands are written so that you can.

Throughout, `REPO/` is this clone and `SNAPSHOT/` is the unpacked archive.

---

## 0. Which numbers you want

There are three kinds, and they need different amounts of work:

1. **Recompute a table cell from the deposited sequences.** Minutes to hours.
   No GPU. This is the cheapest check and it verifies our arithmetic.
2. **Regenerate the sequences from the deposited checkpoints, then recompute.**
   Hours on one GPU. This verifies the generation settings.
3. **Retrain from scratch, then generate, then recompute.** Days on several
   GPUs. This verifies the training itself.

Most readers want (1). Section 2 covers it column by column. Sections 3 and 4
cover (2) and (3).

---

## 1. Environment

```bash
git clone https://gitlab.com/machine-biology-group-public/apexgpt.git
cd apexgpt
conda env create -f env/environment.yml
conda activate apex_env        # note: use ';' after activate, not '&&' --
                               # conda emits a libmamba warning and returns
                               # non-zero even on success
```

The evaluation steps additionally need four external tools. They are **not**
conda dependencies of `apex_env`; install them separately and record the
versions, because several metrics move with the version:

| tool | used for | version used here |
|---|---|---|
| MAFFT | multiple sequence alignment, entropy column | see `SNAPSHOT/05_code/ENVIRONMENT.txt` |
| EMBOSS `needleall` | sequence similarity column | 6.6.0 |
| HMMER `hmmscan` | Pfam domain check (MDH) | see `ENVIRONMENT.txt` |
| Macrel | `Avg_macrel` / `P_macrel` columns | 1.6.0 |

`SNAPSHOT/05_code/ENVIRONMENT.txt` records the exact versions the reported
numbers were produced with.

The README describes a CPU generation smoke test that checks the generation
path works at all, without a checkpoint. Run that first; if it fails, nothing
below will work either. Entry-point filenames are given in the README — this
document refers to them by role (`main_*.py` for training, `generated_*.py`
for generation) rather than repeating names that may be renamed.

---

## 2. Recomputing each reported column

All of section 2 reads deposited files. Nothing here needs a GPU.

### 2.1 The sequence sets

`SNAPSHOT/01_generated_sequences/` holds one file per arm, named by the table
and row it feeds: `T1-07_BaseGPT_N-terminal.fasta` is Supplementary Table 1,
row 7. `SNAPSHOT/MANIFEST.tsv` maps every deposited path to its table, row,
model, decoding condition and a one-line note, with an md5 and a byte count.
Start there:

```bash
column -t -s$'\t' SNAPSHOT/MANIFEST.tsv | less -S
```

Two properties of these files matter for every metric below:

- **Fig. 1d and Supplementary Tables 1 and 2 are pooled across decoding
  settings.** For each of the twenty-five combinations of top-p in
  {0.6, 0.7, 0.8, 0.9, 1.0} and temperature in {0.1, 0.2, 0.3, 0.4, 0.5},
  400 sequences were drawn, giving 10,000 per arm.
- **Deduplicate before computing anything that is reported "on unique
  sequences."** The duplicate rate itself is computed on the *drawn* set;
  every other metric is computed after dropping duplicates. Mixing the two is
  the single easiest way to fail to reproduce a cell.

### 2.2 `Avg_apex` and `P_apex` (Supplementary Table 1, Fig. 1d)

APEX predicts MICs against thirty-four strains. **Every value in the paper is
the median across eleven of them**, the panel the synthesis cascade used:

```
E. coli ATCC 11775, E. coli AIC221, E. coli AIC222,
P. aeruginosa PAO1, P. aeruginosa PA14,
K. pneumoniae ATCC 13883, A. baumannii ATCC 19606,
S. aureus ATCC 12600, methicillin-resistant S. aureus ATCC BAA-1556,
vancomycin-resistant E. faecalis ATCC 700802,
vancomycin-resistant E. faecium ATCC 700221
```

- `Avg_apex` = mean, over sequences, of that per-sequence median MIC.
- `P_apex` = percentage of sequences whose median MIC is below 80 µmol L⁻¹.

The per-sequence predictions are deposited, so you do not need to run APEX:

```
SNAPSHOT/04_derived_values/predictions/APEX_11pathogen_per_sequence.csv.gz
SNAPSHOT/04_derived_values/predictions/APEX/*_MIC_prediction.csv.gz
```

`SNAPSHOT/05_code/rescore_11pathogen.py` is the script that performs the
thirty-four to eleven reduction. Run it against the per-strain files to
confirm the reduction, then take the mean and the sub-80 fraction.

> The submitted version of the manuscript reported medians across all
> thirty-four strains. Every value in the revised version is on the eleven-
> pathogen basis. If you are comparing against the submitted PDF, that is the
> difference.

### 2.3 `Avg_macrel` and `P_macrel` (Supplementary Table 1)

Macrel v1.6.0, `AMP_probability` is column index 4 of its output.
`Avg_macrel` is the mean of that column; `P_macrel` is the percentage above
0.5. Per-sequence output is deposited at
`SNAPSHOT/04_derived_values/predictions/Macrel/`.

### 2.4 `Avg_mdh` and `P_mdh` (Supplementary Table 2)

A BERT head fine-tuned on the 33-layer ESM2 model, trained on the balanced
MDH / non-MDH set of Zeng et al. (4,500 train / 500 validation / 28,412 test).
`Avg_mdh` is the mean predicted probability of MDH activity; `P_mdh` is the
percentage above the functional threshold.

**This predictor shares its backbone with the latent-space reward**, so it is
not independent of the training objective. That is why we also report the Pfam
check in 2.7, and why the paper's MDH conclusions rest on the Pfam result
rather than on this column alone.

The scorer is `code/apexgpt/`'s MDH prediction entry point; note that it
writes to a **relative** path and must be run from its own directory.

### 2.5 MSA entropy (Supplementary Tables 1 and 2)

```
MAFFT --auto on the generated set
per column: Shannon entropy in NATS, over NON-GAP residues only
value: the sum over all columns
```

```bash
mafft --auto arm.fasta > arm.aln
python SNAPSHOT/05_code/msa_all.py arm.aln       # prints the summed entropy
```

**Read the caveat before comparing cells.** The statistic is a *sum over
columns*, so it scales with alignment width. A set of more divergent sequences
produces a wider alignment and therefore a larger sum at identical per-residue
diversity. Both table legends say so. Where the paper compares entropy it
reports the per-column value alongside: for the compute-matched MDH arms, the
summed values differ by 28% while the per-column values are 1.0955 against
1.0963 nats — that gap is alignment width, not residue usage.

Entropy is not reported for ProtGPT, whose sequences contain a non-standard
residue that MAFFT excludes by default.

### 2.6 Sequence similarity (Supplementary Tables 1 and 2)

```
EMBOSS needleall, global Needleman-Wunsch
gap-open 10, gap-extend 0.5
query: the generated set;  reference: the training set
per query: the MAXIMUM percent identity to any training sequence
value: mean +/- population SD of those maxima
```

```bash
needleall -asequence arm.fasta -bsequence <the training FASTA under SNAPSHOT/02_reference_sets/> \
          -gapopen 10 -gapextend 0.5 -aformat3 pair -outfile arm.needle -auto
python SNAPSHOT/05_code/seqsimi_all.py arm.needle       # AMP
python SNAPSHOT/05_code/needle_mdh.py   arm.needle       # MDH
```

Lower is more novel. `SNAPSHOT/05_code/mdh_diversity.py` runs the whole
entropy-plus-similarity pipeline chunked across cores and is resumable; it
carries a published arm as a built-in control and refuses to be believed until
that control reproduces.

> One reproducibility note we owe you: `needle_mdh.py` defaults to a
> subsample (`NSAMP=120`). The reported *means* reproduce on the full set, but
> the reported SDs for the MDH table do not — our recomputation of the control
> gives 0.6550 ± 0.2290 against the tabulated 0.6622 ± 0.2710. The means are
> what the comparisons in the text turn on.

### 2.7 Pfam domain recovery (MDH, Results)

Independent of ESM2 entirely, which is the point of it:

```bash
hmmpress <the Pfam profiles deposited under SNAPSHOT/02_reference_sets/>
hmmscan --cut_ga --domtblout arm.domtbl <those profiles> arm.fasta
```

against the two profiles that define the malate/lactate dehydrogenase fold:

| profile | accession | gathering threshold |
|---|---|---|
| Ldh_1_N | PF00056 | 27.0 |
| Ldh_1_C | PF02866 | 22.6 |

- "carries both domains" = both profiles score above their gathering threshold.
- the reported **median bit score** is the median, over sequences, of the
  **better of the two domain scores**. (Not the sum, not the N score, not the
  C score. `SNAPSHOT/05_code/pfam_all.py` prints all four candidate
  readings side by side so the published value identifies its own definition.)

`pfam_all.py` keeps the six published ApexGPT MDH arms as a built-in
control; pooled over them it must return 86.1% at median 146.2.

### 2.8 Duplicate rate and verbatim reproduction (Fig. 1e, Supplementary Table 5)

- **Duplicate rate**: percentage of *drawn* sequences that duplicate an
  earlier draw within the same decoding condition. Computed on the raw draws,
  before deduplication.
- **Verbatim reproduction**: percentage of *unique* generated sequences that
  exactly match a training peptide.

Both are reported at five training lengths (epochs 5, 10, 20, 30, 50) against
the compute-matched unguided baseline.

> Diversity is **not** compared against the five published external baselines.
> Those files were collected until 10,000 *distinct* sequences had been
> obtained, so their duplicate rate is not measured and is not comparable with
> our fixed 10,000-draw runs.

### 2.9 Effect sizes

Every comparison in Fig. 1d, Fig. S5 and the response to referees is reported
as **Cliff's delta** with a 95% percentile bootstrap confidence interval
(1,000 resamples for the length-stratified analysis), not as a p-value:

```
|delta| < 0.147   negligible
        < 0.33    small
        < 0.474   medium
        otherwise large
```

At these group sizes essentially every comparison reaches p < 0.05 regardless
of magnitude, which is why significance annotations were removed from Fig. S2.

`SNAPSHOT/05_code/recompute.py` reproduces the Fig. 1d effect-size table;
`SNAPSHOT/06_figure_source_data/length_stratified_effect_sizes.csv` holds the
per-band values behind Fig. S5 (six length bands x three decoding directions).

---

## 3. Regenerating the sequences

Download `models.zip` from
[10.5281/zenodo.19597375](https://doi.org/10.5281/zenodo.19597375) and unpack
it. Then generate with the configuration files in this repository:

```bash
conda activate apex_env ;
python code/apexgpt/generated_<task>.py \
    --config config_AMP_EMA.json \
    --checkpoint <path from models.zip> \
    --strategy N_terminal
```

Three decoding strategies are used throughout, and the naming is consistent
across both tasks:

| strategy | meaning | file suffix |
|---|---|---|
| N-terminal | generate N-terminus to C-terminus | `_1_aa` |
| C-terminal | generate C-terminus to N-terminus | `_2_aa` |
| Mixture | even mixture of both | `_1_2_aa` |

Model naming: `AMPGPT` / `MDHGPT` are ApexGPT instantiated on the two tasks;
the `_EMA` suffix means the exponential-moving-average reward baseline;
`BaseGPT` is the ablation trained without the latent-space policy-gradient
term. In the main text `ApexGPT` is the general name for the guided model.

For Fig. 1d and Supplementary Tables 1 and 2, models labelled `AMPGPT` were
trained with λ = 100 and those labelled `AMPGPT_EMA` with λ = 10 plus the EMA
baseline. All 50 synthesized peptides came from the single λ = 10 EMA model.

**Checkpoint selection.** The routine intended to retain the best-perplexity
checkpoint did not work: for the AMP models it compared a summed evaluation
loss against a threshold on the scale of a mean, so it never fired; for the
MDH models the retained value was stored as a sum while the comparison used a
mean, so the checkpoint was overwritten every epoch. Checkpoints were
therefore kept at every epoch and chosen by hand. The manuscript states the
checkpoint index for each analysis that depends on it. Do not assume a
"best" checkpoint exists.

---

## 4. Retraining

```bash
python code/apexgpt/main_<task>.py --config config_<TASK>_EMA.json
```

Training details that the configs encode: AdamW, betas 0.9 / 0.999, eps 1e-8;
AMP 50 epochs at batch size 16, MDH 50 epochs at batch size 4, both at
learning rate 1e-5; 5% warm-up to a peak of 1e-5 then decay; start and end
tokens added to every sequence; reversed sequences used as augmentation so the
model learns both directions; dipeptide seeding from the training-set terminal
dipeptide frequencies (`data/frequency_seeds/*.pkl`).

To build the **unguided matched baseline**, do not write a new trainer. Derive
it from the guided script by removing only the latent-space term, so that data
loading, the training set, the augmentation, the seeding, the schedule, the
seed, the batch size, the learning rate and the epoch count are inherited
unchanged. `SNAPSHOT/05_code/make_matched_basegpt.sh` does exactly this and
prints the diff before running anything. It deliberately leaves the ESM2
forward pass in place even though its output is then unused, so that batching
and RNG consumption stay identical between the two arms.

That property is what makes the comparison a control rather than a
coincidence, and it is worth preserving in any reimplementation: the published
unguided MDH baseline was trained for ten epochs through a different code path,
and the difference between it and the guided models turned out to be mostly
that, not the guidance.

---

## 5. If a number does not reproduce

In order of how often it was the cause, for us:

1. **Deduplication.** Duplicate rate on drawn sequences; everything else on
   unique sequences.
2. **The eleven-pathogen reduction.** APEX returns thirty-four strains; the
   paper reports medians across eleven.
3. **Alignment width.** MSA entropy is a sum over columns and is comparable
   only among sets of similar length distribution.
4. **Which bit score.** "Median bit score" is the median of the better of the
   two Pfam domain scores.
5. **Tool version.** MAFFT, HMMER and Macrel versions all move these numbers.
   `SNAPSHOT/05_code/ENVIRONMENT.txt` has the ones we used.

If it still does not reproduce, please open an issue on this repository with
the arm, the metric, the value you obtained and the tool versions. Every
deposited file carries an md5 in `MANIFEST.tsv`, which is the fastest way to
establish whether we are looking at the same bytes.
