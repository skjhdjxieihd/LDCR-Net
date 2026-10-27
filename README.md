# Anonymous review code

This review snapshot implements synthetic-data generation, LDCR-Net training,
calibration, evaluation and plotting. Author identities and manuscript-title
metadata are omitted during anonymous review.

This is a **source-only release**. It provides a complete runnable workflow:

```text
Generate synthetic training/calibration/test data
    -> Train LDCR-Net, five ablations and three learned baselines
    -> Calibrate detectors and fit grouping prototypes
    -> Test all methods and compute metrics/intervals
    -> Export comparison plots, a recovered timeline and training curves
```

No original dataset, pretrained model, saved grouping prototype, or paper prediction record is distributed. All of these runtime assets are generated locally by the commands below. The workflow produces a new experiment; obtaining exactly the paper's numerical values is not required. The small `smoke` profile checks the complete program flow, while the `full` profile provides a larger training and evaluation configuration.

## 1. Installation

Use Python 3.10 or 3.11. The checked environment uses Python 3.10.20 and PyTorch 2.5.1. Open a terminal in this code directory and create a virtual environment:

```text
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```sh
. .venv/bin/activate
```

Install dependencies and inspect the command line:

```text
python -m pip install -r requirements.txt
python run.py --help
```

Dependencies are NumPy, SciPy, scikit-learn, Matplotlib, PyTorch, threadpoolctl and ruptures. CPU execution is supported. To use CUDA, install a PyTorch build compatible with the local GPU/driver before running with `--device cuda`. No LaTeX, Word, original research directory, downloaded dataset or pretrained checkpoint is needed. Internet access is only needed for installing dependencies.

## 2. Run the complete workflow with one command

First run the small check:

```text
python run.py pipeline --profile smoke --workdir runs/smoke
```

This starts from an empty directory, generates three independent corpora, trains **all nine learned models** for one epoch each, calibrates on generated calibration sequences, tests every method, computes confidence intervals, exports all figures and checks the generated artifacts. A successful run ends with:

```text
CHECK PASSED: ... workflow checks.
PIPELINE COMPLETE: ...
```

The report is `runs/smoke/integrity_report.json`; figures are in `runs/smoke/figures/`. One epoch and three test cases are intended for checking execution. Use the larger profile for substantive experiments:

```text
python run.py pipeline --profile full --workdir runs/full
```

For CUDA training and supported inference:

```text
python run.py pipeline --profile full --workdir runs/full_cuda --device cuda
```

Boundary-ablation inference, CPC evaluation, PELT and LSTM-FCN evaluation use the extracted CPU implementations. CUDA is used for training and supported sequence/BiLSTM inference. Full experiments can take substantially longer than the smoke run, especially on CPU: training six LDCR variants, recurrent baselines, PELT and bootstrapping all contribute.

| Setting | `smoke` | `full` |
|---|---:|---:|
| Training sequences | 16 | 4,000 |
| Calibration sequences | 8 | 300 |
| Test sequences | 3 | 300 |
| Full LDCR / structural-ablation epochs | 1 | 30 |
| Invariance-only ablation epochs | 1 | 12 |
| CPC epochs | 1 | 30 |
| Supervised maximum epochs | 1 | 20 |
| LDCR/CPC batch size | 4 | 32 |
| Whole-sequence ARI bootstrap replicates | 10 | 1,000 |

Supervised training uses at most 1,000 training sequences, with validation-based checkpoint selection and early stopping. Each fresh run needs a new empty `--workdir`; generation refuses to overwrite an existing experiment. To repeat a run, choose another directory such as `runs/smoke_2`.

## 3. Run each stage separately

These commands perform the same workflow as `pipeline`. Use the **same workdir** at every stage:

```text
python run.py generate --profile full --workdir runs/manual
python run.py train --workdir runs/manual
python run.py calibrate --workdir runs/manual
python run.py evaluate --workdir runs/manual
python run.py plot --workdir runs/manual
python run.py check --workdir runs/manual
```

For a small manual check, change only the generation profile:

```text
python run.py generate --profile smoke --workdir runs/manual_smoke
python run.py train --workdir runs/manual_smoke
python run.py calibrate --workdir runs/manual_smoke
python run.py evaluate --workdir runs/manual_smoke
python run.py plot --workdir runs/manual_smoke
python run.py check --workdir runs/manual_smoke
```

The generated `run_config.json` stores the chosen profile, sizes, seeds and training recipe. Subsequent stages read that saved configuration. Add `--device cuda` separately to `train`, `calibrate` and `evaluate` when desired. `plot` and `check` require no device option.

To train only one model after generation:

```text
python run.py train --workdir runs/manual --model full
python run.py train --workdir runs/manual --model cpc
```

Available names are `full`, `no_median`, `no_multiscale`, `no_reconstruction`, `no_tv`, `no_invariance`, `cpc`, `bilstm`, `lstm_fcn`, and `all`. A complete comparison requires `all`, which is the default. PELT has no learned checkpoint; its penalty is chosen during calibration. After changing trained weights, run calibration and evaluation again before plotting.

## 4. Customize a new experiment

Run with different generated data sizes and seeds:

```text
python run.py pipeline --profile smoke --training-size 64 --calibration-size 16 --test-size 10 --seed 81000 --epochs 2 --workdir runs/custom
```

Here training/calibration/test use seeds 81000, 81001 and 81002. At least seven calibration sequences are required for seven grouping prototypes. `--epochs` overrides all selected models' epoch limits; without it, the profile-specific limits apply. The same size/seed options are available on `generate`, and `--epochs` is also available on `train`.

The public recipe is `config/experiment.json`. It defines architecture/loss settings, profile sizes, training recipes, perturbation levels and generation seeds. Edit it **before generating a new run**; already generated experiments continue using their own saved `run_config.json`. Training defaults to random seed 42. Generation defaults to separate seeds 61731, 61732 and 61733. Training and calibration never read the generated test corpus.

For the supplied figure layout, main loss levels must lie in 0–20% and include 0/20%; extended loss must include those main levels and reach above 20% within 0–60%; receiver errors must lie in 0–2 and include 0/2. Values must be unique. Generation validates these requirements, including the 20% timeline operating point, before writing data. Additional intermediate levels within these ranges are supported.

## 5. Generated data and simulator

`generate` writes three local NPZ files: `data/training.npz`, `data/calibration.npz` and `data/test.npz`. Each contains an object array called `data`. Every entry is a sequence dictionary with:

- `pdw_data`: events × 4 relative pulse descriptors;
- `radar_pdw`, `jammer_pdw`: the paired absolute pulse records;
- `change_points`: reference switch indices;
- `labels`: per-event simulator mechanism labels;
- correspondence/generation metadata, including split, sequence ID and seed.

The four input channels are relative delay (microseconds), relative carrier frequency (MHz), pulse-width ratio and relative amplitude (dB). The seven descriptor templates are FRFTJ, RGPO, VGPO, R-VGPO, ISRJ, DFTJ and NSJ. Default sequences have 4–10 state changes and 80–180 events per segment.

This is a controlled descriptor-level simulation with deliberately separable mechanism templates. It does not implement waveform reception or pulse detection/association. The package keeps the paper's descriptor generator and scientific model/metric implementations, with a complete new-run workflow around them.

All three clean corpora are generated without additional event loss. Evaluation applies nested event loss at 0, 5, 10, 15 and 20%, and additionally 30, 40, 50 and 60% for sequence recovery. Receiver-error scales are 0, 0.5, 1, 1.5 and 2. Error fields combine bias, drift, low-frequency wander and jitter. Conditions share each clean test sequence and paired perturbation realizations.

## 6. Training and calibration

LDCR-Net and its five ablations train from random initialization on generated training inputs. The objective excludes boundary and semantic labels. LDCR uses AdamW (`lr=3e-4`, `weight_decay=1e-4`), gradient clipping 1 and the architecture/loss parameters in the configuration. `no_invariance` sets only the cross-view invariance coefficient to zero while retaining variance/covariance regularization and the remaining constraints.

CPC trains its supplied architecture and InfoNCE objective from random initialization using raw four-channel RPDW crops of length 256, stride 128, Adam (`lr=1e-3`) and gradient clipping 1. Fixed-length crops avoid padding in the contrastive objective. This is the explicit training recipe of this source release.

BiLSTM and LSTM-FCN use generated boundary annotations, Adam (`lr=1e-3`) and gradient clipping 5. The existing crop/window sampling, label radii and validation-based selection procedures are retained. Training histories, selected checkpoint identities, optimizer states, data fingerprints and run seeds are saved locally.

`calibrate` then uses **only the generated calibration split** to select supervised thresholds, CPC prominence and PELT penalty, and to fit the seven grouping prototypes for each of TV, TI and raw-descriptor pooling. Semantic alignment is used for offline scoring. No calibration setting or prototype is bundled with the source release.

LDCR boundary detection retains fixed input-coordinate settings: dual-window sides 12/16/20, Gaussian sigma 2, minimum width 2, minimum separation 50 and prominence 0.30. The no-median variant converts those settings using its own output stride. One-to-one boundary matching uses tolerance 20 retained observations. All grouping comparisons use the same detected LDCR boundaries.

## 7. Outputs and figures

```text
<workdir>/
  run_config.json             Exact configuration for this new experiment
  generation.json             Split seeds, sizes and generated-file hashes
  data/                       Locally generated training/calibration/test corpora
  checkpoints/                Nine fresh models, training histories and metadata
  calibration/                New detector settings, prototypes and calibration grids
  results/                    Test records, metric summaries and bootstrap intervals
  figures/                    PDFs, ablation previews, tables and timeline records
  integrity_report.json       Workflow checks and environment information
```

Important measurement outputs include `boundary_records.csv`, `boundary_summary.csv`, `ablation_summary.csv`, `paired_per_sequence.csv`, `paired_summary.csv`, `global_ari_bootstrap.csv`, `reference_consistency.csv/.json` and `complexity.csv`. Global segment ARI differs from mean per-sequence ARI. Edit, transition-edge and duration metrics use the supplied sequence-scoring definitions and retained-event durations. ARI intervals resample complete sequences; mean sequence-metric intervals use the extracted 2,000-resample procedure. The smoke profile reduces only the whole-sequence ARI bootstrap count to 10.

| Figure output | Contents |
|---|---|
| `figure24_unified_boundary.pdf` | LDCR-Net, CPC, PELT, BiLSTM and LSTM-FCN comparison |
| `figure25_unified_ablation_loss.pdf` | Six LDCR variants under observation loss |
| `figure26_unified_ablation_error.pdf` | Six LDCR variants under receiver error |
| `figure21_paired_pressure.pdf` | TV/TI/raw-descriptor sequence recovery |
| `figure22_paired_timeline.pdf` | First generated test case at 20% event loss |
| `figure23_extended_loss.pdf` | Sequence recovery through 60% event loss |
| `training_losses.pdf` | Fresh training losses for all nine learned models |

The original asset numbers are retained in filenames. All figures use this run's measurements. `paper_tables.csv` exports representative operating points from the new run. Conceptual diagrams and idealized explanatory waveform illustrations are outside the numerical pipeline.

`check` verifies generated-data hashes and split seeds, fresh model provenance and finite losses, calibration/evaluation/plot fingerprints, expected result counts, numerical finiteness, boundary-summary consistency and valid figure files. Changing a trained model requires calibration, evaluation and plotting again; changing plotted measurements requires plotting again. It does not impose a minimum accuracy or equality to published numerical values.

## 8. Public code directory and cleanup

The distributed folder contains only source, configuration, documentation, dependency specifications, tests, license notices and verification metadata. `PROVENANCE.json` maps the packaged components without exposing local development paths or historical submission records. `SHA256SUMS.json` records current release-file hashes. `VERIFICATION.json` records the scope of execution checks; `ANONYMITY_CHECK.json` records content-level distribution checks.

Run the source-only acceptance tests with:

```text
python -B -m unittest discover -s tests -v
```

Additional state-integrity tests use a completed local run when the environment variable `LDCR_TEST_WORKDIR` points to it. Without that variable they are skipped. The tests temporarily alter and restore run metadata/figures to check that stale assets are rejected.

Generated data, checkpoints, prototypes, predictions, plots and caches belong to the chosen workdir. Delete that experiment directory when no longer needed. Keep the code directory for another run. `runs/`, virtual environments, caches, NPZ datasets and model files are excluded by `.gitignore`; exclude generated experiment directories when preparing a source-only public archive.

## 9. License and anonymous review

Project contributions are released under the MIT license in `LICENSE`,
using a collective contributor designation during anonymous review.
Upstream-derived CPC portions retain the separate MIT copyright and
permission notice in `LICENSES/UnsupSeg-MIT.txt`; see
`THIRD_PARTY_NOTICES.md`. Required third-party notices must remain intact.

This source snapshot contains no repository history, author contact fields,
affiliation, funding identifiers or personal filesystem paths. It does not
control the identity of a future hosting account, commit author/email,
repository history, DOI record or an already public copy of the code.
Use a journal-permitted anonymous review link and verify its downloadable
archive before supplying it to reviewers. After anonymous review ends,
author/citation metadata can be added for the public research release.
