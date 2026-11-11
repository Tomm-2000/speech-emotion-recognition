# Setup and execution

Run all commands from the repository root in Windows PowerShell.

## Existing working environment

Keep the existing `.venv`, `data/`, personal recordings, caches, and ignored
model bundles. Applying the supplied Git patch changes tracked source files,
not those local assets. Do not recreate an environment that already works.

The original `config.yaml` is intentionally unchanged: historical locks compare
its encoder settings and package versions. New bundle outputs use a separate
`artifacts/models/equal_ensemble_refactor_v1` directory.

1. Run the lightweight checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

2. Test the existing bundle with the usual `predict.py` command.
3. Run `--stage fit`, then `--stage verify` with the original config only when
   ready to validate the new final-fitting workflow.

Verification checks one development recording per speaker against saved
features and probabilities. It never reads the held-out test recordings.
It calls the actual inference function, including loading pinned models.
This is slower than retaining both encoders in memory, but limits memory use.
No new accuracy result is produced.

## Fresh environment

The following is the intended workflow, not a claim that the historical package
versions were successfully installed in a clean environment during this refactor.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

If a historical version is unavailable, stop and record the error. Do not silently
replace versions and claim the old results have been reproduced.

Obtain RAVDESS speech audio from https://zenodo.org/records/1188976 and comply
with its terms. Extract so that this path exists:

`data/raw/Actor_01/03-01-01-01-01-01-01.wav`

```powershell
.\.venv\Scripts\python.exe .\check_dataset.py
.\.venv\Scripts\python.exe .\build_manifest.py
```

The manifest builder refuses to overwrite an existing manifest. Preserve
`speaker_splits.json` and `speaker_folds.json`.

## Fresh reproduction run

`configs/reproduce.yaml` pins the WavLM revision and uses new output directories.
The reference stage pins Wav2Vec2 to the revision verified in the original project.
It retains the historical representation and classifier design.

Run in this order:

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage smoke
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage reference
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage extract
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage evaluate
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage ensemble
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage fit
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage verify --allow-download
```

`smoke` checks five training recordings; `reference` and `extract` compute the
two encoders' development features; `evaluate` scores WavLM; `ensemble` scores
both encoders and equal fusion; `fit` trains on all development speakers;
`verify` checks audio-to-prediction consistency before creating a verified bundle.

Outputs refuse replacement. If a stage fails after creating a partial output,
inspect it and select a fresh experiment output path; do not delete historical
results or relax provenance checks merely to continue.

Optional historical regularization comparison, after `ensemble`:

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py --config .\configs\reproduce.yaml --stage regularization
```

This experiment did not beat the baseline in the historical run and is not
required for fitting the selected model.

## Earlier experiments

Run historical modules with `-m` from the root, not by their nested file path:

```powershell
.\.venv\Scripts\python.exe -m experiments.legacy.build_baseline_features mfcc
.\.venv\Scripts\python.exe -m experiments.legacy.build_baseline_features sequences
.\.venv\Scripts\python.exe -m experiments.legacy.build_baseline_features prosody
.\.venv\Scripts\python.exe -m experiments.legacy.evaluate_folds wav2vec2 wav2vec2_all_layers
```

The prosody builder requires the existing MFCC feature file. Earlier neural
experiments require their own feature files. They were retained for traceability,
not included in the current ensemble's default execution path.

Do not rerun `experiments.legacy.evaluate_test` to guide model selection.
The original test set has already been inspected.
