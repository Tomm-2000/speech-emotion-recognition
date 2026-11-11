# Refactor scope and migration

Base commit: `93cf2b04870dece3f7eae921e625b32cf5855601`.

## Changes

- Replaced the dynamically loaded `run_experiment_previous.py` with normal
  package modules. The root runner dispatches all current stages.
- Moved 16 historical scripts out of the root into `experiments/legacy/`.
  Their internal imports and repository-root paths were updated.
- Consolidated three historical feature builders into
  `experiments/legacy/build_baseline_features.py`.
- Consolidated MFCC extraction in `ser/mfcc.py`; older extractor module names
  remain as compatibility entry points inside the legacy package.
- Moved reusable inference into `ser/inference.py`; kept root `predict.py`.
- Added final ensemble fitting and audio verification as explicit stages.
- Added pinned fresh-run configuration separate from historical locked settings.
- Preserved all original JSON, CSV, YAML and text experiment artifacts byte-for-byte.
- Removed only the tracked listening playlist `targeted_audio_audit.m3u8`.
- Preserved the old environment snapshot and reconciled PyYAML to the version
  recorded in the actual experiment provenance.
- Replaced the stale project status, which still described the MFCC baseline.

## Deliberately not merged

The historical CNN evaluator and the larger feature/pooling evaluator have
substantially different input masks, normalization and training protocols.
They remain separate legacy modules. Collapsing those into one untested training
loop would risk changing the numerical experiments. They do not clutter the
current root workflow.

Original report paths were retained because existing provenance and documentation
refer to them. Moving them solely for aesthetics would add migration risk.

## Applying the supplied patch

Save `speech-emotion-cleanup.patch` in the parent Downloads folder.
From the existing project directory:

```powershell
git status --short
git switch -c refactor/clean-project
git apply --check ..\speech-emotion-cleanup.patch
git apply ..\speech-emotion-cleanup.patch
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If the working tree contains changes or `git apply --check` fails, stop and
inspect the difference. Do not force, reset, or delete local work.
The patch is based on the commit above. It does not change ignored audio,
features, virtual environments, or saved model files.

After the tests, run the existing `predict.py` command on a known recording
and compare it with its previous output. Then review and commit the branch:

```powershell
git add .
git --no-pager diff --cached --check
git commit -m "Organize speech emotion pipelines and add ensemble fitting workflow"
git push -u origin refactor/clean-project
```

Open a pull request to main. This package has not pushed or merged anything.
The source ZIP is a clean snapshot, but extracting it over the old project will
not remove obsolete files. Use the patch for migration.
