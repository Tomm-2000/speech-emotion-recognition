"""Unified entry point. Heavy audio dependencies are loaded only when needed."""
import argparse
import random

import numpy as np
import yaml

from ser.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description="Speaker-disjoint speech emotion experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", required=True, choices=[
        "smoke", "reference", "extract", "evaluate", "ensemble",
        "regularization", "fit", "verify",
    ])
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow pinned encoder downloads during bundle verification.")
    args = parser.parse_args()
    from ser import experiments as core
    with (ROOT / args.config).open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if config["classifier"]["name"] != "logistic":
        raise ValueError("Only logistic classifiers are supported by this runner.")
    if config["encoder"]["pooling"] != "all_transformer_mean":
        raise ValueError("This runner requires all_transformer_mean pooling.")
    if config["runtime"]["smoke_recordings"] < 1:
        raise ValueError("smoke_recordings must be positive.")
    seed = config["runtime"]["seed"]
    random.seed(seed); np.random.seed(seed)
    if args.stage in {"reference", "fit", "verify"}:
        from ser import bundle
        if args.stage == "verify":
            bundle.verify(config, args.allow_download)
        else:
            getattr(bundle, args.stage)(config)
        return
    if args.stage == "regularization":
        from ser.regularization import run_regularization
        run_regularization(config, core)
        return
    rows, folds, manifest, fold_file = core.load_data(config)
    if args.stage == "ensemble":
        core.ensemble(rows, folds, config, manifest, fold_file)
        return
    if args.stage in {"smoke", "extract"}:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(config["runtime"]["cpu_threads"])
    output, revision = core.prepare_encoder_run(config, manifest, fold_file)
    if args.stage == "evaluate":
        core.evaluate(rows, folds, config, output, revision)
    else:
        core.extract_stage(args.stage, rows, config, output, revision)


if __name__ == "__main__":
    main()
