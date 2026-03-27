import torch
import pickle
from torch.utils.data import Subset
from model.dataset import train_test_split, YaleDatasetWithMissingnessInfo
from model.train import train_rnn_yale, test_rnn_yale, cross_validate_rnn_yale
from pathlib import Path
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import os

# Optional wandb import
try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(root: str, p: str) -> str:
    """Use absolute path as-is; otherwise join with root."""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(root, p)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Arguments:")
    print(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = None
    if cfg.logger.use_wandb:
        if wandb is None:
            raise ImportError("logger.use_wandb=true but wandb is not installed.")
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )

    seed_everything(seed=cfg.training.seed)

    dataset_path = resolve_path(cfg.paths.root, cfg.paths.dataset)
    result_root = resolve_path(cfg.paths.root, cfg.paths.result)

    dataset_tag = os.path.splitext(os.path.basename(cfg.paths.dataset))[0].split("_")[-1]
    path_prepend = (
        f"{result_root}/{dataset_tag}/{cfg.training.seed}/"
        f"{cfg.model.task}_target{cfg.model.targetidx}dim{cfg.model.output_dim}_{cfg.model.type}{cfg.model.rnn_type}"
    )

    # important for custom pickled Dataset objects
    yaledataset = torch.load(dataset_path, weights_only=False)

    if cfg.validation.mode == "cross_val":
        cross_val(cfg, yaledataset, path_prepend)
    elif cfg.validation.mode == "train_only":
        train_only(cfg, yaledataset, path_prepend)
    else:
        raise ValueError(f"Unknown validation mode: {cfg.validation.mode}")

    if run is not None:
        run.finish()


def cross_val(cfg, yaledataset, path_prepend):
    fold_idx_path = resolve_path(cfg.paths.root, cfg.paths.fold_idx)
    with open(fold_idx_path, "rb") as f:
        fold_idx = pickle.load(f)

    datasets = [Subset(yaledataset, fold_id) for fold_id in fold_idx]
    metrics, train_datasets, calib_datasets, test_datasets, models = cross_validate_rnn_yale(
        datasets,
        model_type=cfg.model.type,
        rnn_type=cfg.model.rnn_type,
        task=cfg.model.task,
        target_index=cfg.model.targetidx,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        output_dim=cfg.model.output_dim,
        calibration=cfg.calibration.enabled,
        calibration_pct=cfg.calibration.pct,
        calibration_epochs=cfg.calibration.epochs,
        calibration_lr=cfg.calibration.lr,
        n_bins_ece=cfg.calibration.n_bins_ece,
        seed=cfg.training.seed,
    )

    savepath = f"{path_prepend}_crossval"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for i in range(len(metrics)):
        torch.save(metrics[i], f"{savepath}/metrics_fold{i}.pt")
        torch.save(train_datasets[i], f"{savepath}/train_dataset_fold{i}.pt")
        if cfg.calibration.enabled:
            torch.save(calib_datasets[i], f"{savepath}/calib_dataset_fold{i}.pt")
        torch.save(test_datasets[i], f"{savepath}/test_dataset_fold{i}.pt")
        torch.save(models[i], f"{savepath}/model_fold{i}.pt")
        torch.save(models[i].state_dict(), f"{savepath}/model_state_dict_fold{i}.pt")


def train_only(cfg, yaledataset, path_prepend):
    savepath = f"{path_prepend}_trainonly"
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    model = train_rnn_yale(
        yaledataset,
        model_type=cfg.model.type,
        rnn_type=cfg.model.rnn_type,
        task=cfg.model.task,
        target_index=cfg.model.targetidx,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        output_dim=cfg.model.output_dim,
    )
    torch.save(model, f"{savepath}/model.pt")


if __name__ == "__main__":
    main()