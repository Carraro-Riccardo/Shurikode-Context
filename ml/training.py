from typing import Callable, Dict, List, Sequence
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report
import pandas as pd

from utils import (
    ModelEvaluationFunction,
    log_elapsed_remaining_total_time,
    ConsoleStatsLogger,
    ConditionalSave,
    Result,
    EarlyStopping,
)
from custom_types import DeviceType

import torch.nn as nn
import time
import torch
import wandb
from tqdm import tqdm

def train(model, loss_function, optimizer, scheduler,
          train_dataloader, val_dataloader, device,
          evaluation_functions, epoch_n, saver, early_stopping=None):

    console_logger = ConsoleStatsLogger(epoch_n)
    elapsed_time = 0

    for i in range(epoch_n):
        start_time_epoch = time.time()

        train_epoch(
            model, loss_function, optimizer, train_dataloader,
            device, evaluation_functions, epoch=i, epoch_n=epoch_n
        )

        val_stats = validate_model(
            model, loss_function, val_dataloader, device,
            evaluation_functions, prefix="val", epoch=i
        )

        # Stampa validazione in console
        val_str = " | ".join([f"{r.get_name()}: {r.get_value():.4f}" for r in val_stats])
        print(f"\n[Val] Epoch {i}/{epoch_n-1} → {val_str}")

        scheduler.step()
        saver(model, val_stats, i)

        elapsed_time += time.time() - start_time_epoch
        log_elapsed_remaining_total_time(elapsed_time, i + 1, epoch_n)

        if early_stopping and early_stopping(val_stats):
            break


def train_epoch(
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: DeviceType,
    evaluation_functions: Sequence[ModelEvaluationFunction],
    epoch: int = 0,
    epoch_n: int = 0,
):
    model.train()

    running_loss = 0.0
    running_evals = {ef.get_name(): 0.0 for ef in evaluation_functions}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epoch_n-1} [Train]", leave=True)

    for batch_idx, (img, gt) in enumerate(pbar):
        img, gt = img.to(device), gt.to(device)

        pred: Tensor = model(img)
        loss: Tensor = loss_function(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running averages
        running_loss = ((running_loss * batch_idx) + loss.item()) / (batch_idx + 1)
        evals: Dict[str, float] = {}
        for eval_func in evaluation_functions:
            val = eval_func(pred, gt).get_value()
            evals[eval_func.get_name()] = val
            running_evals[eval_func.get_name()] = (
                (running_evals[eval_func.get_name()] * batch_idx) + val
            ) / (batch_idx + 1)

        # Aggiorna la barra con loss e metriche correnti
        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            **{k: f"{v:.4f}" for k, v in running_evals.items()},
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
        })

        wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], **evals})


def validate_model(
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    dataloader: DataLoader,
    device: DeviceType,
    evaluation_functions: Sequence[ModelEvaluationFunction],
    prefix="val",
    epoch=0,
) -> List[Result]:
    model.eval()
    evals = [0.0] * len(evaluation_functions)
    avg_loss = 0

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(dataloader):
            img, gt = img.to(device), gt.to(device)
            pred = model(img)
            loss = loss_function(pred, gt)
            avg_loss = ((avg_loss * batch_idx) + loss.item()) / (batch_idx + 1)
            for func_idx, eval_func in enumerate(evaluation_functions):
                evals[func_idx] = (
                    (evals[func_idx] * batch_idx) + eval_func(pred, gt).get_value()
                ) / (batch_idx + 1)

    stats_w_names = [Result("Loss", avg_loss)] + [
        Result(evaluation_functions[i].get_name(), evals[i])
        for i in range(len(evaluation_functions))
    ]
    wandb.log({
        f"{prefix}/loss": avg_loss,
        **{f"{prefix}/{e.get_name()}": e.get_value() for e in stats_w_names},
        "val_epoch": epoch,
    }, commit=True)
    return stats_w_names

def log_confusion_matrix(model, dataloader, device, n_classes, prefix="val"):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for img, gt in dataloader:
            img = img.to(device)
            pred = model(img).argmax(dim=1).cpu()
            all_preds.extend(pred.tolist())
            all_gts.extend(gt.tolist())
    
    wandb.log({
        f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_gts,
            class_names=[str(i) for i in range(n_classes)]
        )
    })

def log_per_class_metrics(model, dataloader, device, n_classes, prefix="val"):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for img, gt in dataloader:
            img = img.to(device)
            pred = model(img).argmax(dim=1).cpu()
            all_preds.extend(pred.tolist())
            all_gts.extend(gt.tolist())

    report = classification_report(all_gts, all_preds, output_dict=True)
    df = pd.DataFrame(report).T
    wandb.log({f"{prefix}/per_class_metrics": wandb.Table(dataframe=df)})