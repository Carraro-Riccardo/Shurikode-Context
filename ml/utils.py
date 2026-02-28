from typing import Dict, Any, Literal, List
from abc import ABC, abstractmethod
from torch import Tensor

import torch.nn as nn
import numpy as np
import random
import torch
import os

from custom_types import EvaluationType


def find_device():
    """
    Returns the type of device found on the system.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("TRAINING ON CUDA")
    elif torch.mps.is_available():
        device = "mps"
        print("TRAINING ON MPS")
    return device


def save_model(
    checkpoints_path: str, name: str, model_state_dict: Dict[str, Any]
) -> str:
    """
    Saves the model as a checkpoint file.

    :param checkpoints_path: The path of the directory where checkpoints are stored.
    :param name: The name that has to be used for the current checkpoint.
    :param model_state_dict: The model state.

    :return: The path to the newly created checkpoint file.
    """
    checkpoint_filename = os.path.join(checkpoints_path, f"checkpoint_{name}.pth.tar")
    torch.save(model_state_dict, checkpoint_filename)
    return checkpoint_filename


class Result:
    """
    A class that represents evaluation results.
    """

    def __init__(self, name: str, val: float) -> None:
        self.__name = name
        self.__val = val

    def get_name(self):
        return self.__name

    def get_value(self):
        return self.__val


class ModelEvaluationFunction(ABC):
    """
    An abstract class for implementing evaluation functions for model prediction evaluation.
    """

    def __init__(self, name: EvaluationType) -> None:
        super().__init__()
        self.__name = name

    def get_name(self):
        return self.__name

    @abstractmethod
    def __call__(self, pred: Tensor, gt: Tensor) -> Result:
        """
        Given the prediction logits and the ground truth, it computes the result on the current batch.
        """
        pass


class AverageAccuracyVectorOutput(ModelEvaluationFunction):
    def __init__(self) -> None:
        super().__init__("Accuracy")

    def __call__(self, pred: Tensor, gt: Tensor) -> Result:
        batch_size = pred.shape[0]
        pred = pred.argmax(dim=1)
        corrects = (pred == gt).int().sum()
        return Result(self.get_name(), corrects.item() / batch_size)


class ConsoleStatsLogger:
    def __init__(self, epoch_n: int) -> None:
        """
        Creates a logger for the `Result` type.

        :param epoch_n: The total number of epochs for training.
        """
        self.__epoch_n = epoch_n - 1

    def __call__(self, type: str, stats: List[Result], epoch: int) -> Any:
        """
        Logs the various collected statistics in the console.

        :param type: The type of dataset which the statistics refers to.
        :param stats: An array of results to be logged.
        :param epoch: The current epoch of training.
        """
        print(
            "##########################################################################################################################"
        )
        str_stats = "| "
        for stat in stats:
            str_stats = str_stats + f"{stat.get_name()}: {stat.get_value():.4f} | "
        print(f"({type}) Epoch {epoch}/{self.__epoch_n}: {str_stats}")


def log_elapsed_remaining_total_time(
    elapsed_time: float, epochs_done: int, epochs_n: int
) -> None:
    """
    Logs the elapsed, remaining and total time for training.

    :param elapsed_time: The elapsed number of seconds.
    :param epochs_done: The number of completed epochs.
    :param epochs_n: The total number of epochs.
    """
    e_hours, remainder = divmod(elapsed_time, 3600)
    e_minutes, e_seconds = divmod(remainder, 60)

    total_time = elapsed_time / epochs_done * epochs_n
    t_hours, remainder = divmod(total_time, 3600)
    t_minutes, t_seconds = divmod(remainder, 60)

    remaining_time = total_time - elapsed_time
    r_hours, remainder = divmod(remaining_time, 3600)
    r_minutes, r_seconds = divmod(remainder, 60)

    print(f"ELAPSED TIME: {e_hours}h {e_minutes}m {e_seconds:.2f}s")
    print(f"REMAINING TIME: {r_hours}h {r_minutes}m {r_seconds:.2f}s")
    print(f"TOTAL TIME: {t_hours}h {t_minutes}m {t_seconds:.2f}s")

class EarlyStopping:
    def __init__(self, patience: int, objective: Literal["minimize", "maximize"], metric_name: str) -> None:
        self.patience = patience
        self.objective = objective
        self.metric_name = metric_name
        self._counter = 0
        self._best = float("inf") if objective == "minimize" else float("-inf")

    def __call__(self, model_stats: List[Result]) -> bool:
        value = next((r.get_value() for r in model_stats if r.get_name() == self.metric_name), None)
        assert value is not None, f"Metric {self.metric_name} not found"

        improved = value < self._best if self.objective == "minimize" else value > self._best
        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            print(f"Early stopping triggered after {self._counter} epochs without improvement.")
            return True
        return False

class ConditionalSave:
    """
    Given an objective to maximize or minimize, it saves the checkpoints that improve on the objective, while deleting
    the less performing ones.
    """

    def __init__(
        self,
        objective: Literal["minimize", "maximize"],
        metric_name: EvaluationType,
        checkpoints_dir: str,
        experiment_name: str,
    ) -> None:
        """
        Creates the conditional saver.

        :param objective: Explicits wether the objective metric has to be maximized or minimized.
        :param metric_name: The metric name to be maximized or minimized.
        :param checkpoints_dir: The path of where to save the checkpoints.
        :param experiment_name: Name of the experiment (it will be used to personalize the checkpoint name).
        """
        self.__checkpoint_files: List[str] = []
        self.__best_evaluation = float("-inf")
        if objective == "minimize":
            self.__best_evaluation = float("inf")
        self.__metric_name = metric_name
        self.__objective = objective
        self.__checkpoints_dir = checkpoints_dir
        self.__experiment_name = experiment_name

    def __call__(
        self,
        model: nn.Module,
        model_stats: List[Result],
        epoch: int,
    ) -> None:
        # Finding the evaluation metric index from the various statistics retrieved
        metric_index = [
            i
            for i, metric in enumerate(model_stats)
            if metric.get_name() == self.__metric_name
        ]
        assert (
            len(metric_index) != 0
        ), f"Metric name {self.__metric_name} for conditional saving doesn't exist within defined metrics."
        current_evaluation = model_stats[metric_index[0]].get_value()

        # Guard for correct objective prediction
        is_greater_than_best = current_evaluation > self.__best_evaluation
        is_maximising = self.__objective == "maximize"
        if (not (is_greater_than_best != is_maximising)) or abs(
            self.__best_evaluation
        ) == float("inf"):

            self.__best_evaluation = current_evaluation
            checkpoint_filename = save_model(
                self.__checkpoints_dir,
                f"e{epoch:03d}_{self.__experiment_name}",
                model.state_dict(),
            )
            self.__checkpoint_files.append(checkpoint_filename)
            for ckpt_file in self.__checkpoint_files[:-1]:
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
            self.__checkpoint_files = self.__checkpoint_files[-1:]
    
    def get_best_checkpoint(self) -> str | None:
        return self.__checkpoint_files[0] if self.__checkpoint_files else None


def hamming_encode(bits: List[bool]) -> List[bool]:
    """
    Computes and add to the input the Hamming correction bits.

    :param bits: A list of floats, that represents a list of bits.
    :return: The original list of bit with the Hamming correciton bits correctly placed.
    """
    m = len(bits)
    r = 0

    # Determine the number of parity bits needed
    while (2**r) < (m + r + 1):
        r += 1

    # Create an array with space for parity bits
    encoded_length = m + r
    encoded = [False] * encoded_length

    # Place data bits into non-parity positions
    j = 0
    for i in range(1, encoded_length + 1):
        if i & (i - 1) == 0:  # Powers of two are parity bits
            continue
        encoded[i - 1] = bits[j]
        j += 1

    # Calculate parity bits
    for i in range(r):
        parity_pos = 2**i
        parity_value = False

        for j in range(1, encoded_length + 1):
            if j & parity_pos:
                parity_value ^= encoded[j - 1]

        encoded[parity_pos - 1] = parity_value

    return encoded


def set_seed(seed):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For current GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs (if you use multi-GPU)
    np.random.seed(seed)
    random.seed(seed)


def load_weights(file_path: str, device):
    return torch.load(file_path, map_location=torch.device(device), weights_only=True)
