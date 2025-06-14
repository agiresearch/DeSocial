import os
import torch
import torch.nn as nn
import logging

class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, model_name: str = None):
        """
            Early stop strategy.
            Input:
                patience: int, max patience
                save_model_folder: str, save model folder
                save_model_name: str, save model name
                logger: Logger
                model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pt")
        self.model_name = model_name

    def step(self, metrics: list, model: nn.Module):
        """
            Execute the early stop strategy for each evaluation process
            Input:
                metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
                model: nn.Module
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
            saves model at self.save_model_path
            Input:
                model: nn.Module
        """
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
            load model at self.save_model_path
            Input:
                model: nn.Module
        """
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path))
