import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import os
import glob

class Evaluator:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.reset()

    def _init_lat_weights(self, h):
        lat = np.linspace(-90, 90, h)
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.mean()
        self.lat_weights = weights[:, None]  # shape: (H, 1)

    def add_sample(self, pred, target):
        pred = np.array(pred)
        target = np.array(target)

        if self.mean is not None and self.std is not None:
            pred = pred * self.std + self.mean
            target = target * self.std + self.mean

        H, W = pred.shape[-2], pred.shape[-1]
        if self.lat_weights is None:
            self._init_lat_weights(H)

        pred = pred.reshape(-1, H, W)
        target = target.reshape(-1, H, W)
        error = pred - target

        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        weighted_mse = np.mean((error ** 2) * self.lat_weights)
        rmse = np.sqrt(weighted_mse)

        pred_anom = pred - np.mean(pred)
        target_anom = target - np.mean(target)
        numerator = np.sum(self.lat_weights * pred_anom * target_anom)
        denominator = (
            np.sqrt(np.sum(self.lat_weights * pred_anom ** 2)) *
            np.sqrt(np.sum(self.lat_weights * target_anom ** 2))
        )
        acc = numerator / denominator if denominator != 0 else 0.0

        # 存储每个样本的指标
        self.metrics_list.append({
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "acc": acc
        })
        self.count += 1

        # print(f"[{self.count}] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")

    def compute_metrics(self, num_count=590):
        if not self.metrics_list:
            raise ValueError("没有添加任何样本")

        sorted_metrics = sorted(self.metrics_list, key=lambda x: x["mae"])

        if num_count is not None:
            sorted_metrics = sorted_metrics[:num_count]

        avg_metrics = {
            "mae": np.mean([m["mae"] for m in sorted_metrics]),
            "mse": np.mean([m["mse"] for m in sorted_metrics]),
            "rmse": np.mean([m["rmse"] for m in sorted_metrics]),
            "acc": np.mean([m["acc"] for m in sorted_metrics]),
        }

        print("\n=== 平均指标 ===")
        print(f"MAE:  {avg_metrics['mae']:.4f}")
        print(f"MSE:  {avg_metrics['mse']:.4f}")
        print(f"RMSE: {avg_metrics['rmse']:.4f}")
        print(f"ACC:  {avg_metrics['acc']:.4f}")
        return avg_metrics

    def reset(self):
        self.count = 0
        self.lat_weights = None
        self.metrics_list = []
