import math
import torch
import torch.nn as nn
import torch.optim as optim


class BaseLRScheduler:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.current_step = 0

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_step += 1


class LinearDecayLR(BaseLRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        super().__init__(optimizer, total_steps)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def get_lr(self):
        ratio = min(self.current_step / self.total_steps, 1.0)
        return self.start_lr + (self.end_lr - self.start_lr) * ratio


class ExponentialDecayLR(BaseLRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        super().__init__(optimizer, total_steps)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.k = math.log(end_lr / start_lr) / total_steps

    def get_lr(self):
        return self.start_lr * math.exp(self.k * self.current_step)


class CosineDecayLR(BaseLRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        super().__init__(optimizer, total_steps)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def get_lr(self):
        ratio = min(self.current_step / self.total_steps, 1.0)
        cos_decay = 0.5 * (1 + math.cos(math.pi * ratio))
        return self.end_lr + (self.start_lr - self.end_lr) * cos_decay


class PiecewiseLinearLR(BaseLRScheduler):
    def __init__(self, optimizer, milestones: dict, total_steps):
        """
        milestones: dict of step -> lr
        Example: {0: 0.01, 50: 0.005, 100: 0.001}
        """
        super().__init__(optimizer, total_steps)
        self.points = sorted(milestones.items())

    def get_lr(self):
        if self.current_step >= self.points[-1][0]:
            return self.points[-1][1]
        for i in range(len(self.points) - 1):
            start_step, start_lr = self.points[i]
            end_step, end_lr = self.points[i + 1]
            if start_step <= self.current_step < end_step:
                t = (self.current_step - start_step) / (end_step - start_step)
                return start_lr + t * (end_lr - start_lr)
        return self.points[0][1]


class WarmupLinearLR(BaseLRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, peak_lr, end_lr, total_steps):
        super().__init__(optimizer, total_steps)
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.start_lr + (self.peak_lr - self.start_lr) * (self.current_step / self.warmup_steps)
        else:
            decay_steps = self.total_steps - self.warmup_steps
            step_in_decay = self.current_step - self.warmup_steps
            ratio = min(step_in_decay / decay_steps, 1.0)
            return self.peak_lr + (self.end_lr - self.peak_lr) * ratio
