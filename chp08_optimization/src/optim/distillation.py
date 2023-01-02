# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import numpy as np
import torch
from transformers import Trainer, TrainingArguments


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(
        self, model: torch.nn.Module, inputs: torch.Tensor, return_outputs: bool = False
    ) -> torch.tensor:
        """overwrites the default Trainer compute_loss method.
        loss is calculated as a \alpha * Loss(cross-entropy) + (1 - \alpha) * Loss(Kullbac-Leibler divergence)
        """
        # compute logits from student learner
        outputs_stu = model(**inputs)
        logits_stu = outputs_stu.logits

        # compute logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits

        # get cross entropy loss from student
        ce_loss = outputs_stu.loss

        # calculate KD loss between student and teacher
        kd_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        kd_loss = self.args.temperature**2 * kd_loss_fct(
            torch.nn.functional.log_softmax(logits_stu, dim=-1),
            torch.nn.functional.softmax(logits_tea, dim=-1),
        )

        # return weighted student loss
        loss = self.args.alpha * ce_loss + (1 - self.args.alpha) * kd_loss

        return (loss, outputs_stu) if return_outputs else loss
