import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.teacher_model.eval()
        student_preds = model(**inputs)
        logits_student = student_preds.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        softmax_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)

        soft_loss = F.kl_div(
            F.log_softmax(logits_student / self.temperature, dim=-1),
            softmax_teacher,
            reduction="batchmean",
        ) * (self.temperature**2)

        labels = inputs["labels"]
        hard_loss = F.cross_entropy(
            logits_student.view(-1, logits_student.size(-1)),
            labels.view(-1),
            reduction="mean",
        )

        total_loss = soft_loss + hard_loss
        return (total_loss, student_preds) if return_outputs else total_loss
