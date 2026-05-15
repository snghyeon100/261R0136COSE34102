"""Losses for UNLEARN task-matrix training."""

import torch
import torch.nn.functional as F

from unlearning_methods.unlearn_UNLEARN.low_rank import task_matrix_norms, task_parameter_count


def model_device(model):
    return next(model.parameters()).device


def batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def zero_from_logits(logits):
    return logits.sum() * 0.0


def shift_logits_and_labels(logits, labels):
    return logits[..., :-1, :].float().contiguous(), labels[..., 1:].contiguous()


def answer_ce_loss(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits, shift_labels = shift_logits_and_labels(outputs.logits, labels)
    mask = shift_labels.ne(-100)
    if not mask.any():
        return zero_from_logits(shift_logits), outputs

    token_ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.clamp_min(0).view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return token_ce[mask].mean(), outputs


def retain_kl_loss(model, teacher_model, inputs):
    if teacher_model is None:
        raise ValueError("retain_kl_loss requires a teacher model.")

    student_device = model_device(model)
    teacher_device = model_device(teacher_model)
    input_ids, labels, attention_mask = batch_to_device(inputs, student_device)
    student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    student_logits, shift_labels = shift_logits_and_labels(student_outputs.logits, labels)
    mask = shift_labels.ne(-100)
    if not mask.any():
        return zero_from_logits(student_logits), student_outputs

    with torch.no_grad():
        teacher_input_ids, teacher_labels, teacher_attention_mask = batch_to_device(inputs, teacher_device)
        teacher_outputs = teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        )
        teacher_logits, teacher_shift_labels = shift_logits_and_labels(teacher_outputs.logits, teacher_labels)
        teacher_mask = teacher_shift_labels.ne(-100)
        teacher_probs = F.softmax(teacher_logits[teacher_mask], dim=-1).to(student_device)

    student_log_probs = F.log_softmax(student_logits[mask], dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return token_kl.mean(), student_outputs


def compute_unlearn_loss(model, teacher_model, batch, cfg):
    forget_loss, forget_outputs = answer_ce_loss(model, batch["forget"])
    retain_loss = forget_loss.detach() * 0.0
    if bool(cfg.use_retain_regularization) and batch.get("retain") is not None:
        retain_loss, _ = retain_kl_loss(model, teacher_model, batch["retain"])

    total = forget_loss + float(cfg.retain_weight) * retain_loss
    norm_mean, norm_max = task_matrix_norms(model)
    if norm_mean is None:
        norm_mean = total.detach() * 0.0
        norm_max = total.detach() * 0.0

    logs = {
        "loss_total": total.detach(),
        "loss_forget": forget_loss.detach(),
        "loss_retain": retain_loss.detach(),
        "task_matrix_norm_mean": norm_mean.detach(),
        "task_matrix_norm_max": norm_max.detach(),
        "wrapped_param_count": torch.tensor(float(task_parameter_count(model)), device=total.device),
    }
    return total, forget_outputs, logs

