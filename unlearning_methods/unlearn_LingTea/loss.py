"""LingTea-style forget and adaptive retain losses."""

import torch
import torch.nn.functional as F


def model_device(model):
    return next(model.parameters()).device


def batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _zero_from_logits(logits):
    return logits.sum() * 0.0


def _shift_logits_and_labels(logits, labels):
    return logits[..., :-1, :].float().contiguous(), labels[..., 1:].contiguous()


def answer_ce_loss(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits, shift_labels = _shift_logits_and_labels(outputs.logits, labels)
    mask = shift_labels.ne(-100)
    if not mask.any():
        zero = _zero_from_logits(shift_logits)
        return zero, outputs

    token_ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.clamp_min(0).view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return token_ce[mask].mean(), outputs


def negative_forget_loss(model, inputs):
    ce_loss, outputs = answer_ce_loss(model, inputs)
    return -ce_loss, outputs, {"forget_ce": ce_loss.detach()}


def adaptive_retain_loss(model, teacher_model, inputs, temperature=1.0):
    if teacher_model is None:
        raise ValueError("LingTea retain loss requires a frozen teacher/reference model.")

    student_device = model_device(model)
    teacher_device = model_device(teacher_model)
    input_ids, labels, attention_mask = batch_to_device(inputs, student_device)

    student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    student_logits, shift_labels = _shift_logits_and_labels(student_outputs.logits, labels)
    mask = shift_labels.ne(-100)
    if not mask.any():
        zero = _zero_from_logits(student_logits)
        return zero, student_outputs, {
            "retain_kd": zero.detach(),
            "retain_ce": zero.detach(),
            "retain_kd_raw": zero.detach(),
            "retain_ce_raw": zero.detach(),
            "teacher_confidence_mean": zero.detach(),
        }

    valid_student_logits = student_logits[mask]
    valid_labels = shift_labels[mask].clamp_min(0)

    with torch.no_grad():
        teacher_input_ids, teacher_labels, teacher_attention_mask = batch_to_device(inputs, teacher_device)
        teacher_outputs = teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        )
        teacher_logits, teacher_shift_labels = _shift_logits_and_labels(teacher_outputs.logits, teacher_labels)
        teacher_mask = teacher_shift_labels.ne(-100)
        valid_teacher_logits = teacher_logits[teacher_mask]
        valid_teacher_labels = teacher_shift_labels[teacher_mask].clamp_min(0)

        teacher_probs_t = F.softmax(valid_teacher_logits / temperature, dim=-1).to(student_device)
        teacher_confidence = F.softmax(valid_teacher_logits, dim=-1).gather(
            dim=-1,
            index=valid_teacher_labels.unsqueeze(-1),
        ).squeeze(-1).to(student_device)
        teacher_confidence = teacher_confidence.clamp(0.0, 1.0)

    student_log_probs_t = F.log_softmax(valid_student_logits / temperature, dim=-1)
    kd_per_token = F.kl_div(
        student_log_probs_t,
        teacher_probs_t,
        reduction="none",
    ).sum(dim=-1) * (temperature ** 2)
    ce_per_token = F.cross_entropy(valid_student_logits, valid_labels, reduction="none")

    weighted_kd = teacher_confidence * kd_per_token
    weighted_ce = (1.0 - teacher_confidence) * ce_per_token
    loss = (weighted_kd + weighted_ce).mean()

    return loss, student_outputs, {
        "retain_kd": weighted_kd.detach().mean(),
        "retain_ce": weighted_ce.detach().mean(),
        "retain_kd_raw": kd_per_token.detach().mean(),
        "retain_ce_raw": ce_per_token.detach().mean(),
        "teacher_confidence_mean": teacher_confidence.detach().mean(),
    }


def compute_lingtea_loss(model, teacher_model, batch, cfg):
    forget_loss, forget_outputs, forget_logs = negative_forget_loss(model, batch["forget_source"])
    retain_loss, _, retain_logs = adaptive_retain_loss(
        model,
        teacher_model,
        batch["retain"],
        temperature=float(cfg.temperature),
    )

    weights = cfg.loss_weights
    total = float(weights.forget) * forget_loss + float(weights.retain) * retain_loss
    logs = {
        "loss_total": total.detach(),
        "loss_forget": forget_loss.detach(),
        "loss_retain": retain_loss.detach(),
        **forget_logs,
        **retain_logs,
    }
    return total, forget_outputs, logs
