"""Author-name-noise NPO loss for forget/retain paired batches."""

import torch
import torch.nn.functional as F
from torch import nn


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def compute_batch_nll(model, inputs):
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(inputs, device)
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
    loss = loss.view(shifted_labels.size()).sum(dim=-1)
    return loss, outputs


def compute_batch_nll_with_author_noise(model, inputs, trigger_mask, noise_sigma):
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(inputs, device)
    trigger_mask = trigger_mask.to(device=device, dtype=torch.bool)

    embeds = model.get_input_embeddings()(input_ids)
    if trigger_mask.any() and noise_sigma > 0:
        scale = embeds.detach().float().std().to(dtype=embeds.dtype, device=device)
        noise = torch.randn_like(embeds) * scale * float(noise_sigma)
        embeds = embeds.clone()
        embeds[trigger_mask] = embeds[trigger_mask] + noise[trigger_mask]

    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
    loss = loss.view(shifted_labels.size()).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if ref_model is None:
        raise ValueError("NPO requires an oracle/reference model.")
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    device = _model_device(model)
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss.to(device))

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss.to(device))

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_dpo_loss_with_ref_nll(model, lose_inputs, ref_nll, beta=1.0):
    device = _model_device(model)
    lose_nll, outputs = compute_batch_nll(model, lose_inputs)
    ref_nll = ref_nll.to(device=device, dtype=lose_nll.dtype)
    lose_log_ratio = -(lose_nll - ref_nll)
    loss = -2 / beta * F.logsigmoid(beta * (0.0 - lose_log_ratio)).mean()
    return loss, outputs


def compute_noisy_forget_npo_loss(model, ref_model, forget_inputs, trigger_mask, has_trigger, beta=1.0, noise_sigma=0.1):
    if ref_model is None:
        raise ValueError("Noisy NPO requires an oracle/reference model.")

    device = _model_device(model)
    has_trigger = has_trigger.to(device=device, dtype=torch.bool)
    if not has_trigger.any():
        zero = next(model.parameters()).sum() * 0.0
        return zero, None, zero.detach()

    noisy_nll, noisy_outputs = compute_batch_nll_with_author_noise(
        model,
        forget_inputs,
        trigger_mask=trigger_mask,
        noise_sigma=noise_sigma,
    )
    with torch.no_grad():
        ref_nll, _ = compute_batch_nll(ref_model, forget_inputs)

    lose_log_ratio = -(noisy_nll - ref_nll.to(device))
    per_example_loss = -2 / beta * F.logsigmoid(beta * (0.0 - lose_log_ratio))
    valid_loss = per_example_loss[has_trigger].mean()
    valid_ratio = has_trigger.float().mean().detach()
    return valid_loss, noisy_outputs, valid_ratio


def compute_noisy_forget_npo_loss_with_ref_nll(
    model,
    forget_inputs,
    trigger_mask,
    has_trigger,
    ref_nll,
    beta=1.0,
    noise_sigma=0.1,
):
    device = _model_device(model)
    has_trigger = has_trigger.to(device=device, dtype=torch.bool)
    if not has_trigger.any():
        zero = next(model.parameters()).sum() * 0.0
        return zero, None, zero.detach()

    noisy_nll, noisy_outputs = compute_batch_nll_with_author_noise(
        model,
        forget_inputs,
        trigger_mask=trigger_mask,
        noise_sigma=noise_sigma,
    )
    ref_nll = ref_nll.to(device=device, dtype=noisy_nll.dtype)
    lose_log_ratio = -(noisy_nll - ref_nll)
    per_example_loss = -2 / beta * F.logsigmoid(beta * (0.0 - lose_log_ratio))
    valid_loss = per_example_loss[has_trigger].mean()
    valid_ratio = has_trigger.float().mean().detach()
    return valid_loss, noisy_outputs, valid_ratio


def compute_author_noise_npo_loss(
    model,
    oracle_model,
    batch,
    beta=1.0,
    gamma=1.0,
    alpha=1.0,
    clean_npo_weight=1.0,
    noisy_npo_weight=1.0,
    noise_sigma=0.1,
):
    forget_inputs = batch["forget"]
    retain_inputs = batch["retain"]

    has_precomputed_ref = "reference_nll" in batch and torch.isfinite(batch["reference_nll"]).all()
    if has_precomputed_ref:
        clean_forget_loss, forget_outputs = compute_dpo_loss_with_ref_nll(
            model=model,
            lose_inputs=forget_inputs,
            ref_nll=batch["reference_nll"],
            beta=beta,
        )
        noisy_forget_loss, _, trigger_ratio = compute_noisy_forget_npo_loss_with_ref_nll(
            model=model,
            forget_inputs=forget_inputs,
            trigger_mask=batch["trigger_mask"],
            has_trigger=batch["has_trigger"],
            ref_nll=batch["reference_nll"],
            beta=beta,
            noise_sigma=noise_sigma,
        )
    else:
        clean_forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=oracle_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=beta,
        )
        noisy_forget_loss, _, trigger_ratio = compute_noisy_forget_npo_loss(
            model=model,
            ref_model=oracle_model,
            forget_inputs=forget_inputs,
            trigger_mask=batch["trigger_mask"],
            has_trigger=batch["has_trigger"],
            beta=beta,
            noise_sigma=noise_sigma,
        )

    retain_input_ids, retain_labels, retain_attention_mask = _batch_to_device(retain_inputs, _model_device(model))
    retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    weighted_forget = float(clean_npo_weight) * clean_forget_loss + float(noisy_npo_weight) * noisy_forget_loss
    total = float(gamma) * weighted_forget + float(alpha) * retain_outputs.loss
    logs = {
        "loss_total": total.detach(),
        "loss_clean_npo": clean_forget_loss.detach(),
        "loss_noisy_npo": noisy_forget_loss.detach(),
        "loss_retain": retain_outputs.loss.detach(),
        "trigger_batch_ratio": trigger_ratio.detach(),
    }
    return total, forget_outputs, logs
