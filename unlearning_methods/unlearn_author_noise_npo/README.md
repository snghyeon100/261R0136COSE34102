# unlearn_author_noise_npo

Author-name-noise NPO for TOFU `forget01`.

This method keeps the baseline NPO objective and adds a second forget term where
Gaussian noise is injected only into the input embeddings of the two forget01
author names:

- `Basil Mahfouz Al-Kuwaiti`
- `Nikolai Abilov`

For TOFU `forget01`, the author name appears in 38 of 40 forget questions. The
remaining two examples use clean NPO only.

## Objective

```text
L =
  gamma * (
      clean_npo_weight * NPO(clean forget)
    + noisy_npo_weight * NPO(author-noised forget)
  )
+ alpha * RetainCE
```

The reference model is evaluated on clean forget inputs. Noise is applied only
to the current model's forget forward pass.

By default this runner trains a LoRA adapter (`v_proj`, `o_proj`) because full
fine-tuning Qwen 2B does not fit reliably on 12GB GPUs.

## Run

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_author_noise_npo/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./finetuned/author_noise_npo_2e-05_forget01_5_en
```

Useful smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_author_noise_npo/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./finetuned/author_noise_npo_smoke \
  max_steps=3 \
  batch_size=2
```

Start with `noise_sigma=0.1`; useful ablations are `0.05`, `0.1`, and `0.2`.

## Merge For Evaluation

The default output is a PEFT adapter. Merge before running `evaluate_util.py`:

```bash
python unlearning_methods/unlearn_sh/merge_lora.py \
  --base_model_path ./finetuned/finetuned_100 \
  --adapter_path ./finetuned/author_noise_npo_2e-05_forget01_5_en \
  --output_path ./finetuned/author_noise_npo_2e-05_forget01_5_en/merged \
  --model_family qwen3_5_2b
```
