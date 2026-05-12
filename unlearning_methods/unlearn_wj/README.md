# PCE-ICUL: Paired Counterfactual Equivalence In-Context Unlearning

This method does not train or save a new checkpoint. The evaluated model is the
original fine-tuned model plus an inference-time context.

The method gives the model:

1. an English-only perturb-aligned counterfactual forget case,
2. an equivalence bridge saying the final target-language question asks the same fact,
3. retain anchors showing that unrelated questions should still be answered normally,
4. a retrieval gate that avoids adding unlearning context to non-forget queries.

For a Korean forget query `q_ko_i`, the context uses the paired English forget
question `q_en_i` and an English perturb answer from the same forget index. It
does not provide a Korean forget answer in context.

By default, `retrieval_gate_mode=eval_task`, which is an evaluation-time oracle
gate: PCE context is applied to `eval_log_forget` and normal prompting is used
for retain / real-author / real-world tasks. For a stricter non-oracle gate, use
`retrieval_gate_mode=lexical`.

The evaluator preserves the root `evaluate_util.py` metric path. Method-local
code only injects context before the final query and returns batches shaped as:

```python
(input_ids, labels, attention_mask, indices)
```

## Run

From the `NLP_rice` root:

```bash
python unlearning_methods/unlearn_wj/evaluate.py
```

Minimal Korean run:

```bash
python unlearning_methods/unlearn_wj/evaluate.py \
  'conditions=[pce_icul]' \
  'languages=[ko]' \
  model_path=./finetuned/epoch5
```

Run with an explicit no-context baseline:

```bash
python unlearning_methods/unlearn_wj/evaluate.py \
  'conditions=[no_context,pce_icul]' \
  'languages=[ko]' \
  model_path=./finetuned/epoch5
```

If the English TOFU perturb split is not cached locally, either make
`locuslab/TOFU` available or point the config to a local English perturbed split:

```bash
python unlearning_methods/unlearn_wj/evaluate.py \
  'conditions=[pce_icul]' \
  'languages=[ko]' \
  source_perturbed_forget_data_path=./dataset/forget01_perturbed_en
```

Fallback to the older profile-swap counterfactual:

```bash
python unlearning_methods/unlearn_wj/evaluate.py \
  'conditions=[pce_icul]' \
  'languages=[ko]' \
  counterfactual_answer_policy=source_slot_swap
```

The default output root is:

```text
outputs/unlearn_wj/pce_icul_en_to_ko/
```

Each condition writes the same summary files as `evaluate_util.py`, including
`eval_summary.json` and `eval_summary.csv`. Raw logs are enabled by default, so
`eval_log.json`, `eval_log_forget.json`, and related task logs are also saved.
