# Russian Inclusion Transfer Experiment

This report documents a small-GPU autoresearch-style experiment testing whether
including Russian in a multilingual pretraining mix helps related European
languages. The short answer from this proxy is yes: adding Russian improved the
Slavic target-language average even when the total token budget was held fixed.

The experiment was designed to answer a practical data-mix question, not a
political one:

> If the training objective is to build a stronger multilingual model, does
> excluding Russian leave useful linguistic signal on the table?

## Result Summary

Lower BPB is better. The target average is over Bulgarian, Ukrainian,
Macedonian, Serbian, Polish, Czech, Slovak, Slovene, Croatian, and Bosnian.

| condition | target avg BPB | delta vs no Russian | relative delta | train tokens | peak GPU |
|---|---:|---:|---:|---:|---:|
| No Russian | 3.0576 | 0.0000 | 0.00% | 19.7M | 1.55 GB |
| Russian, same total budget | 3.0444 | -0.0131 | -0.43% | 19.7M | 1.55 GB |
| Russian, extra budget | 3.0388 | -0.0187 | -0.61% | 20.0M | 1.55 GB |

The same-budget condition is the most important comparison. It means Russian
was added while keeping the total number of training tokens constant, so the
improvement is not just from seeing more data.

## What Exactly Was Evaluated?

This is the most important scope point: the evaluation is **language-model
validation loss on held-out proxy text**, not a downstream benchmark.

For each available language, the script streamed text from FinePDFs-Edu
one-shard samples and built:

```text
12 MB training bytes per language
 2 MB held-out validation bytes per language
```

During evaluation, the model is given random 256-byte chunks from the validation
split and is scored on how well it predicts the next byte. The score is BPB
(bits per byte). Lower BPB means better prediction.

More concretely, the evaluation loop is:

1. Pick one language, such as Ukrainian or Bulgarian.
2. Sample batches of 256-byte chunks from that language's 2 MB validation split.
3. Feed bytes `1..255` to the model.
4. Ask the model to predict bytes `2..256` from the same held-out text.
5. Convert the prediction loss to bits per byte.
6. Average that score over 40 evaluation batches for the language.

In pseudocode:

```python
for language in eval_languages:
    chunks = sample(validation_bytes[language])
    prediction = model(chunks[:, :-1])
    target = chunks[:, 1:]
    bpb[language] = cross_entropy(prediction, target) / log(2)
```

There are no task labels, prompts, multiple-choice questions, or human
judgments in this evaluation. The "label" is simply the next byte from the same
held-out text. The eval asks: given this language's validation text, how
surprised is the model by the next byte?

So the question answered here is:

> After training with or without Russian, does the model predict held-out text in
> related languages better?

In this proxy, yes. The Slavic target average improved when Russian was
included.

This report does **not** claim direct improvement on question answering,
translation, instruction following, factual reasoning, or other downstream
benchmarks. It is a cheap transfer signal that says Russian is worth including
and testing more seriously in the real training setup.

## Strongest Signals

Russian itself improved substantially, which is a sanity check that the
condition is working:

| language | no Russian | Russian, same budget | delta |
|---|---:|---:|---:|
| Russian | 2.4379 | 2.2929 | -0.1450 |
| Ukrainian | 2.5491 | 2.4974 | -0.0517 |
| Bulgarian | 2.2334 | 2.1921 | -0.0414 |
| Macedonian | 2.3048 | 2.2719 | -0.0329 |
| Serbian | 2.3494 | 2.3213 | -0.0281 |

The largest target-language gains are in Cyrillic Slavic languages. That is
exactly the effect we would expect if Russian contributes useful script,
morphology, vocabulary, and byte/subword regularities that overlap with nearby
languages.

The result is not uniformly positive for every language. Some Latin-script
Slavic languages move by only a few thousandths of a BPB, and a few become
slightly worse under the same total budget because adding Russian dilutes their
own sampling mass. That trade-off is precisely why the same-budget arm matters.

## Why This Supports Including Russian

This experiment does not prove that every downstream benchmark will improve.
It does provide a useful technical signal:

1. Russian improved the Slavic target average under a fixed token budget.
2. The gains are concentrated where linguistic transfer is plausible.
3. The extra-budget condition improved more broadly, suggesting that Russian is
   useful additional data rather than only a reweighting artifact.
4. The cost in this proxy is small: Russian was added at the same collapsed mass
   as Ukrainian, about 1.52% of the local language-coded mix.

For a multilingual model, excluding Russian should therefore be treated as an
active trade-off that needs evidence. The default technical prior should be to
include it, then measure policy, safety, tokenizer, and data-quality concerns
explicitly.

## Experimental Design

The provided production-like allocation table contains many source rows and many
language-coded rows, for example:

- `finepdfs-1.0.0/megatron-lm/ukr_Cyrl`
- `hplt-3.0/megatron-lm/bul_Cyrl`
- `nemotron-cc-opus-1.1/megatron-lm/srp_Cyrl`
- `nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/pol_Latn`

The experiment parser extracts the target `m tokens` count from every row with a
language suffix of the form `xxx_Script`, collapses those counts by local
language code, and normalizes them into sampling weights.

Rows without a language suffix, such as generic CLM, StarCoder, math, OLMo, and
generic Nemotron rows, are not used as language sampling weights in this proxy.
They are real production-mix mass, but they do not answer the narrow question
"what happens if we add a Russian language slice?"

Russian does not appear in the supplied allocation table. For the comparison,
`rus_Cyrl` was inserted at the same collapsed target mass as Ukrainian:

```text
russian_target_m = 44,680m
```

The local language-coded base mix covered 33 languages:

```text
bg, bs, ca, cs, da, de, el, en, es, et, eu, fi, fr, gl, hr, hu,
is, it, lt, lv, mk, mt, nl, no, pl, pt, ro, sk, sl, sr, sv, tr, uk
```

Russian was then added as the 34th evaluation language.

Unavailable in the local proxy data:

```text
als_Latn, gle_Latn, kat_Geor, nno_Latn, sqi_Latn
```

## Conditions

| condition | meaning |
|---|---|
| `base_no_ru` | Original collapsed language-coded mix, without Russian. |
| `with_ru_same_total` | Russian added, all weights renormalized, same training steps. |
| `with_ru_extra_budget` | Russian added, and training steps scaled by total mix growth. |

The same-budget arm tests whether Russian is valuable enough to survive
competition for fixed training budget. The extra-budget arm tests whether
Russian is useful as additional data.

## Model And Compute

The model is deliberately small so it can run repeatedly on the available L4
GPU:

| parameter | value |
|---|---:|
| architecture | byte-level causal Transformer |
| layers | 6 |
| heads | 6 |
| embedding dim | 384 |
| sequence length | 256 bytes |
| batch size | 64 |
| optimizer | AdamW |
| seeds | 1, 2 |
| base steps | 1,200 |
| extra-budget steps | 1,219 |
| eval batches/language | 40 |
| peak memory | 1.55 GB |

Each condition sees about 20M byte tokens. This is tiny compared with a real
pretraining run, but sufficient as a cheap transfer probe.

## Data Pipeline

The GPU box did not contain the full production corpora, so the experiment uses
FinePDFs-Edu one-shard samples as proxy text. For each available language, the
script streams parquet batches and builds:

```text
12 MB train bytes
 2 MB validation bytes
```

This keeps the experiment reproducible and avoids pulling hundreds of gigabytes.
The full FinePDFs-Edu language sample estimate was about 227 GB, which would not
fit comfortably on the machine. The actual one-shard proxy corpus is about 31
GB, leaving about 49 GB free after the run.

## Reproduction

The experiment was run on the GPU box from:

```bash
cd /home/ubuntu/birger/russian-transfer-experiment
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/birger/autoresearch/.venv/bin/python \
  mix_matched_russian_experiment.py \
  --data-root /home/ubuntu/birger/russian-transfer-experiment-data/raw \
  --work-dir /home/ubuntu/birger/russian-transfer-experiment/mix_matched_full_20260529 \
  --train-bytes 12000000 \
  --val-bytes 2000000 \
  --seq-len 256 \
  --batch-size 64 \
  --steps 1200 \
  --eval-batches 40 \
  --seeds 1,2 \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384
```

The script writes:

- `REPORT.md`
- `summary.json`
- `results.json`
- `mixture_metadata.json`

## Interpretation

The experiment supports including Russian as a technical default in multilingual
pretraining, especially when Ukrainian, Bulgarian, Macedonian, Serbian, and
other Slavic/Cyrillic languages are in scope.

The important nuance is that fixed-budget multilingual mixtures always involve
trade-offs. Adding any language reduces the sampling mass of others unless the
total budget grows. This run says Russian appears worth testing seriously, not
that it is free or that the exact production ratio is solved.

The next stronger version of this experiment would use the exact production
tokenizer and stream from the actual Megatron/HPLT/Nemotron shards, then evaluate
on downstream tasks such as Ukrainian, Bulgarian, Serbian, Macedonian, and
cross-lingual knowledge transfer benchmarks.
