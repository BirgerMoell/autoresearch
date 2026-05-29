#!/usr/bin/env python3
"""Mix-matched Russian inclusion pilot.

This is a small-GPU proxy for the question:

    Given the proposed language/source token allocation, what happens if we add
    Russian versus leave it out?

The script parses a token-allocation table like:

    finepdfs-edu-1.0.0/megatron-lm/ukr_Cyrl: 4,780m tokens ...

It collapses target token counts by language code and uses those collapsed counts
as sampling weights. Since the L4 box has small raw FinePDFs-Edu samples rather
than the full Megatron mixture, this trains a byte-level LM on local proxy text
while matching the proposed language weights as closely as available data allows.

Russian is not in the provided allocation. By default `rus_Cyrl` is added with
the same collapsed target mass as `ukr_Cyrl`; override with --russian-target-m.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


MIX_TEXT = r"""
clm-1.0/megatron-lm: 3,240,000m tokens of 3,845,395m (84.3%).
finemath-0.0.0/megatron-lm/finemath-4plus: 40,000m tokens of 10,684m (374.4%).
finepdfs-1.0.0/megatron-lm/als_Latn: 6,110m tokens of 2,353m (259.7%).
finepdfs-1.0.0/megatron-lm/bos_Latn: 3,700m tokens of 5,627m (65.8%).
finepdfs-1.0.0/megatron-lm/bul_Cyrl: 4,910m tokens of 7,943m (61.8%).
finepdfs-1.0.0/megatron-lm/cat_Latn: 4,360m tokens of 11,084m (39.3%).
finepdfs-1.0.0/megatron-lm/ces_Latn: 2,840m tokens of 25,735m (11.0%).
finepdfs-1.0.0/megatron-lm/dan_Latn: 3,170m tokens of 11,723m (27.0%).
finepdfs-1.0.0/megatron-lm/deu_Latn: 150m tokens of 154,018m (0.1%).
finepdfs-1.0.0/megatron-lm/ekk_Latn: 4,200m tokens of 2,912m (144.2%).
finepdfs-1.0.0/megatron-lm/ell_Grek: 7,700m tokens of 16,495m (46.7%).
finepdfs-1.0.0/megatron-lm/eng_Latn: 1,063,760m tokens of 1,316,304m (80.8%).
finepdfs-1.0.0/megatron-lm/eus_Latn: 3,470m tokens of 2,696m (128.7%).
finepdfs-1.0.0/megatron-lm/fin_Latn: 3,700m tokens of 13,197m (28.0%).
finepdfs-1.0.0/megatron-lm/fra_Latn: 4,840m tokens of 168,589m (2.9%).
finepdfs-1.0.0/megatron-lm/gle_Latn: 3,620m tokens of 300m (1,206.7%).
finepdfs-1.0.0/megatron-lm/glg_Latn: 3,710m tokens of 1,737m (213.6%).
finepdfs-1.0.0/megatron-lm/hrv_Latn: 3,580m tokens of 10,440m (34.3%).
finepdfs-1.0.0/megatron-lm/hun_Latn: 2,800m tokens of 28,761m (9.7%).
finepdfs-1.0.0/megatron-lm/isl_Latn: 3,320m tokens of 3,125m (106.2%).
finepdfs-1.0.0/megatron-lm/ita_Latn: 5,960m tokens of 81,555m (7.3%).
finepdfs-1.0.0/megatron-lm/kat_Geor: 36,070m tokens of 5,335m (676.1%).
finepdfs-1.0.0/megatron-lm/lit_Latn: 3,910m tokens of 5,341m (73.2%).
finepdfs-1.0.0/megatron-lm/lvs_Latn: 5,920m tokens of 4,229m (140.0%).
finepdfs-1.0.0/megatron-lm/mkd_Cyrl: 4,220m tokens of 1,655m (255.0%).
finepdfs-1.0.0/megatron-lm/mlt_Latn: 4,670m tokens of 737m (633.6%).
finepdfs-1.0.0/megatron-lm/nld_Latn: 4,600m tokens of 43,632m (10.5%).
finepdfs-1.0.0/megatron-lm/nno_Latn: 3,150m tokens of 340m (926.5%).
finepdfs-1.0.0/megatron-lm/nob_Latn: 3,080m tokens of 9,974m (30.9%).
finepdfs-1.0.0/megatron-lm/pol_Latn: 2,900m tokens of 41,724m (7.0%).
finepdfs-1.0.0/megatron-lm/por_Latn: 5,280m tokens of 94,299m (5.6%).
finepdfs-1.0.0/megatron-lm/ron_Latn: 3,370m tokens of 20,365m (16.5%).
finepdfs-1.0.0/megatron-lm/slk_Latn: 3,401m tokens of 10,722m (31.7%).
finepdfs-1.0.0/megatron-lm/slv_Latn: 4,140m tokens of 7,058m (58.7%).
finepdfs-1.0.0/megatron-lm/spa_Latn: 4,150m tokens of 214,982m (1.9%).
finepdfs-1.0.0/megatron-lm/srp_Cyrl: 3,410m tokens of 10,096m (33.8%).
finepdfs-1.0.0/megatron-lm/swe_Latn: 3,310m tokens of 21,917m (15.1%).
finepdfs-1.0.0/megatron-lm/tur_Latn: 6,390m tokens of 15,429m (41.4%).
finepdfs-1.0.0/megatron-lm/ukr_Cyrl: 2,370m tokens of 24,556m (9.7%).
finepdfs-edu-1.0.0/megatron-lm/als_Latn: 470m tokens of 472m (99.6%).
finepdfs-edu-1.0.0/megatron-lm/bos_Latn: 980m tokens of 981m (99.9%).
finepdfs-edu-1.0.0/megatron-lm/bul_Cyrl: 1,250m tokens of 1,245m (100.4%).
finepdfs-edu-1.0.0/megatron-lm/cat_Latn: 1,150m tokens of 1,148m (100.2%).
finepdfs-edu-1.0.0/megatron-lm/ces_Latn: 4,080m tokens of 4,076m (100.1%).
finepdfs-edu-1.0.0/megatron-lm/dan_Latn: 1,690m tokens of 1,685m (100.3%).
finepdfs-edu-1.0.0/megatron-lm/deu_Latn: 16,810m tokens of 16,814m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/ekk_Latn: 410m tokens of 406m (101.0%).
finepdfs-edu-1.0.0/megatron-lm/ell_Grek: 2,340m tokens of 2,338m (100.1%).
finepdfs-edu-1.0.0/megatron-lm/eng_Latn: 141,840m tokens of 141,839m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/eus_Latn: 300m tokens of 295m (101.7%).
finepdfs-edu-1.0.0/megatron-lm/fin_Latn: 1,960m tokens of 1,960m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/fra_Latn: 17,540m tokens of 17,538m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/glg_Latn: 240m tokens of 235m (102.1%).
finepdfs-edu-1.0.0/megatron-lm/hrv_Latn: 1,630m tokens of 1,631m (99.9%).
finepdfs-edu-1.0.0/megatron-lm/hun_Latn: 4,870m tokens of 4,870m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/isl_Latn: 250m tokens of 249m (100.4%).
finepdfs-edu-1.0.0/megatron-lm/ita_Latn: 10,210m tokens of 10,213m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/kat_Geor: 870m tokens of 872m (99.8%).
finepdfs-edu-1.0.0/megatron-lm/lit_Latn: 880m tokens of 877m (100.3%).
finepdfs-edu-1.0.0/megatron-lm/lvs_Latn: 560m tokens of 556m (100.7%).
finepdfs-edu-1.0.0/megatron-lm/mkd_Cyrl: 270m tokens of 265m (101.9%).
finepdfs-edu-1.0.0/megatron-lm/mlt_Latn: 101m tokens of 96m (105.2%).
finepdfs-edu-1.0.0/megatron-lm/nld_Latn: 4,900m tokens of 4,900m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/nno_Latn: 40m tokens of 41m (97.6%).
finepdfs-edu-1.0.0/megatron-lm/nob_Latn: 1,300m tokens of 1,300m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/pol_Latn: 7,530m tokens of 7,533m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/por_Latn: 12,840m tokens of 12,838m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/ron_Latn: 4,440m tokens of 4,438m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/slk_Latn: 1,840m tokens of 1,842m (99.9%).
finepdfs-edu-1.0.0/megatron-lm/slv_Latn: 620m tokens of 617m (100.5%).
finepdfs-edu-1.0.0/megatron-lm/spa_Latn: 28,470m tokens of 28,472m (100.0%).
finepdfs-edu-1.0.0/megatron-lm/srp_Cyrl: 1,920m tokens of 1,921m (99.9%).
finepdfs-edu-1.0.0/megatron-lm/swe_Latn: 3,270m tokens of 3,274m (99.9%).
finepdfs-edu-1.0.0/megatron-lm/tur_Latn: 2,450m tokens of 2,447m (100.1%).
finepdfs-edu-1.0.0/megatron-lm/ukr_Cyrl: 4,780m tokens of 4,784m (99.9%).
hplt-3.0/megatron-lm/als_Latn: 20,580m tokens of 10,529m (195.5%).
hplt-3.0/megatron-lm/bos_Latn: 14,620m tokens of 27,695m (52.8%).
hplt-3.0/megatron-lm/bul_Cyrl: 19,250m tokens of 42,713m (45.1%).
hplt-3.0/megatron-lm/cat_Latn: 17,250m tokens of 21,035m (82.0%).
hplt-3.0/megatron-lm/ces_Latn: 21,640m tokens of 93,521m (23.1%).
hplt-3.0/megatron-lm/dan_Latn: 15,190m tokens of 55,465m (27.4%).
hplt-3.0/megatron-lm/deu_Latn: 53,000m tokens of 548,693m (9.7%).
hplt-3.0/megatron-lm/ekk_Latn: 14,410m tokens of 17,670m (81.6%).
hplt-3.0/megatron-lm/ell_Grek: 31,380m tokens of 103,562m (30.3%).
hplt-3.0/megatron-lm/eus_Latn: 11,760m tokens of 2,537m (463.5%).
hplt-3.0/megatron-lm/fin_Latn: 17,700m tokens of 63,971m (27.7%).
hplt-3.0/megatron-lm/fra_Latn: 69,950m tokens of 625,713m (11.2%).
hplt-3.0/megatron-lm/gle_Latn: 12,080m tokens of 827m (1,460.7%).
hplt-3.0/megatron-lm/glg_Latn: 12,320m tokens of 2,993m (411.6%).
hplt-3.0/megatron-lm/hrv_Latn: 16,260m tokens of 30,263m (53.7%).
hplt-3.0/megatron-lm/hun_Latn: 23,950m tokens of 90,780m (26.4%).
hplt-3.0/megatron-lm/isl_Latn: 11,160m tokens of 4,816m (231.7%).
hplt-3.0/megatron-lm/ita_Latn: 50,540m tokens of 297,646m (17.0%).
hplt-3.0/megatron-lm/kat_Geor: 115,420m tokens of 44,120m (261.6%).
hplt-3.0/megatron-lm/lit_Latn: 14,970m tokens of 22,728m (65.9%).
hplt-3.0/megatron-lm/lvs_Latn: 20,260m tokens of 18,704m (108.3%).
hplt-3.0/megatron-lm/mkd_Cyrl: 14,020m tokens of 4,808m (291.6%).
hplt-3.0/megatron-lm/mlt_Latn: 14,910m tokens of 752m (1,982.7%).
hplt-3.0/megatron-lm/nld_Latn: 29,710m tokens of 168,278m (17.7%).
hplt-3.0/megatron-lm/nno_Latn: 9,970m tokens of 1,420m (702.1%).
hplt-3.0/megatron-lm/nob_Latn: 13,700m tokens of 44,588m (30.7%).
hplt-3.0/megatron-lm/pol_Latn: 32,590m tokens of 221,631m (14.7%).
hplt-3.0/megatron-lm/por_Latn: 56,650m tokens of 327,821m (17.3%).
hplt-3.0/megatron-lm/ron_Latn: 24,410m tokens of 98,113m (24.9%).
hplt-3.0/megatron-lm/slk_Latn: 16,370m tokens of 33,059m (49.5%).
hplt-3.0/megatron-lm/slv_Latn: 14,860m tokens of 17,563m (84.6%).
hplt-3.0/megatron-lm/spa_Latn: 101,920m tokens of 692,156m (14.7%).
hplt-3.0/megatron-lm/srp_Cyrl: 16,670m tokens of 9,022m (184.8%).
hplt-3.0/megatron-lm/swe_Latn: 20,570m tokens of 101,847m (20.2%).
hplt-3.0/megatron-lm/tur_Latn: 27,630m tokens of 149,012m (18.5%).
hplt-3.0/megatron-lm/ukr_Cyrl: 22,340m tokens of 75,602m (29.5%).
megamath-0.0.0/megatron-lm/megamath-text-code-block: 20,000m tokens of 48,138m (41.5%).
megamath-0.0.0/megatron-lm/megamath-web-pro: 20,000m tokens of 14,183m (141.0%).
nemotron-cc-1.0/megatron-lm/high/actual: 540,000m tokens of 565,924m (95.4%).
nemotron-cc-1.0/megatron-lm/medium/actual: 1,314,400m tokens of 2,063,798m (63.7%).
nemotron-cc-1.0/megatron-lm/medium-high/actual: 540,000m tokens of 515,465m (104.8%).
nemotron-cc-opus-1.1/megatron-lm/bos_Latn: 9,940m tokens of 98,924m (10.0%).
nemotron-cc-opus-1.1/megatron-lm/bul_Cyrl: 13,090m tokens of 123,243m (10.6%).
nemotron-cc-opus-1.1/megatron-lm/cat_Latn: 11,730m tokens of 124,353m (9.4%).
nemotron-cc-opus-1.1/megatron-lm/ces_Latn: 14,710m tokens of 95,432m (15.4%).
nemotron-cc-opus-1.1/megatron-lm/ell_Grek: 21,330m tokens of 154,930m (13.8%).
nemotron-cc-opus-1.1/megatron-lm/est_Latn: 9,800m tokens of 114,680m (8.5%).
nemotron-cc-opus-1.1/megatron-lm/eus_Latn: 8,000m tokens of 102,052m (7.8%).
nemotron-cc-opus-1.1/megatron-lm/gle_Latn: 8,460m tokens of 126,261m (6.7%).
nemotron-cc-opus-1.1/megatron-lm/glg_Latn: 8,380m tokens of 109,875m (7.6%).
nemotron-cc-opus-1.1/megatron-lm/hrv_Latn: 11,060m tokens of 108,210m (10.2%).
nemotron-cc-opus-1.1/megatron-lm/kat_Geor: 78,490m tokens of 1,038,476m (7.6%).
nemotron-cc-opus-1.1/megatron-lm/lav_Latn: 13,770m tokens of 170,792m (8.1%).
nemotron-cc-opus-1.1/megatron-lm/lit_Latn: 10,180m tokens of 114,912m (8.9%).
nemotron-cc-opus-1.1/megatron-lm/mkd_Cyrl: 9,530m tokens of 125,518m (7.6%).
nemotron-cc-opus-1.1/megatron-lm/mlt_Latn: 10,140m tokens of 134,951m (7.5%).
nemotron-cc-opus-1.1/megatron-lm/slk_Latn: 11,130m tokens of 113,218m (9.8%).
nemotron-cc-opus-1.1/megatron-lm/slv_Latn: 10,100m tokens of 112,583m (9.0%).
nemotron-cc-opus-1.1/megatron-lm/sqi_Latn: 14,000m tokens of 174,219m (8.0%).
nemotron-cc-opus-1.1/megatron-lm/srp_Cyrl: 11,340m tokens of 130,924m (8.7%).
nemotron-cc-opus-1.1/megatron-lm/tur_Latn: 18,790m tokens of 117,196m (16.0%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower72b/deu_Latn: 36,040m tokens of 81,859m (44.0%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower72b/fin_Latn: 12,040m tokens of 93,248m (12.9%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower72b/ita_Latn: 34,370m tokens of 82,285m (41.8%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower72b/spa_Latn: 69,300m tokens of 91,611m (75.6%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower72b/swe_Latn: 13,990m tokens of 89,681m (15.6%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/dan_Latn: 10,330m tokens of 87,700m (11.8%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/fra_Latn: 47,560m tokens of 102,377m (46.5%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/hun_Latn: 16,290m tokens of 99,853m (16.3%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/isl_Latn: 7,590m tokens of 97,348m (7.8%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/nld_Latn: 20,200m tokens of 93,648m (21.6%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/nno_Latn: 6,780m tokens of 90,132m (7.5%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/nob_Latn: 9,320m tokens of 84,161m (11.1%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/pol_Latn: 22,160m tokens of 83,828m (26.4%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/por_Latn: 38,520m tokens of 90,185m (42.7%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/ron_Latn: 16,600m tokens of 100,047m (16.6%).
nemotron-cc-tower+-0.1/megatron-lm/parallel/tower9b/ukr_Cyrl: 15,190m tokens of 102,496m (14.8%).
olmo-mix-1124/megatron-lm/arxiv: 92,000m tokens of 22,325m (412.1%).
olmo-mix-1124/megatron-lm/pes2o: 253,000m tokens of 60,351m (419.2%).
olmo-mix-1124/megatron-lm/wiki: 15,000m tokens of 3,836m (391.0%).
starcoder-0.0.0/megatron-lm: 720,000m tokens of 276,861m (260.1%).
"""


CODE_TO_LOCAL = {
    "als_Latn": "sq",
    "bos_Latn": "bs",
    "bul_Cyrl": "bg",
    "cat_Latn": "ca",
    "ces_Latn": "cs",
    "dan_Latn": "da",
    "deu_Latn": "de",
    "ekk_Latn": "et",
    "ell_Grek": "el",
    "eng_Latn": "en",
    "est_Latn": "et",
    "eus_Latn": "eu",
    "fin_Latn": "fi",
    "fra_Latn": "fr",
    "gle_Latn": "ga",
    "glg_Latn": "gl",
    "hrv_Latn": "hr",
    "hun_Latn": "hu",
    "isl_Latn": "is",
    "ita_Latn": "it",
    "kat_Geor": "ka",
    "lav_Latn": "lv",
    "lit_Latn": "lt",
    "lvs_Latn": "lv",
    "mkd_Cyrl": "mk",
    "mlt_Latn": "mt",
    "nld_Latn": "nl",
    "nno_Latn": "nn",
    "nob_Latn": "no",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "ron_Latn": "ro",
    "rus_Cyrl": "ru",
    "slk_Latn": "sk",
    "slv_Latn": "sl",
    "spa_Latn": "es",
    "sqi_Latn": "sq",
    "srp_Cyrl": "sr",
    "swe_Latn": "sv",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
}

LOCAL_TO_CODE = {v: k for k, v in CODE_TO_LOCAL.items()}
TARGET_LOCAL = ["bg", "uk", "mk", "sr", "pl", "cs", "sk", "sl", "hr", "bs"]


@dataclass
class RunResult:
    condition: str
    seed: int
    steps: int
    train_tokens_m: float
    seconds: float
    peak_gb: float
    target_avg_bpb: float
    weights: dict[str, float]
    eval_bpb: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--train-bytes", type=int, default=12_000_000)
    p.add_argument("--val-bytes", type=int, default=2_000_000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--russian-target-m", type=float, default=None)
    p.add_argument("--skip-extra-budget", action="store_true")
    return p.parse_args()


def parse_mix() -> dict[str, float]:
    out: dict[str, float] = {}
    pattern = re.compile(r"/([a-z]{3}_[A-Za-z]+):\s*([\d,]+)m tokens")
    for line in MIX_TEXT.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        code = m.group(1)
        value = float(m.group(2).replace(",", ""))
        out[code] = out.get(code, 0.0) + value
    return out


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in sorted(weights.items()) if v > 0}


def find_parquets(data_root: Path, lang: str) -> list[Path]:
    lang_dir = data_root / lang
    paths = sorted(p for p in lang_dir.glob("*.parquet") if p.is_file())
    if not paths:
        paths = sorted(p for p in lang_dir.rglob("*.parquet") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No parquet files found for {lang} under {lang_dir}")
    return paths


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def build_language_cache(
    data_root: Path,
    cache_dir: Path,
    lang: str,
    train_bytes: int,
    val_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"{lang}.train.u8"
    val_path = cache_dir / f"{lang}.val.u8"
    meta_path = cache_dir / f"{lang}.meta.json"

    if train_path.exists() and val_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("train_bytes") >= train_bytes and meta.get("val_bytes") >= val_bytes:
            train = np.memmap(train_path, dtype=np.uint8, mode="r", shape=(meta["train_bytes"],))
            val = np.memmap(val_path, dtype=np.uint8, mode="r", shape=(meta["val_bytes"],))
            return np.asarray(train[:train_bytes]), np.asarray(val[:val_bytes])

    needed = train_bytes + val_bytes
    docs: list[bytes] = []
    total = 0
    for parquet_path in find_parquets(data_root, lang):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                b = (normalize_text(text) + "\n\n").encode("utf-8", errors="ignore")
                if len(b) < 200:
                    continue
                docs.append(b)
                total += len(b)
                if total >= needed * 1.2:
                    break
            if total >= needed * 1.2:
                break
        if total >= needed * 1.2:
            break
    if total < needed:
        raise RuntimeError(f"Only collected {total:,} bytes for {lang}; need {needed:,}")
    rng = random.Random(17)
    rng.shuffle(docs)
    joined = b"".join(docs)
    val_blob = joined[:val_bytes]
    train_blob = joined[val_bytes : val_bytes + train_bytes]
    train_path.write_bytes(train_blob)
    val_path.write_bytes(val_blob)
    meta_path.write_text(json.dumps({"lang": lang, "train_bytes": len(train_blob), "val_bytes": len(val_blob)}, indent=2))
    return np.frombuffer(train_blob, dtype=np.uint8).copy(), np.frombuffer(val_blob, dtype=np.uint8).copy()


def make_batch(
    arrays: dict[str, np.ndarray],
    weights: dict[str, float],
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    langs = list(weights)
    probs = np.asarray([weights[l] for l in langs], dtype=np.float64)
    probs /= probs.sum()
    buf = np.empty((batch_size, seq_len + 1), dtype=np.uint8)
    sampled = rng.choice(len(langs), size=batch_size, p=probs)
    for i, lang_idx in enumerate(sampled):
        arr = arrays[langs[int(lang_idx)]]
        start = int(rng.integers(0, len(arr) - seq_len - 1))
        buf[i] = arr[start : start + seq_len + 1]
    t = torch.from_numpy(buf.astype(np.int64, copy=False)).to(device, non_blocking=True)
    return t[:, :-1], t[:, 1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ByteLM(nn.Module):
    def __init__(self, seq_len: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(256, n_embd)
        self.pos = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 256, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.tok.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        _, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln(h))
        loss = None if y is None else F.cross_entropy(logits.view(-1, 256), y.reshape(-1))
        return logits, loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_arrays: dict[str, np.ndarray],
    eval_langs: list[str],
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out = {}
    for lang in eval_langs:
        rng = np.random.default_rng(10_000 + sum(ord(c) for c in lang))
        losses = []
        for _ in range(eval_batches):
            x, y = make_batch(val_arrays, {lang: 1.0}, batch_size, seq_len, rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[lang] = float(np.mean(losses) / math.log(2))
    model.train()
    return out


def train_one(
    condition: str,
    weights: dict[str, float],
    steps: int,
    seed: int,
    args: argparse.Namespace,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    eval_langs: list[str],
    target_langs: list[str],
    device: torch.device,
) -> RunResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = ByteLM(args.seq_len, args.n_layer, args.n_head, args.n_embd, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    start_time = time.time()
    for step in range(steps):
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * step / max(1, steps)))
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_mult
        x, y = make_batch(train_arrays, weights, args.batch_size, args.seq_len, rng, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"{condition} seed={seed} step={step+1}/{steps} loss={loss.item():.4f}", flush=True)
    seconds = time.time() - start_time
    eval_bpb = evaluate(model, val_arrays, eval_langs, args.batch_size, args.seq_len, args.eval_batches, device)
    target_avg = float(np.mean([eval_bpb[l] for l in target_langs]))
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    tokens_m = steps * args.batch_size * args.seq_len / 1e6
    return RunResult(condition, seed, steps, tokens_m, seconds, peak_gb, target_avg, weights, eval_bpb)


def summarize(results: list[RunResult], work_dir: Path, eval_langs: list[str], target_langs: list[str]) -> None:
    rows = [asdict(r) for r in results]
    (work_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    by_cond: dict[str, list[RunResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)
    lines = [
        "# Mix-Matched Russian Inclusion Results",
        "",
        f"Lower BPB is better. Target average is {','.join(target_langs)}.",
        "",
        "| condition | seeds | target_avg_bpb | " + " | ".join(eval_langs) + " | train_tokens_M | peak_gb |",
        "|---|---:|---:|" + "|".join(["---:"] * len(eval_langs)) + "|---:|---:|",
    ]
    summary = {}
    for cond, runs in by_cond.items():
        mean_eval = {lang: float(np.mean([r.eval_bpb[lang] for r in runs])) for lang in eval_langs}
        mean_target = float(np.mean([r.target_avg_bpb for r in runs]))
        mean_tokens = float(np.mean([r.train_tokens_m for r in runs]))
        mean_peak = float(np.mean([r.peak_gb for r in runs]))
        summary[cond] = {
            "target_avg_bpb": mean_target,
            "eval_bpb": mean_eval,
            "train_tokens_m": mean_tokens,
            "peak_gb": mean_peak,
            "seeds": [r.seed for r in runs],
            "weights": runs[-1].weights,
        }
        lines.append(
            f"| {cond} | {len(runs)} | {mean_target:.4f} | "
            + " | ".join(f"{mean_eval[l]:.4f}" for l in eval_langs)
            + f" | {mean_tokens:.1f} | {mean_peak:.2f} |"
        )
    if "base_no_ru" in summary:
        base = summary["base_no_ru"]["target_avg_bpb"]
        lines.extend(["", "## Deltas vs base_no_ru", ""])
        for cond, stats in summary.items():
            if cond == "base_no_ru":
                continue
            delta = stats["target_avg_bpb"] - base
            rel = 100.0 * delta / base
            direction = "better" if delta < 0 else "worse"
            lines.append(f"- `{cond}`: {delta:+.4f} BPB ({rel:+.2f}%), {direction} on target average.")
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (work_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    mix_by_code = parse_mix()
    russian_target = args.russian_target_m if args.russian_target_m is not None else mix_by_code["ukr_Cyrl"]
    available_local = sorted(k.name for k in args.data_root.iterdir() if k.is_dir())

    base_m = {}
    missing = {}
    for code, token_m in sorted(mix_by_code.items()):
        local = CODE_TO_LOCAL.get(code)
        if local and local in available_local:
            base_m[local] = base_m.get(local, 0.0) + token_m
        else:
            missing[code] = token_m

    base_weights = normalize(base_m)
    with_ru_raw = dict(base_m)
    with_ru_raw["ru"] = with_ru_raw.get("ru", 0.0) + russian_target
    with_ru_same_total = normalize(with_ru_raw)
    total_base = sum(base_m.values())
    total_with_ru = sum(with_ru_raw.values())
    extra_steps = math.ceil(args.steps * total_with_ru / total_base)

    eval_langs = sorted(set(base_weights) | {"ru"})
    target_langs = [l for l in TARGET_LOCAL if l in eval_langs]

    metadata = {
        "base_target_m_by_local_lang": base_m,
        "base_weights": base_weights,
        "with_ru_same_total_weights": with_ru_same_total,
        "russian_target_m": russian_target,
        "russian_target_default": "ukr_Cyrl collapsed target mass",
        "total_base_m": total_base,
        "total_with_ru_m": total_with_ru,
        "extra_steps": extra_steps,
        "missing_or_unavailable_target_m_by_code": missing,
    }
    (args.work_dir / "mixture_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

    langs_to_cache = sorted(set(eval_langs) | set(base_weights) | {"ru"})
    cache_dir = args.work_dir / "byte_cache"
    train_arrays: dict[str, np.ndarray] = {}
    val_arrays: dict[str, np.ndarray] = {}
    for lang in langs_to_cache:
        train, val = build_language_cache(args.data_root, cache_dir, lang, args.train_bytes, args.val_bytes)
        train_arrays[lang] = train
        val_arrays[lang] = val
        print(f"{lang}: train={len(train):,} bytes val={len(val):,} bytes weight={base_weights.get(lang, 0):.6f}", flush=True)

    conditions = [
        ("base_no_ru", base_weights, args.steps),
        ("with_ru_same_total", with_ru_same_total, args.steps),
    ]
    if not args.skip_extra_budget:
        conditions.append(("with_ru_extra_budget", with_ru_same_total, extra_steps))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results: list[RunResult] = []
    for seed in seeds:
        for condition, weights, steps in conditions:
            print(f"\n=== {condition} seed={seed} steps={steps} ===", flush=True)
            print("weights:", json.dumps(weights, sort_keys=True), flush=True)
            result = train_one(condition, weights, steps, seed, args, train_arrays, val_arrays, eval_langs, target_langs, device)
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False), flush=True)
            results.append(result)
            summarize(results, args.work_dir, eval_langs, target_langs)


if __name__ == "__main__":
    main()
