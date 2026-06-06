"""STU-Net warm-start fine-tune 런처 (PENGWIN 2026 Task 1).

목적
----
nnU-Net 2.5.2 의 표준 `nnUNetv2_train` 은 pretrained 적재 시
`nnunetv2.run.run_training.load_pretrained_weights` 를 호출하는데, 이 기본 로더는
(1) STUNet 의 seg head 네이밍(`seg_outputs.*`)을 skip 하지 못하고(`.seg_layers.` 만 skip),
(2) 입력 채널 inflation 을 처리하지 못해 STUNet warm-start 에서 깨진다.

본 런처는 STU-Net 공식 run_finetuning_stunet.py 와 동일한 monkey-patch 방식으로,
표준 nnU-Net 학습 진입점(run_training_entry)을 그대로 쓰되 pretrained 로더만 우리
`stunet.load_stunet_pretrained_weights`(prefix 처리 + stem inflate + 안전 역직렬화)로
교체한다. 설치된 nnU-Net 파일은 건드리지 않는다.

사용 예
-------
    PYTHONPATH=/workspace/code_task1 \
    nnUNet_raw=.../result/raw nnUNet_preprocessed=.../result/preprocessed \
    nnUNet_results=.../result/results \
    python -m run_finetuning_stunet 539 3d_fullres 0 \
        -tr PengwinTrainerSTUNetBaseAnatomyV301 \
        -p nnUNetResEncUNetLPlans \
        -pretrained_weights result/weights/stunet_base_TotalSeg.pth \
        --npz [-num_gpus 2]

인자는 전부 nnUNetv2_train(run_training_entry)과 동일하게 전달된다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# code_task1 을 path 에 올려 stunet/core 를 import 가능하게.
_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import nnunetv2.run.run_training as _rt  # noqa: E402
from stunet import load_stunet_pretrained_weights  # noqa: E402


def main(argv=None):
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    # nnU-Net 의 run_training 네임스페이스에 import 된 load_pretrained_weights 를
    # 우리 STUNet 로더로 교체. 시그니처(network, fname, verbose) 호환.
    with patch.object(_rt, "load_pretrained_weights", load_stunet_pretrained_weights):
        _rt.run_training_entry()


if __name__ == "__main__":
    main()
