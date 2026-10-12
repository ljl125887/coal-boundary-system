from __future__ import annotations


EXTRACTION_STAGES = {
    "INITIALIZING": ("初始化", 0.0, 4.0),
    "READ_VALIDATE": ("读取并校验数据", 5.0, 14.0),
    "SURFACE_GRID": ("建立或加载表面栅格", 15.0, 31.0),
    "GROUND_ESTIMATION": ("估计坡地地面曲面", 32.0, 44.0),
    "MODEL_INFERENCE": ("执行模型推理", 45.0, 57.0),
    "CANDIDATE_EXTRACTION": ("提取煤堆候选区", 58.0, 69.0),
    "VALLEY_SPLIT": ("按低洼沟槽分堆", 70.0, 75.0),
    "SLOPE_RECOVERY": ("补全连续坡面并剔除结构误入", 76.0, 79.0),
    "GEOMETRY_VOLUME": ("生成边界与体积", 80.0, 87.0),
    "EVALUATION": ("评估人工参考边界", 88.0, 95.0),
    "EXPORT": ("输出 JSON/GeoJSON", 96.0, 99.0),
    "COMPLETED": ("完成", 100.0, 100.0),
}

TRAINING_STAGES = {
    "DATA_VALIDATE": ("校验训练数据", 5.0, 12.0),
    "DATA_PREPARE": ("准备训练样本", 13.0, 24.0),
    "TRAINING": ("训练模型", 25.0, 84.0),
    "VALIDATION": ("验证模型", 85.0, 94.0),
    "MODEL_EXPORT": ("导出模型与报告", 95.0, 99.0),
}

RECALCULATION_STAGES = {
    "RECALC_VALIDATE": ("校验源结果与目标边界", 5.0, 15.0),
    "RECALC_LOAD_LOCAL": ("读取目标边界周边点云", 16.0, 45.0),
    "RECALC_GROUND": ("估计局部坡地地面曲面", 46.0, 68.0),
    "RECALC_VOLUME": ("计算单煤堆挖方体积", 69.0, 90.0),
    "RECALC_EXPORT": ("输出单煤堆重算结果", 91.0, 99.0),
}

ALL_STAGES = {**EXTRACTION_STAGES, **TRAINING_STAGES, **RECALCULATION_STAGES}


def stage_name(stage_code: str) -> str:
    return ALL_STAGES.get(stage_code, (stage_code, 0.0, 100.0))[0]


def stage_start(stage_code: str) -> float:
    return ALL_STAGES.get(stage_code, (stage_code, 0.0, 100.0))[1]


def stage_fraction_progress(stage_code: str, current: int, total: int) -> float:
    _, start, end = ALL_STAGES.get(stage_code, (stage_code, 0.0, 100.0))
    fraction = min(1.0, max(0.0, current / max(total, 1)))
    return round(start + (end - start) * fraction, 2)
