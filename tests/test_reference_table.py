from pathlib import Path

import pandas as pd


def test_transcribed_headline_table() -> None:
    table = pd.read_csv(Path("data/reference/paper_balanced_accuracy_5fold_headline.csv"))
    assert len(table) == 25
    expected_averages = {
        "Acc": 0.68,
        "Gyro": 0.66,
        "WAcc": 0.70,
        "Loc": 0.67,
        "Aud": 0.70,
        "PS": 0.75,
        "EF": 0.77,
        "LFA": 0.80,
        "LFL": 0.80,
    }
    for model, published_rounded_average in expected_averages.items():
        assert round(table[model].mean(), 2) == published_rounded_average
