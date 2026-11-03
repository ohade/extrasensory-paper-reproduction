from extrasensory_reproduction.constants import (
    EXPECTED_SENSOR_FEATURE_COUNTS,
    SENSOR_ORDER,
)
from extrasensory_reproduction.data import sensor_for_feature


def test_sensor_prefix_routing() -> None:
    examples = {
        "raw_acc:magnitude_stats:mean": "Acc",
        "proc_gyro:magnitude_stats:mean": "Gyro",
        "watch_acceleration:magnitude_stats:mean": "WAcc",
        "location:num_valid_updates": "Loc",
        "location_quick_features:std_lat": "Loc",
        "audio_naive:mfcc0:mean": "Aud",
        "discrete:time_of_day:between0and6": "PS",
        "raw_magnet:magnitude_stats:mean": None,
    }
    assert {feature: sensor_for_feature(feature) for feature in examples} == examples
    assert tuple(EXPECTED_SENSOR_FEATURE_COUNTS) == SENSOR_ORDER
