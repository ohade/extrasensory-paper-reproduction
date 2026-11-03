import numpy as np
import torch

from extrasensory_reproduction.research.neural import (
    CausalTCN,
    LowRankLabelRefinement,
    SensorGatedMoETCN,
)


def test_neural_models_are_causal_shape_safe_and_finite() -> None:
    features = torch.randn(3, 30, 12)
    availability = torch.ones(3, 30, 6, dtype=torch.bool)
    valid = torch.ones(3, 30, dtype=torch.bool)
    valid[0, :20] = False
    features[0, :20] = 0

    tcn = CausalTCN(features=12, hidden_dim=16, labels=5, rank=3, dropout=0.0)
    tcn.eval()
    output = tcn(features, availability, valid)
    assert output.shape == (3, 5)
    assert torch.isfinite(output).all()

    sensor_columns = [np.asarray([2 * sensor, 2 * sensor + 1]) for sensor in range(6)]
    moe = SensorGatedMoETCN(
        sensor_columns,
        hidden_dim=16,
        labels=5,
        rank=3,
        dropout=0.0,
        sensor_dropout=0.2,
        output_features=12,
    )
    availability[1, :, 2:] = False
    moe.eval()
    output = moe(features, availability, valid)
    assert output.shape == (3, 5)
    assert torch.isfinite(output).all()


def test_label_refinement_has_no_self_coupling() -> None:
    layer = LowRankLabelRefinement(hidden_dim=4, labels=3, rank=2)
    coupling = layer.left @ layer.right.transpose(0, 1)
    zero_diagonal = coupling - torch.diag_embed(torch.diagonal(coupling))
    assert torch.equal(torch.diagonal(zero_diagonal), torch.zeros(3))
