"""Small MPS-compatible multi-label, temporal, and sensor-gated models."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..constants import SENSOR_ORDER
from ..data import ExtraSensoryDataset
from .baselines import BaselinePrediction, Standardizer
from .config import ExperimentConfig
from .metrics import evaluate_probabilities
from .sequence import (
    CausalSequenceDataset,
    SequenceArrays,
    build_history_indices,
)


class LowRankLabelRefinement(nn.Module):
    """One prediction-only refinement step with a zero diagonal."""

    def __init__(self, hidden_dim: int, labels: int, rank: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden_dim, labels)
        self.rank = rank
        if rank:
            self.left = nn.Parameter(torch.empty(labels, rank))
            self.right = nn.Parameter(torch.empty(labels, rank))
            self.scale = nn.Parameter(torch.tensor(0.1))
            nn.init.normal_(self.left, std=0.02)
            nn.init.normal_(self.right, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden)
        if not self.rank:
            return logits
        coupling = self.left @ self.right.transpose(0, 1)
        coupling = coupling - torch.diag_embed(torch.diagonal(coupling))
        correction = torch.sigmoid(logits) @ coupling.transpose(0, 1)
        return logits + self.scale * correction / math.sqrt(self.rank)


class MultitaskMLP(nn.Module):
    def __init__(self, features: int, hidden_dim: int, labels: int, rank: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = LowRankLabelRefinement(hidden_dim, labels, rank)

    def encode(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        del sensor_available, timestep_valid
        return self.encoder(features[:, -1])

    def forward(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(features, sensor_available, timestep_valid))


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _causal(self, convolution: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return convolution(F.pad(values, (2 * self.dilation, 0)))

    def forward(self, values: torch.Tensor, timestep_valid: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self._causal(self.first, values)
        output = self.dropout(F.gelu(output))
        output = self._causal(self.second, output)
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        output = F.gelu(output + residual)
        return output * timestep_valid.unsqueeze(1)


class CausalTCN(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        labels: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(features, hidden_dim)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.head = LowRankLabelRefinement(hidden_dim, labels, rank)

    def encode(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        del sensor_available
        output = self.input_projection(features)
        output = output * timestep_valid.unsqueeze(-1)
        output = output.transpose(1, 2)
        for block in self.blocks:
            output = block(output, timestep_valid)
        return output[:, :, -1]

    def forward(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(features, sensor_available, timestep_valid))


class SensorGatedMoETCN(nn.Module):
    def __init__(
        self,
        sensor_columns: list[np.ndarray],
        hidden_dim: int,
        labels: int,
        rank: int,
        dropout: float,
        sensor_dropout: float,
        output_features: int,
    ) -> None:
        super().__init__()
        expert_dim = 64
        self.sensor_columns = [torch.as_tensor(columns, dtype=torch.long) for columns in sensor_columns]
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(len(columns), expert_dim), nn.GELU())
            for columns in sensor_columns
        )
        self.gate = nn.Linear(len(sensor_columns), len(sensor_columns))
        self.input_projection = nn.Linear(expert_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.head = LowRankLabelRefinement(hidden_dim, labels, rank)
        self.reconstruction = nn.Linear(hidden_dim, output_features)
        self.sensor_dropout = sensor_dropout

    def _drop_sensors(self, availability: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sensor_dropout <= 0.0:
            return availability
        keep = torch.rand_like(availability, dtype=torch.float32) >= self.sensor_dropout
        dropped = availability & keep
        empty = ~dropped.any(dim=-1, keepdim=True)
        return torch.where(empty, availability, dropped)

    def encode(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        availability = self._drop_sensors(sensor_available)
        expert_outputs = []
        for columns, expert in zip(self.sensor_columns, self.experts, strict=True):
            selected = features.index_select(-1, columns.to(features.device))
            expert_outputs.append(expert(selected))
        experts = torch.stack(expert_outputs, dim=-2)
        experts = experts * availability.unsqueeze(-1)

        logits = self.gate(availability.float())
        any_available = availability.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~availability, -1e9)
        logits = torch.where(any_available, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * any_available
        fused = torch.sum(experts * weights.unsqueeze(-1), dim=-2)

        output = self.input_projection(fused) * timestep_valid.unsqueeze(-1)
        output = output.transpose(1, 2)
        for block in self.blocks:
            output = block(output, timestep_valid)
        return output[:, :, -1]

    def forward(
        self,
        features: torch.Tensor,
        sensor_available: torch.Tensor,
        timestep_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(features, sensor_available, timestep_valid))


def fit_predict_neural(
    config: ExperimentConfig,
    dataset: ExtraSensoryDataset,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    label_columns: np.ndarray,
) -> BaselinePrediction:
    _seed_everything(config.seed, config.max_workers)
    device = _device()
    fit_rows, validation_rows = _user_validation_split(
        train_rows, dataset.user_index, config.seed
    )
    standardizer = Standardizer.fit(dataset.features[fit_rows])
    scaled_features = standardizer.transform(dataset.features)
    sensor_available = np.column_stack(
        [dataset.sensor_available(sensor) for sensor in SENSOR_ORDER]
    )
    sensor_columns = [
        np.flatnonzero(dataset.feature_sensors == sensor) for sensor in SENSOR_ORDER
    ]

    train_loader = _loader(
        dataset,
        scaled_features,
        sensor_available,
        fit_rows,
        label_columns,
        config,
        shuffle=True,
    )
    validation_loader = _loader(
        dataset,
        scaled_features,
        sensor_available,
        validation_rows,
        label_columns,
        config,
        shuffle=False,
    )
    test_loader = _loader(
        dataset,
        scaled_features,
        sensor_available,
        test_rows,
        label_columns,
        config,
        shuffle=False,
    )
    model = _build_model(config, dataset.features.shape[1], len(label_columns), sensor_columns)
    model.to(device)
    if config.pretrain_epochs:
        if not isinstance(model, SensorGatedMoETCN):
            raise ValueError("Masked reconstruction pretraining requires sensor_moe_tcn")
        _pretrain_reconstruction(model, train_loader, config, device)
    _fit_supervised(model, train_loader, validation_loader, config, device)

    validation_probabilities, validation_truth, validation_valid, _ = _predict(
        model, validation_loader, device
    )
    thresholds = _optimize_thresholds(
        validation_truth, validation_probabilities, validation_valid
    )
    test_probabilities, _, _, _ = _predict(model, test_loader, device)
    ablations = {
        name: _predict(
            model,
            test_loader,
            device,
            ablated_feature_columns=feature_columns,
            ablated_sensors=sensors,
        )[0]
        for name, (feature_columns, sensors) in _ablation_specs(sensor_columns).items()
    }
    return BaselinePrediction(
        probabilities=test_probabilities,
        thresholds=thresholds,
        ablations=ablations,
    )


def _build_model(
    config: ExperimentConfig,
    features: int,
    labels: int,
    sensor_columns: list[np.ndarray],
) -> nn.Module:
    if config.model == "multitask_mlp":
        return MultitaskMLP(features, config.hidden_dim, labels, config.label_rank, config.dropout)
    if config.model == "causal_tcn":
        return CausalTCN(features, config.hidden_dim, labels, config.label_rank, config.dropout)
    if config.model == "sensor_moe_tcn":
        return SensorGatedMoETCN(
            sensor_columns,
            config.hidden_dim,
            labels,
            config.label_rank,
            config.dropout,
            config.sensor_dropout,
            features,
        )
    raise ValueError(f"Unsupported neural model: {config.model}")


def _loader(
    dataset: ExtraSensoryDataset,
    scaled_features: np.ndarray,
    sensor_available: np.ndarray,
    rows: np.ndarray,
    label_columns: np.ndarray,
    config: ExperimentConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    histories, timestep_valid = build_history_indices(
        dataset.user_index,
        dataset.timestamps,
        rows,
        history_steps=config.history_steps,
        max_gap_seconds=config.max_gap_seconds,
    )
    label_valid = (
        ~dataset.label_missing[rows][:, label_columns]
        if config.label_policy == "exclude-missing"
        else np.ones((len(rows), len(label_columns)), dtype=bool)
    )
    arrays = SequenceArrays(
        features=scaled_features,
        sensor_available=sensor_available,
        labels=dataset.labels[rows][:, label_columns],
        label_valid=label_valid,
        users=dataset.user_index[rows],
        histories=histories,
        timestep_valid=timestep_valid,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        CausalSequenceDataset(arrays),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _fit_supervised(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    validation_loader: DataLoader[tuple[torch.Tensor, ...]],
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    labels = train_loader.dataset.arrays.labels
    valid = train_loader.dataset.arrays.label_valid
    positives = np.sum(labels * valid, axis=0)
    negatives = np.sum((1 - labels) * valid, axis=0)
    positive_weight = torch.as_tensor(
        np.clip(negatives / np.maximum(positives, 1), 1.0, 50.0),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    group_weights = torch.ones(60, device=device) / 60.0
    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        for batch in train_loader:
            features, availability, timestep_valid, truth, label_valid, users = _to_device(
                batch, device
            )
            logits = model(features, availability, timestep_valid)
            loss_by_label = F.binary_cross_entropy_with_logits(
                logits,
                truth,
                pos_weight=positive_weight,
                reduction="none",
            )
            if config.asymmetric_gamma_negative > 0.0:
                negative_weight = torch.sigmoid(logits).pow(
                    config.asymmetric_gamma_negative
                )
                loss_by_label = torch.where(truth > 0.5, loss_by_label, loss_by_label * negative_weight)
            valid_float = label_valid.float()
            loss_by_example = torch.sum(loss_by_label * valid_float, dim=1) / torch.clamp(
                valid_float.sum(dim=1), min=1.0
            )
            loss = _group_robust_loss(
                loss_by_example, users, group_weights, config.group_dro_eta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        probabilities, truth, label_valid, users = _predict(
            model, validation_loader, device
        )
        thresholds = _optimize_thresholds(truth, probabilities, label_valid)
        score = evaluate_probabilities(
            truth,
            probabilities,
            users,
            thresholds,
            valid_mask=label_valid,
        ).macro_balanced_accuracy
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(best_state)


def _pretrain_reconstruction(
    model: SensorGatedMoETCN,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    original_dropout = model.sensor_dropout
    model.sensor_dropout = max(original_dropout, 0.2)
    for _ in range(config.pretrain_epochs):
        model.train()
        for batch in train_loader:
            features, availability, timestep_valid, _, _, _ = _to_device(batch, device)
            hidden = model.encode(features, availability, timestep_valid)
            reconstruction = model.reconstruction(hidden)
            loss = F.mse_loss(reconstruction, features[:, -1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.sensor_dropout = original_dropout


def _predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
    *,
    ablated_feature_columns: np.ndarray | None = None,
    ablated_sensors: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    users: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features, availability, timestep_valid, batch_truth, batch_valid, batch_users = (
                _to_device(batch, device)
            )
            if ablated_feature_columns is not None:
                features = features.clone()
                columns = torch.as_tensor(
                    ablated_feature_columns, dtype=torch.long, device=device
                )
                features[:, :, columns] = 0.0
            if ablated_sensors:
                availability = availability.clone()
                availability[:, :, list(ablated_sensors)] = False
            probabilities.append(
                torch.sigmoid(model(features, availability, timestep_valid)).cpu().numpy()
            )
            truth.append(batch_truth.cpu().numpy().astype(np.uint8))
            valid.append(batch_valid.cpu().numpy())
            users.append(batch_users.cpu().numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(truth),
        np.concatenate(valid),
        np.concatenate(users),
    )


def _ablation_specs(
    sensor_columns: list[np.ndarray],
) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    specs = {
        sensor: (columns, (index,))
        for index, (sensor, columns) in enumerate(
            zip(SENSOR_ORDER, sensor_columns, strict=True)
        )
    }
    for name, indices in {
        "Acc+Gyro": (0, 1),
        "WAcc+Loc": (2, 3),
        "Aud+PS": (4, 5),
    }.items():
        specs[name] = (
            np.concatenate([sensor_columns[index] for index in indices]),
            indices,
        )
    return specs


def _optimize_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    thresholds = np.full(truth.shape[1], 0.5)
    grid = np.linspace(0.05, 0.95, 37)
    for label in range(truth.shape[1]):
        known = valid[:, label]
        if np.unique(truth[known, label]).size != 2:
            continue
        scores = [
            balanced_accuracy_score(
                truth[known, label], probabilities[known, label] >= threshold
            )
            for threshold in grid
        ]
        thresholds[label] = grid[int(np.argmax(scores))]
    return thresholds


def _group_robust_loss(
    loss_by_example: torch.Tensor,
    users: torch.Tensor,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    if eta <= 0.0:
        return loss_by_example.mean()
    batch_groups = torch.unique(users)
    group_losses = torch.stack(
        [loss_by_example[users == group].mean() for group in batch_groups]
    )
    with torch.no_grad():
        group_weights[batch_groups] *= torch.exp(eta * group_losses.detach())
        group_weights /= group_weights.sum()
    weights = group_weights[batch_groups]
    weights = weights / weights.sum()
    return torch.sum(weights * group_losses)


def _user_validation_split(
    rows: np.ndarray,
    user_index: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    users = np.unique(user_index[rows])
    if len(users) < 5:
        raise ValueError("Neural training requires at least five training users")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users)
    validation_count = max(2, round(0.2 * len(users)))
    validation_users = shuffled[:validation_count]
    validation = rows[np.isin(user_index[rows], validation_users)]
    training = rows[~np.isin(user_index[rows], validation_users)]
    return training, validation


def _to_device(
    batch: Iterable[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int, workers: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(workers)
