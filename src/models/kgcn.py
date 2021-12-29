# -*- coding: utf-8 -*-
"""Implements the KGCN model."""


import math
from typing import Any, Dict, Sequence

import numpy as np
import torch
from jsonargparse import Namespace
from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from .base import BaseModel
from .utils import get_activation


class KGCN(BaseModel):
    def __init__(
        self,
        optim_args: Namespace,
        num_users: PositiveInt,
        num_entities: PositiveInt,
        num_relations: PositiveInt,
        embed_dim: PositiveInt,
        bias: bool = True,
        activation: str = "tanh",
        dropout: ClosedUnitInterval = 0.0,
        dropout_attn: ClosedUnitInterval = 0.0,
    ):
        """Constructs a KGCN model.

        Args:
            optim_args: Arguments for optimization.
            num_users: Number of users.
            num_entities: Number of entities.
            num_relations: Number of relations.
            embed_dim: Dimension of the embedding space.
            bias: Whether to learn an additive bias.
            activation: Activation function used in the model.
            dropout: Dropout rate of the model.
            dropout_attn: Dropout rate of the attention model.
        """
        super().__init__(optim_args)
        self._embed_dim = embed_dim

        self.embeddings_user = nn.Parameter(torch.empty(num_users, embed_dim))
        self.embeddings_entity = nn.Parameter(
            torch.empty(num_entities, embed_dim)
        )
        self.fc_relation = nn.Linear(num_relations, embed_dim)
        self.fc = nn.Linear(self._embed_dim, self._embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.activation = get_activation(activation)
        self.register_buffer(
            "inf", torch.Tensor([torch.finfo(torch.float32).max])
        )
        self.reset_parameters()

        self.probs = None
        self.labels = None

    def reset_parameters(self):
        fan_out = self.embeddings_user.size(1)
        nn.init.normal_(self.embeddings_user, std=1 / math.sqrt(fan_out))

        fan_out = self.embeddings_entity.size(1)
        nn.init.normal_(self.embeddings_entity, std=1 / math.sqrt(fan_out))

    def forward(
        self,
        uids: torch.Tensor,
        iids: torch.Tensor,
        neighbors: torch.Tensor,
        feats_edge: torch.Tensor,
        num_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        embeddings_user = self.embeddings_user[uids]
        embeddings_nbrs = self.embeddings_entity[neighbors]
        key = self.fc_relation(feats_edge)
        logits = torch.sum(
            embeddings_user.unsqueeze(1).contiguous() * key, dim=2
        )
        # logits = mask * logits
        probs = self.dropout_attn(
            F.softmax(logits, dim=1).unsqueeze(2).contiguous()
        )
        values = torch.sum(probs * embeddings_nbrs, dim=1)
        embeddings_item = self.embeddings_entity[iids] + values
        embeddings_item = self.activation(
            self.fc(self.dropout(embeddings_item))
        )
        logits = torch.sum(embeddings_user * embeddings_item, dim=1)
        return {"logits": logits}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.forward(
            batch["uids"],
            batch["iids"],
            batch["neighbors"],
            batch["feats_edge"],
            batch["num_neighbors"],
        )
        cls_loss = F.binary_cross_entropy_with_logits(
            outputs["logits"], target=batch["labels"].float()
        )
        return cls_loss

    def training_epoch_end(self, outputs: Sequence[torch.Tensor]):
        losses = []
        for out in outputs:
            losses.append(out["loss"].item())
        print(np.mean(losses))

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(
            batch["uids"],
            batch["iids"],
            batch["neighbors"],
            batch["feats_edge"],
            batch["num_neighbors"],
        )
        probs = torch.sigmoid(outputs["logits"])
        return {"probs": probs, "labels": batch["labels"]}

    def validation_step_end(self, outputs: Dict[str, torch.Tensor]):
        self.probs.extend(outputs["probs"].detach().cpu().numpy())
        self.labels.extend(outputs["labels"].detach().cpu().numpy())

    def validation_epoch_end(self, outputs: Sequence[Any]):
        probs = np.hstack(self.probs)
        preds = (probs >= 0.5).astype(np.int32)
        labels = np.hstack(self.labels)
        print(f"shape = {probs.shape}")
        print(f"auc = {roc_auc_score(y_true=labels, y_score=probs)}")
        print(f"f1 = {f1_score(y_true=labels, y_pred = preds)}")

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs: Dict[str, torch.Tensor]):
        return self.validation_step_end(outputs)

    def test_epoch_end(self, outputs: Sequence[Any]):
        self.validation_epoch_end(outputs)

    # methods of ModelHook
    def on_validation_epoch_start(self):
        self.probs = []
        self.labels = []

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
