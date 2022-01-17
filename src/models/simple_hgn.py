# -*- coding: utf-8 -*-
"""Implements the SimpleHCN model."""


import math
from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from jsonargparse import Namespace
from jsonargparse.typing import (
    ClosedUnitInterval,
    NonNegativeInt,
    OpenUnitInterval,
    PositiveInt,
)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import CTRPredictionDataset, create_dataloader
from ..data.utils import read_data_lastfm
from .base import BaseModel
from .layers import SimpleHGAT
from .utils import ensure_list

import dgl  # isort: skip


class SimpleHGN(BaseModel):
    def __init__(
        self,
        optim_args: Namespace,
        dataset: str,
        data_dir: str,
        embed_dim: PositiveInt,
        num_layers: PositiveInt,
        hidden_dims_node: Union[PositiveInt, Sequence[PositiveInt]],
        hidden_dims_edge: Optional[
            Union[PositiveInt, Sequence[PositiveInt]]
        ] = None,
        num_heads: Union[PositiveInt, Sequence[PositiveInt]] = 1,
        bias: bool = True,
        activation: str = "relu",
        dropout: ClosedUnitInterval = 0.0,
        dropout_attn: ClosedUnitInterval = 0.0,
        residual: bool = True,
        val_ratio: OpenUnitInterval = 0.2,
        test_ratio: OpenUnitInterval = 0.2,
        reverse_triplet: bool = True,
        reverse_relation: bool = True,
        use_edges_user: bool = False,
        batch_size_train: PositiveInt = 32,
        batch_size_val: Optional[PositiveInt] = None,
        batch_size_test: Optional[PositiveInt] = None,
        num_workers: NonNegativeInt = 0,
        pin_memory: bool = True,
    ):
        """Constructs a SimpleHGN model.

        Args:
            optim_args: Arguments for optimization.
            dataset: Name of the dataset.
            data_dir: Directory containing the dataset files.
            embed_dim: Dimension of the embedding space.
            num_layers: Number of hidden layers.
            hidden_dims_node: Dimensions of hidden node representations per
                attention head. If set to a list of integers, the length should
                be `num_layers` and `hidden_dims_node[i]` is the dimension
                of node representations at the `i-th` layer. If set to an
                integer, all hidden node representations will have this many
                dimensions.
            hidden_dims_edge: Dimensions of hidden edge representations per
                attention head. If set to a list of integers, the length should
                be `num_layers` and `hidden_dims_edge[i]` is the dimension
                of edge representations at the `i-th` layer. If set to an
                integer, all hidden edge representations will have this many
                dimensions. If set to `None`, will use `hidden_dims_node`.
            num_heads: Numbers of attention heads at hidden layers. If set to a
                list of integers, the length should be `num_layers` and
                `num_heads[i]` is the number of attention heads at the
                `i-th` layer. If set to an integer, all layers will
                have this many attention heads.
            bias: Whether to learn an additive bias.
            activation: Activation function used in the model.
            dropout: Dropout rate of the model.
            dropout_attn: Dropout rate of the attention model.
            residual: Whether to enable the residual structure.
            val_ratio: A value between 0.0 and 1.0 representing the proportion
                of the dataset to include in the validation split.
            test_ratio: A value between 0.0 and 1.0 representing the proportion
                of the dataset to include in the test split.
            reverse_triplet: If set to `True`, will create edges from `head_id`
                to `tail_id`.
            reverse_relation: If set to `False`, the relation type of
                `(tail_id, head_id)` will be the same as the relation type of
                `(head_id, tail_id)`. If set to `True`, the relation type of
                `(tail_id, head_id)` will be the different from the relation
                type of `(head_id, tail_id)`.
            use_edges_user: Whether to use social network between users.
            batch_size_train: Batch size in the training stage.
            batch_size_val: Batch size in the validation stage.
                If not specified, `batch_size_train` will be used.
            batch_size_test: Batch size in the test stage.
                If not specified, 'batch_size_val' will be used.
            batch_size_predict: Batch size in the prediction stage.
                If not specified, 'batch_size_test' will be used.
            num_workers: How many subprocesses to use for data loading.
            pin_memory: If `True`, the data loader will copy Tensors
                into CUDA pinned memory before returning them.
        """
        super().__init__(optim_args)
        self.save_hyperparameters()
        # Arguments for dataloader
        self._batch_size_train = batch_size_train
        self._batch_size_val = batch_size_val or self._batch_size_train
        self._batch_size_test = batch_size_test or self._batch_size_val
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._num_users = 0
        self._num_entities = 0
        self._num_relations = 0
        self._dataset_train = None
        self._dataset_val = None
        self._dataset_test = None
        self.graph = None
        self.register_buffer("feats_edge", None)
        self._build_dataset(
            dataset,
            data_dir,
            val_ratio,
            test_ratio,
            reverse_triplet,
            reverse_relation,
            use_edges_user,
        )

        # Model architecture
        hidden_dims_node = ensure_list(hidden_dims_node, num_layers)
        hidden_dims_edge = ensure_list(
            hidden_dims_node or hidden_dims_node, num_layers
        )
        num_heads = ensure_list(num_heads, num_layers)
        self.embeddings_node = nn.Parameter(
            torch.empty(self.graph.number_of_nodes(), embed_dim)
        )
        feat_dim_edge = self.feats_edge.size(1)
        self.hgat_layers = nn.ModuleList()
        # disable residual at the first layer
        self.hgat_layers.append(
            SimpleHGAT(
                embed_dim,
                hidden_dims_node[0],
                feat_dim_edge,
                hidden_dims_node[0],
                bias=bias,
                activation=activation,
                num_heads=num_heads[0],
                dropout=dropout,
                dropout_attn=dropout_attn,
                residual=False,
            )
        )
        for idx in range(1, num_layers):
            self.hgat_layers.append(
                SimpleHGAT(
                    hidden_dims_node[idx - 1] * num_heads[idx - 1],
                    hidden_dims_node[idx],
                    feat_dim_edge,
                    hidden_dims_edge[idx],
                    bias=bias,
                    activation=activation,
                    num_heads=num_heads[idx],
                    dropout=dropout,
                    dropout_attn=dropout_attn,
                    residual=residual,
                )
            )
        self.register_buffer(
            "eps", torch.Tensor([torch.finfo(torch.float).tiny])
        )
        self.reset_parameters()

    def _build_graph(
        self,
        ratings: np.ndarray,
        triplets_kg: np.ndarray,
        reverse_triplet: bool,
        reverse_relation: bool,
        edges_user: Optional[np.ndarray] = None,
        reverse_edge_user: bool = False,
    ) -> dgl.DGLGraph:
        reverse_relation = reverse_triplet and reverse_relation
        num_feats = 0
        # `(src, dst)` -> a set of `feat_id`
        edge_feat_ids = defaultdict(set)

        # Self-loops of users
        for uid in range(self._num_users):
            edge_feat_ids[(uid, uid)].add(num_feats)
        num_feats += 1

        # Self-loops of entities
        for eid in range(self._num_users, self._num_users + self._num_entities):
            edge_feat_ids[(eid, eid)].add(num_feats)
        num_feats += 1

        if edges_user is not None:
            # Edges between users
            feat_id_rev = num_feats + 1
            for uid, vid in edges_user:
                edge_feat_ids[(uid, vid)].add(num_feats)
                if reverse_edge_user:
                    edge_feat_ids[(vid, uid)].add(feat_id_rev)
            num_feats += 2 if reverse_edge_user else 1

        # Ratings between users and entities.
        feat_id_rev = num_feats + 1 if reverse_relation else num_feats
        for uid, eid, _ in ratings:
            edge_feat_ids[(uid, eid)].add(num_feats)
            if reverse_triplet:
                edge_feat_ids[(eid, uid)].add(feat_id_rev)
        num_feats = feat_id_rev + 1

        # Relations between entities in the knowledge graph
        for eid_h, rid, eid_t in triplets_kg:
            edge_feat_ids[(eid_h, eid_t)].add(rid + num_feats)
            if reverse_triplet:
                rid_rev = rid + num_feats
                if reverse_relation:
                    rid_rev += self._num_relations
                edge_feat_ids[(eid_t, eid_h)].add(rid_rev)
        scale = 2 if reverse_relation else 1
        num_feats += scale * self._num_relations

        src = torch.zeros(len(edge_feat_ids), dtype=torch.long)
        dst = torch.zeros(len(edge_feat_ids), dtype=torch.long)
        feats_edge = torch.zeros(
            len(edge_feat_ids), num_feats, dtype=torch.float
        )
        for idx, ((s, d), feat_ids) in enumerate(edge_feat_ids.items()):
            src[idx] = s
            dst[idx] = d
            scale = np.sqrt(len(feat_ids))
            for feat_id in feat_ids:
                feats_edge[idx, feat_id] = 1.0 / scale
        self.graph = dgl.graph((src, dst))
        self.feats_edge = feats_edge

    def _build_dataset(
        self,
        dataset: str,
        data_dir: str,
        val_ratio: float,
        test_ratio: float,
        reverse_triplet: bool,
        reverse_relation: bool,
        use_edges_user: bool = False,
    ):
        dataset = dataset.lower()
        edges_user = None
        reverse_edge_user = False
        if dataset == "lastfm":
            ret = read_data_lastfm(data_dir, use_edges_user=use_edges_user)
            ratings, triplets_kg = ret[0], ret[1]
            if use_edges_user:
                edges_user = ret[2]
                # The social network of Last.fm dataset is undirected.
                src, dst = edges_user[:, 0], edges_user[:, 1]
                src_bi = np.hstack([src, dst]).reshape(-1, 1)
                dst_bi = np.hstack([dst, src]).reshape(-1, 1)
                edges_user = np.unique(np.hstack([src_bi, dst_bi]), axis=0)
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported.")

        self._num_users = ratings[:, 0].max() + 1
        self._num_entities = triplets_kg[:, [0, 2]].max() + 1
        self._num_relations = triplets_kg[:, 1].max() + 1

        # Build a graph containing all relations where nodes are the union of
        # users, items and entities.
        # Edges are the union of the following relations:
        # 1. ratings between users and items/entities;
        # 2. relations between entities;
        # 3. self loops between all nodes.

        # We map entity ids from [0, num_entities - 1] to
        # [num_user, num_user + num_entities - 1]
        ratings[:, 1] = ratings[:, 1] + self._num_users
        triplets_kg[:, 0] = triplets_kg[:, 0] + self._num_users
        triplets_kg[:, 2] = triplets_kg[:, 2] + self._num_users

        num_val = int(val_ratio * ratings.shape[0])
        num_test = int(test_ratio * ratings.shape[0])
        ratings, ratings_test = train_test_split(
            ratings, test_size=num_test, shuffle=True, stratify=ratings[:, -1]
        )
        ratings_train, ratings_val = train_test_split(
            ratings, test_size=num_val, shuffle=True, stratify=ratings[:, -1]
        )
        self._graph = self._build_graph(
            ratings_train,
            triplets_kg,
            reverse_triplet,
            reverse_relation,
            edges_user,
            reverse_edge_user,
        )
        self._dataset_train = CTRPredictionDataset(
            torch.as_tensor(ratings_train, dtype=torch.long)
        )
        self._dataset_val = CTRPredictionDataset(
            torch.as_tensor(ratings_val, dtype=torch.long)
        )
        self._dataset_test = CTRPredictionDataset(
            torch.as_tensor(ratings_test, dtype=torch.long)
        )

    def reset_parameters(self):
        a = math.sqrt(3 / self.embeddings_node.size(1))
        nn.init.uniform_(self.embeddings_node, -a, a)

    def train_dataloader(self) -> DataLoader:
        return create_dataloader(
            self._dataset_train,
            self._batch_size_train,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return create_dataloader(
            self._dataset_val,
            self._batch_size_val,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return create_dataloader(
            self._dataset_test,
            self._batch_size_test,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def forward(self) -> torch.Tensor:
        outputs_all = []
        hidden = F.normalize(
            self.embeddings_node.float(), p=2.0, dim=1, eps=self.eps
        )
        outputs_all.append(hidden)
        for layer in self.hgat_layers:
            hidden = layer(hidden, self.feats_edge, self.graph)
            hidden = F.normalize(hidden, p=2.0, dim=1, eps=self.eps)
            outputs_all.append(hidden)
        return torch.cat(outputs_all, dim=1)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, int]]:
        embeddings_node = self.forward()
        embeddings_user = embeddings_node[batch["uids"]]
        embeddings_item = embeddings_node[batch["iids"]]
        logits = torch.sum(embeddings_user * embeddings_item, dim=1)
        loss = F.binary_cross_entropy_with_logits(
            logits, target=batch["labels"].float()
        )
        return {"loss": loss, "wt": logits.numel()}

    def training_epoch_end(
        self, outputs: Sequence[Dict[str, Union[torch.Tensor, int]]]
    ):
        loss = 0.0
        wt = 0.0
        for out in outputs:
            loss += out["loss"].item() * out["wt"]
            wt += out["wt"]
        wt = float(wt)
        self.log("loss", loss / wt)
        self.log("wt", wt)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        embeddings_node = self.forward()
        embeddings_user = embeddings_node[batch["uids"]]
        embeddings_item = embeddings_node[batch["iids"]]
        logits = torch.sum(embeddings_user * embeddings_item, dim=1)
        probs = torch.sigmoid(logits)
        return {"probs": probs, "labels": batch["labels"]}

    def validation_epoch_end(
        self, outputs: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[int, float]]:
        metrics = self.evaluate(outputs)
        for k, v in metrics.items():
            self.log(f"{k}_val", float(v))
        return metrics

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self, outputs: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[int, float]]:
        metrics = self.evaluate(outputs)
        for k, v in metrics.items():
            self.log(f"{k}_test", float(v))
        return metrics

    def evaluate(
        self, outputs: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[int, float]]:
        probs = []
        labels = []
        for out in outputs:
            probs.append(out["probs"].detach().cpu().numpy())
            labels.append(out["labels"].detach().cpu().numpy())
        probs = np.hstack(probs)
        preds = (probs >= 0.5).astype(np.int32)
        labels = np.hstack(labels)
        return {
            "num_samples": probs.size,
            "auc": roc_auc_score(y_true=labels, y_score=probs),
            "f1": f1_score(y_true=labels, y_pred=preds),
        }

    # override method DeviceDtypeModuleMixin.__update_properties
    def _DeviceDtypeModuleMixin__update_properties(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if device is not None:
            self.graph = self.graph.to(device)
        super()._DeviceDtypeModuleMixin__update_properties(device, dtype)
