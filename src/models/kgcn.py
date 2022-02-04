# -*- coding: utf-8 -*-
"""Implements the KGCN model."""


import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
from ..data.utils import adj_list_from_triplets, read_data_lastfm
from .base import BaseModel
from .layers import KGCLayer


class KGCN(BaseModel):
    def __init__(
        self,
        optim_args: Namespace,
        dataset: str,
        data_dir: str,
        num_neighbors: PositiveInt,
        embed_dim: PositiveInt,
        num_layers: PositiveInt,
        aggregation: str = "cat",
        weight_ls: float = 0.0,
        bias: bool = True,
        dropout: ClosedUnitInterval = 0.0,
        dropout_attn: ClosedUnitInterval = 0.0,
        val_ratio: OpenUnitInterval = 0.2,
        test_ratio: OpenUnitInterval = 0.2,
        batch_size_train: PositiveInt = 32,
        batch_size_val: Optional[PositiveInt] = None,
        batch_size_test: Optional[PositiveInt] = None,
        num_workers: NonNegativeInt = 0,
        pin_memory: bool = True,
    ):
        """Constructs a KGCN model.

        Args:
            optim_args: Arguments for optimization.
            dataset: Name of the dataset.
            data_dir: Directory containing the dataset files.
            num_neighbors: Size of the neighborhood for each hop.
            embed_dim: Dimension of the embedding space.
            num_layers: Number of hidden layers.
            aggregation: Aggregation method to use ("sum" or "cat").
            weight_ls: Weight of the label smoothness regularizer. If set to a
            negative number, the label smoothness regularizer will be ignored.
            bias: Whether to learn an additive bias.
            activation: Activation function used in the model.
            dropout: Dropout rate of the model.
            dropout_attn: Dropout rate of the attention model.
            val_ratio: A value between 0.0 and 1.0 representing the proportion
                of the dataset to include in the validation split.
            test_ratio: A value between 0.0 and 1.0 representing the proportion
                of the dataset to include in the test split.
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
        self._num_layers = num_layers
        self._weight_ls = weight_ls
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
        self.register_buffer("adj_list_entity", None)
        self.register_buffer("adj_list_relation", None)
        # for label smoothness regularizer
        self.register_buffer("indices_labeled", None)
        self.register_buffer("labels_for_ls", None)
        self._build_dataset(
            dataset, data_dir, num_neighbors, val_ratio, test_ratio
        )

        # Model architecture
        self.embeddings_user = nn.Parameter(
            torch.empty(self._num_users, embed_dim)
        )
        self.embeddings_entity = nn.Parameter(
            torch.empty(self._num_entities, embed_dim)
        )
        self.embeddings_relation = nn.Parameter(
            torch.empty(self._num_relations, embed_dim)
        )
        self.kgc_layers = nn.ModuleList()
        for _ in range(self._num_layers - 1):
            self.kgc_layers.append(
                KGCLayer(
                    embed_dim,
                    embed_dim,
                    bias=bias,
                    aggregation=aggregation,
                    activation="relu",
                    dropout=dropout,
                    dropout_attn=dropout_attn,
                )
            )
        # The last layer uses the `tanh` activation.
        self.kgc_layers.append(
            KGCLayer(
                embed_dim,
                embed_dim,
                bias=bias,
                aggregation=aggregation,
                activation="tanh",
                dropout=dropout,
                dropout_attn=dropout_attn,
            )
        )
        self.reset_parameters()

    def _build_dataset(
        self,
        dataset: str,
        data_dir: str,
        num_neighbors: int,
        val_ratio: float,
        test_ratio: float,
    ):
        dataset = dataset.lower()
        if dataset in {"lastfm", "yelp"}:
            ratings, triplets_kg = read_data_lastfm(data_dir)
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported")

        self._num_users = ratings[:, 0].max() + 1
        self._num_entities = triplets_kg[:, [0, 2]].max() + 1
        self._num_relations = triplets_kg[:, 1].max() + 1

        num_val = int(val_ratio * ratings.shape[0])
        num_test = int(test_ratio * ratings.shape[0])
        ratings, ratings_test = train_test_split(
            ratings, test_size=num_test, shuffle=True, stratify=ratings[:, -1]
        )
        ratings_train, ratings_val = train_test_split(
            ratings, test_size=num_val, shuffle=True, stratify=ratings[:, -1]
        )

        adj_list_kg = adj_list_from_triplets(triplets_kg, reverse_triplet=True)
        self.adj_list_entity = torch.empty(
            (self._num_entities, num_neighbors), dtype=torch.long
        )
        self.adj_list_relation = torch.empty(
            (self._num_entities, num_neighbors), dtype=torch.long
        )
        for eid_h, rid_eid_list in adj_list_kg.items():
            pop = len(rid_eid_list)
            indices = np.random.choice(
                pop, size=num_neighbors, replace=pop < num_neighbors
            )
            for i, idx in enumerate(indices):
                self.adj_list_entity[eid_h, i] = rid_eid_list[idx][1]
                self.adj_list_relation[eid_h, i] = rid_eid_list[idx][0]

        self._dataset_train = CTRPredictionDataset(
            torch.as_tensor(ratings_train, dtype=torch.long)
        )
        self._dataset_val = CTRPredictionDataset(
            torch.as_tensor(ratings_val, dtype=torch.long)
        )
        self._dataset_test = CTRPredictionDataset(
            torch.as_tensor(ratings_test, dtype=torch.long)
        )

        if self._weight_ls > 0.0:
            # assigns positive `(uid, eid)` pairs to 1.0, negative `(uid, eid)`
            # pairs to 0.0.
            self.indices_labeled = torch.as_tensor(
                ratings_train[:, 0] * self._num_entities + ratings_train[:, 1]
            )
            self.labels_for_ls = torch.sparse_coo_tensor(
                self.indices_labeled.view(1, -1).contiguous(),
                torch.as_tensor(ratings_train[:, 2], dtype=torch.float),
                size=(self._num_users * self._num_entities,),
            ).coalesce()

    def reset_parameters(self):
        a = math.sqrt(3 / self.embeddings_user.size(1))
        nn.init.uniform_(self.embeddings_user, -a, a)

        a = math.sqrt(3 / self.embeddings_entity.size(1))
        nn.init.uniform_(self.embeddings_entity, -a, a)

        a = math.sqrt(3 / self.embeddings_relation.size(1))
        nn.init.uniform_(self.embeddings_relation, -a, a)

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

    def _get_neighbors(
        self, iids: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_size = iids.size(0)
        # The shape of `eids_per_hop[i]` is `[batch_size, num_neighbors ** i]`.
        # The shape of `rid_per_hop[i]` is
        # `[batch_size, num_neighbors ** (i + 1)]`.
        eids_per_hop, rids_per_hop = [iids.view(batch_size, 1).contiguous()], []
        for h in range(self._num_layers):
            eids = (
                self.adj_list_entity[eids_per_hop[h]]
                .view(batch_size, -1)
                .contiguous()
            )
            rids = (
                self.adj_list_relation[eids_per_hop[h]]
                .view(batch_size, -1)
                .contiguous()
            )
            eids_per_hop.append(eids)
            rids_per_hop.append(rids)
        return eids_per_hop, rids_per_hop

    def _get_embeddings(
        self,
        uids: torch.Tensor,
        eids_per_hop: Sequence[torch.Tensor],
        rids_per_hop: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        embeddings_user = self.embeddings_user[uids]

        embeddings_entity_per_hop = [
            self.embeddings_entity[eids] for eids in eids_per_hop
        ]
        embeddings_relation_per_hop = [
            self.embeddings_relation[rids] for rids in rids_per_hop
        ]
        return (
            embeddings_user,
            embeddings_entity_per_hop,
            embeddings_relation_per_hop,
        )

    def _propagate_label(
        self,
        embeddings_user: torch.Tensor,
        labels_per_hop: Sequence[torch.Tensor],
        masks_per_hop: Sequence[torch.Tensor],
        embeddings_relation_per_hop: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings_user: A `torch.Tensor` object of shape
                `[batch_size, dim]`
            labels_per_hop: A list of `torch.Tensor` objects the
                `i`-th entry of which has the shape
                `[batch_size, num_neighbor ** i]`
            mask: A list of `torch.Tensor` objects the
                `i`-th entry of which has the shape
                `[batch_size, num_neighbor ** i, dim]`
            embeddings_ent_per_hop: A list of `torch.Tensor` objects the
                `i`-th entry of which has the shape
                `[batch_size, num_neighbor ** (i + 1), dim]`
        """

        def _propagate(
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            mask: torch.Tensor,
            relations: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                queries: A `torch.Tensor` object of shape `[batch_size, dim]`
                keys: A `torch.Tensor` objects of shape
                    `[batch_size, num_values * num_neighbors]`
                values: A `torch.Tensor` objects of shape
                    `[batch_size, num_values]`
                mask: A `torch.Tensor` objects of shape
                    `[batch_size, num_values]`
                relations: A `torch.Tensor` objects of shape
                    `[batch_size, num_values * num_neighbors, dim]`
            """
            batch_size, dim = queries.size()
            num_values = values.size(1)

            queries = queries.view(batch_size, 1, 1, dim).contiguous()
            keys = keys.view(batch_size, num_values, -1).contiguous()
            relations = relations.view(
                batch_size, num_values, -1, dim
            ).contiguous()

            logits_rel = torch.sum(queries * relations, dim=3)
            probs = F.softmax(logits_rel, dim=2)
            outputs = torch.sum(probs * keys, dim=2)
            mask = mask.float()
            return mask * values + (1 - mask) * outputs

        inputs_label = labels_per_hop
        for idx_l in range(self._num_layers):
            outputs_label = []
            for idx in range(self._num_layers - idx_l):
                outputs_label.append(
                    _propagate(
                        embeddings_user,
                        inputs_label[idx + 1],
                        inputs_label[idx],
                        masks_per_hop[idx],
                        embeddings_relation_per_hop[idx],
                    )
                )
            inputs_label = outputs_label
        return inputs_label[0]

    def forward(
        self,
        embeddings_user: torch.Tensor,
        embeddings_entity_per_hop: Sequence[torch.Tensor],
        embeddings_relation_per_hop: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings_user: A `torch.Tensor` object of shape
                `[batch_size, dim]`
            embeddings_entity_per_hop: A list of `torch.Tensor` objects the
                `i`-th entry of which has the shape
                `[batch_size, num_neighbor ** i, dim]`
            embeddings_relation_per_hop: A list of `torch.Tensor` objects the
                `i`-th entry of which has the shape
                `[batch_size, num_neighbor ** (i + 1), dim]`
        """
        inputs_entity = embeddings_entity_per_hop
        inputs_relation = embeddings_relation_per_hop
        for idx_l in range(self._num_layers):
            outputs_entity = []
            for idx in range(self._num_layers - idx_l):
                outputs_entity.append(
                    self.kgc_layers[idx_l](
                        embeddings_user,
                        inputs_entity[idx + 1],
                        inputs_entity[idx],
                        inputs_relation[idx],
                    )
                )
            inputs_entity = outputs_entity
        embeddings_entity = inputs_entity[0].squeeze(1).contiguous()
        logits = torch.sum(embeddings_user * embeddings_entity, dim=1)
        return logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        uids, iids = batch["uids"], batch["iids"]
        eids_per_hop, rids_per_hop = self._get_neighbors(iids)
        (
            embeddings_user,
            embeddings_entity_per_hop,
            embeddings_relation_per_hop,
        ) = self._get_embeddings(uids, eids_per_hop, rids_per_hop)
        logits = self.forward(
            embeddings_user,
            embeddings_entity_per_hop,
            embeddings_relation_per_hop,
        )
        loss_cls = F.binary_cross_entropy_with_logits(
            logits, target=batch["labels"].float()
        )
        wt_cls = float(logits.numel())
        loss = 0 + loss_cls
        outputs = {
            "loss_cls": loss_cls.detach(),
            "wt_cls": wt_cls,
        }
        if self._weight_ls > 0.0:
            loss_ls = 0.0
            wt_ls = 0
            batch_size = uids.size(0)
            uids = uids.view(batch_size, 1).contiguous()

            labels_per_hop = []
            masks_per_hop = []
            indices_holdout = None
            for eids in eids_per_hop:
                indices = uids * self._num_entities + eids
                if indices_holdout is None:
                    indices_holdout = indices.view(batch_size, 1).contiguous()

                # get labels of `eids` that are not heldout
                has_label = torch.logical_and(
                    torch.isin(indices, self.indices_labeled),
                    (indices - indices_holdout).bool(),
                )
                masks_per_hop.append(has_label)
                # assigns unknown `(uid, eid)` pairs to 0.5
                labels = 0.5 * torch.ones_like(indices, dtype=torch.float)
                labels[has_label] = self.labels_for_ls.index_select(
                    0, indices[has_label]
                ).to_dense()
                labels_per_hop.append(labels)

            logits = self._propagate_label(
                embeddings_user,
                labels_per_hop,
                masks_per_hop,
                embeddings_relation_per_hop,
            )
            logits = logits.squeeze(1).contiguous()

            loss_ls = F.binary_cross_entropy(
                logits, target=batch["labels"].float()
            )
            wt_ls = float(logits.numel())
            loss += self._weight_ls * loss_ls
            outputs["loss_ls"] = loss_ls.detach()
            outputs["wt_ls"] = wt_ls
        outputs["loss"] = loss
        return outputs

    def training_epoch_end(
        self, outputs: Sequence[Dict[str, Union[torch.Tensor, float]]]
    ):
        loss = 0.0
        loss_cls = 0.0
        wt_cls_cum = 0.0
        loss_ls = 0.0
        wt_ls_cum = 0.0
        for out in outputs:
            wt_cls = out["wt_cls"]
            loss += out["loss"].item() * wt_cls
            loss_cls += out["loss_cls"].item() * wt_cls
            wt_cls_cum += wt_cls
            if self._weight_ls > 0.0:
                wt_ls = out["wt_ls"]
                loss_ls += out["loss_ls"].item() * wt_ls
                wt_ls_cum += wt_ls
        self.log("loss", loss / wt_cls_cum)
        self.log("loss_cls", loss_cls / wt_cls_cum)
        self.log("wt_cls", wt_cls_cum)
        if self._weight_ls > 0.0:
            self.log("loss_ls", loss_ls / wt_ls_cum)
            self.log("wt_ls", wt_ls_cum)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        uids, iids = batch["uids"], batch["iids"]
        eids_per_hop, rids_per_hop = self._get_neighbors(iids)
        (
            embeddings_user,
            embeddings_entity_per_hop,
            embeddings_relation_per_hop,
        ) = self._get_embeddings(uids, eids_per_hop, rids_per_hop)
        logits = self.forward(
            embeddings_user,
            embeddings_entity_per_hop,
            embeddings_relation_per_hop,
        )
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
        preds = (probs >= 0.5).astype(np.float32)
        labels = np.hstack(labels)
        return {
            "num_samples": probs.size,
            "auc": roc_auc_score(y_true=labels, y_score=probs),
            "f1": f1_score(y_true=labels, y_pred=preds),
        }
