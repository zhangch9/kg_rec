# -*- coding: utf-8 -*-
"""Implements the SimpleHCN model."""


import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from jsonargparse import Namespace
from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from torch import nn
from torch.nn import functional as F

from .base import BaseModel
from .layers import SimpleHGAT
from .utils import ensure_list

import dgl  # isort: skip


def _compute_recall(k: int, hit: Sequence[int], total: int):
    hit = np.asfarray(hit[:k])
    return hit.sum() / total


def _compute_dcg(k: int, hit: Sequence[int]):
    hit = np.asfarray(hit[:k])
    if hit.size:
        return np.sum(hit / np.log2(np.arange(2, hit.size + 2)))
    return 0.0


def _compute_ndcg(k: int, hit: Sequence[int]):
    sorted_hit = sorted(hit, reverse=True)
    dcg_max = _compute_dcg(k, sorted_hit)
    if dcg_max == 0.0:
        return 0.0
    return _compute_dcg(k, hit) / dcg_max


class SimpleHGN(BaseModel):
    def __init__(
        self,
        optim_args: Namespace,
        graph: dgl.DGLGraph,
        iids: torch.Tensor,
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
    ):
        """Constructs a SimpleHGN model.

        Args:
            optim_args: Arguments for optimization.
            graph: A `dgl.DGLGraph` object.
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
        """
        super().__init__(optim_args)

        self._graph = graph
        self.register_buffer("feats_edge", graph.edata.pop("feats"))
        # This `iids` is the same as that in the BPRDataset.
        self.register_buffer("iids", iids)

        hidden_dims_node = ensure_list(hidden_dims_node, num_layers)
        hidden_dims_edge = ensure_list(
            hidden_dims_node or hidden_dims_node, num_layers
        )
        num_heads = ensure_list(num_heads, num_layers)

        # model parameters
        self.embeddings_node = nn.Parameter(
            torch.empty(graph.number_of_nodes(), embed_dim)
        )
        self.register_buffer("embeddings_node_final", None)
        self.register_buffer("embeddings_item", None)

        feat_dim_edge = self.feats_edge.size(1)
        self.hgat_layers = nn.ModuleList()
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
            "eps", torch.Tensor([torch.finfo(torch.float32).tiny])
        )
        self.register_buffer(
            "inf", torch.Tensor([torch.finfo(torch.float32).max])
        )
        self.reset_parameters()

        self.recall = None
        self.ndcg = None

    def reset_parameters(self):
        fan_out = self.embeddings_node.size(1)
        nn.init.normal_(self.embeddings_node, std=1 / math.sqrt(fan_out))

    def forward(self) -> torch.Tensor:
        outputs_all = []
        hidden = F.normalize(
            self.embeddings_node.float(), p=2.0, dim=1, eps=self.eps
        )
        outputs_all.append(hidden)
        for layer in self.hgat_layers:
            hidden = layer(hidden, self.feats_edge, self._graph)
            hidden = F.normalize(hidden, p=2.0, dim=1, eps=self.eps)
            outputs_all.append(hidden)
        return torch.cat(outputs_all, dim=1)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        embeddings_node = self.forward()
        embeddings_user = embeddings_node[batch["uids"]]
        embeddings_item_pos = embeddings_node[batch["iids_pos"]]
        embeddings_item_neg = embeddings_node[batch["iids_neg"]]
        scores_pos = (embeddings_user * embeddings_item_pos).sum(dim=1)
        scores_neg = (embeddings_user * embeddings_item_neg).sum(dim=1)
        bpr_loss = F.softplus(scores_neg - scores_pos).mean()
        return bpr_loss
        # embed_reg = (
        #     1e-5
        #     * (
        #         (embeddings_user * embeddings_user).sum()
        #         + (embeddings_item_pos * embeddings_item_pos).sum()
        #         + (embeddings_item_neg * embeddings_item_neg).sum()
        #     )
        #     / embeddings_user.size(0)
        # )
        # return bpr_loss + embed_reg

    def training_epoch_end(self, outputs: Sequence[torch.Tensor]):
        losses = []
        for out in outputs:
            losses.append(out["loss"].item())
        print(np.mean(losses))

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        if self.embeddings_node_final is None:
            self.embeddings_node_final = self.forward()
            self.embeddings_item = self.embeddings_node_final[self.iids]
        embeddings_user = self.embeddings_node_final[batch["uids"]]
        ratings = torch.matmul(
            embeddings_user,
            torch.t(self.embeddings_item).contiguous(),
        )
        mask = batch["mask"].float()
        # masks items in the history
        ratings = mask * ratings + (mask - 1) * self.inf
        topk = torch.topk(ratings, 100, dim=1, largest=True, sorted=True)
        return {
            "topk_iids": self.iids[topk[1]],
            "iids_pos": batch["iids_pos"],
            "num_pos": batch["num_pos"],
            "num_unseen": mask.sum(dim=1),
            # "iids_seen": batch["iids_seen"],
            # "num_seen": batch["num_seen"],
        }

    def validation_step_end(self, outputs: Dict[str, torch.Tensor]):
        top_iids = outputs["topk_iids"].detach().cpu().numpy()
        iids_pos = outputs["iids_pos"].detach().cpu().numpy()
        num_pos = outputs["num_pos"].detach().cpu().numpy()
        num_unseen = outputs["num_unseen"].detach().cpu().numpy()
        # iids_seen = outputs["iids_seen"].detach().cpu().numpy()
        # num_seen = outputs["num_seen"].detach().cpu().numpy()
        for idx_u, top_iids_u in enumerate(top_iids):
            assert len(top_iids_u) <= num_unseen[idx_u]
            num_pos_u = num_pos[idx_u]
            hit = []
            iids_pos_u = set(iids_pos[idx_u][:num_pos_u].tolist())
            for iid in top_iids_u:
                if iid in iids_pos_u:
                    hit.append(1)
                else:
                    hit.append(0)
            for k in [20, 40]:
                rec = _compute_recall(k, hit, num_pos_u)
                ndcg = _compute_ndcg(k, hit)
                self.recall[k].append(rec)
                self.ndcg[k].append(ndcg)
            # num_seen_u = num_seen[idx_u]
            # iids_seen_u = set(iids_seen[idx_u][:num_seen_u].tolist())
            # size = min(len(top_iids_u), num_seen_u)
            # for iid in top_iids_u[:size]:
            #     assert iid in iids_seen_u
            # print(len(iids_seen_u & iids_pos_u))

    def validation_epoch_end(self, outputs: Sequence[Any]):
        print(
            "\n".join(
                [
                    f"recall@20 = {np.mean(self.recall[20])}",
                    f"recall@40={np.mean(self.recall[40])}",
                    f"ndcg@20 = {np.mean(self.ndcg[20])}",
                    f"ndcg@40={np.mean(self.ndcg[40])}",
                    f"size = {len(self.recall[20])}",
                    f"size ={len(self.recall[40])}",
                    f"size = {len(self.ndcg[20])}",
                    f"size ={len(self.ndcg[40])}",
                ]
            )
        )

    # methods of ModelHook
    def on_validation_epoch_start(self):
        self.embeddings_node_final = None
        self.embeddings_item = None
        self.recall = {20: [], 40: []}
        self.ndcg = {20: [], 40: []}

    # override method DeviceDtypeModuleMixin.__update_properties
    def _DeviceDtypeModuleMixin__update_properties(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if device is not None:
            self._graph = self._graph.to(device)
        super()._DeviceDtypeModuleMixin__update_properties(device, dtype)
