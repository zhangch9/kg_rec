# -*- coding: utf-8 -*-
"""Contains layers used for building neural networks."""


import math
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .utils import get_activation

import dgl  # isort: skip
from dgl import function as dgl_fn  # isort: skip
from dgl.nn.pytorch import edge_softmax  # isort: skip


class SimpleHGAT(nn.Module):
    def __init__(
        self,
        input_dim_node: int,
        output_dim_node: int,
        input_dim_edge: int,
        hidden_dim_edge: int,
        num_heads: int = 1,
        bias: bool = True,
        activation: Optional[Union[str, Callable]] = None,
        dropout: float = 0.0,
        dropout_attn: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self._output_dim_node = output_dim_node
        self._hidden_dim_edge = hidden_dim_edge
        self._num_heads = num_heads

        # model parameters
        output_dim_node_t = output_dim_node * num_heads
        hidden_dim_edge_t = hidden_dim_edge * num_heads
        self.fc_node = nn.Linear(input_dim_node, output_dim_node_t, bias=False)
        self.fc_edge = nn.Linear(input_dim_edge, hidden_dim_edge_t, bias=False)
        self.attn_query = nn.Parameter(
            torch.empty(1, num_heads, output_dim_node)
        )
        self.attn_key = nn.Parameter(torch.empty(1, num_heads, output_dim_node))
        self.attn_edge = nn.Parameter(
            torch.empty(1, num_heads, hidden_dim_edge)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim_node_t))
        else:
            self.register_parameter("bias", None)
        self.activation = get_activation(activation)

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)

        if residual:
            if input_dim_node == output_dim_node_t:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(input_dim_node, output_dim_node_t)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_uniform_(self.attn_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_edge, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs_node: torch.Tensor,
        inputs_edge: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        inputs_node = inputs_node.float()
        inputs_edge = inputs_edge.float()
        with graph.local_scope():
            inputs_node = self.dropout(inputs_node)
            hidden_node = (
                self.fc_node(inputs_node)
                .view(-1, self._num_heads, self._output_dim_node)
                .contiguous()
            )
            hidden_edge = (
                self.fc_edge(inputs_edge)
                .view(-1, self._num_heads, self._hidden_dim_edge)
                .contiguous()
            )
            w_queries = (hidden_node * self.attn_query).sum(dim=2, keepdim=True)
            w_keys = (hidden_node * self.attn_key).sum(dim=2, keepdim=True)
            w_edges = (hidden_edge * self.attn_edge).sum(dim=2, keepdim=True)

            graph.srcdata.update({"values": hidden_node, "w_keys": w_keys})
            graph.dstdata.update({"w_queries": w_queries})
            graph.edata.update({"w_edges": w_edges})
            graph.apply_edges(dgl_fn.u_add_v("w_keys", "w_queries", "w_qk"))
            logits_attn = F.leaky_relu(
                graph.edata.pop("w_qk") + graph.edata.pop("w_edges")
            )
            graph.edata["w_attn"] = self.dropout_attn(
                edge_softmax(graph, logits_attn)
            )
            graph.update_all(
                dgl_fn.u_mul_e("values", "w_attn", "messages"),
                dgl_fn.sum("messages", "outputs"),
            )
            outputs = graph.dstdata.pop("outputs")
        outputs = outputs.view(
            -1, self._num_heads * self._output_dim_node
        ).contiguous()
        if self.bias is not None:
            outputs += self.bias
        outputs = self.activation(outputs)
        if hasattr(self, "residual"):
            outputs += self.residual(inputs_node)
        return outputs
