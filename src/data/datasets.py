# -*- coding: utf-8 -*-
"""Subclasses of torch.utils.data.Dataset."""


from typing import Dict, Iterable, List, Optional, Sequence

import torch
from jsonargparse.typing import PositiveInt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    BatchSampler,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import default_collate

import dgl  # isort: skip
from dgl.sampling import sample_neighbors  # isort: skip


class BaseDataset(Dataset):
    def collate(
        self, samples: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return default_collate(samples)

    def get_batches(
        self,
        batch_size: PositiveInt,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> Iterable[List]:
        if shuffle:
            sampler = RandomSampler(self, replacement=False)
        else:
            sampler = SequentialSampler(self)
        return BatchSampler(sampler, batch_size, drop_last=drop_last)


class KGCNDataset(BaseDataset):
    def __init__(
        self,
        ratings: torch.Tensor,
        graph: dgl.DGLGraph,
        feats_edge: torch.Tensor,
        num_neighbors: int,
    ):
        super().__init__()
        self._ratings = ratings
        self._graph = graph
        self._feat_edges = feats_edge

        self._num_neighbors = num_neighbors

        self._size = self._ratings.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._ratings[index]
        subgraph = sample_neighbors(
            self._graph, [iid], self._num_neighbors, replace=False
        )
        src, _ = subgraph.edges(order="eid")
        feats_edge = self._feat_edges[subgraph.edata[dgl.EID]]
        return {
            "uids": uid,
            "iids": iid,
            "labels": label,
            "neighbors": src,
            "feats_edge": feats_edge,
            "num_neighbors": src.size(0),
        }

    def __len__(self):
        return self._size

    def collate(
        self, samples: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        sample = samples[0]
        batch = {}
        for key in sample:
            if key in {"neighbors", "feats_edge"}:
                batch[key] = pad_sequence(
                    [s[key] for s in samples],
                    batch_first=True,
                    padding_value=0.0,
                )
            else:
                batch[key] = default_collate([s[key] for s in samples])
        return batch


class RippleDataset(BaseDataset):
    def __init__(
        self, ratings: torch.Tensor, ripple_set: Dict[int, torch.Tensor]
    ):
        super().__init__()
        self._ratings = ratings
        # uid -> a `torch.Tensor` object of shape [max_hop, 3, num_neighbors]
        self._ripple_set = ripple_set

        self._size = self._ratings.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._ratings[index]
        ripple_set_u = self._ripple_set[uid.item()]
        return {
            "iids": iid,
            "ripple_sets": ripple_set_u,
            "labels": label,
        }

    def __len__(self):
        return self._size


class BPRDataset(BaseDataset):
    def __init__(
        self,
        uids: torch.Tensor,
        iids: torch.Tensor,
        graph_rating: dgl.DGLGraph,
        graph_rating_test: Optional[dgl.DGLGraph] = None,
    ):
        super().__init__()
        self._uids = uids
        self._iids = iids
        self._graph_rating = graph_rating
        self._graph_rating_test = graph_rating_test

        self._size = self._uids.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid = self._uids[index]
        iids_pos = self._graph_rating.successors(uid)
        mask = torch.isin(self._iids, iids_pos, invert=True)
        if self._graph_rating_test is None:
            # training stage
            idx_pos = torch.multinomial(
                torch.ones_like(iids_pos, dtype=torch.float32),
                num_samples=1,
                replacement=False,
            )
            idx_neg = torch.multinomial(
                mask.float(), num_samples=1, replacement=False
            )
            return {
                "uids": uid,
                "iids_pos": iids_pos[idx_pos],
                "iids_neg": self._iids[idx_neg],
            }

        iids_pos_test = self._graph_rating_test.successors(uid)
        return {
            "uids": uid,
            "iids_pos": iids_pos_test,
            "num_pos": iids_pos_test.size(0),
            "mask": mask,
            # "iids_seen": iids_pos,
            # "num_seen": iids_pos.size(0),
        }

    def __len__(self) -> int:
        return self._size

    def collate(
        self, samples: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if self._graph_rating_test is None:
            return super().collate(samples)

        sample = samples[0]
        batch = {}
        for key in sample:
            if key in {"iids_pos", "iids_seen"}:
                batch[key] = pad_sequence(
                    [s[key] for s in samples],
                    batch_first=True,
                    padding_value=0.0,
                )
            else:
                batch[key] = default_collate([s[key] for s in samples])
        return batch
