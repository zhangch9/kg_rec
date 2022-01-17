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


class RippleDataset(BaseDataset):
    def __init__(self, ratings: torch.Tensor, ripple_sets: torch.Tensor):
        super().__init__()
        self._ratings = ratings
        self._ripple_sets = ripple_sets

        self._size = self._ratings.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._ratings[index]
        ripple_set = self._ripple_sets[uid]
        return {
            "iids": iid,
            "ripple_sets": ripple_set,
            "labels": label,
        }

    def __len__(self):
        return self._size


class CTRPredictionDataset(BaseDataset):
    def __init__(self, ratings: torch.Tensor):
        self._ratings = ratings
        self._size = ratings.size(0)

    def __len__(self):
        return self._size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._ratings[index]
        return {"uids": uid, "iids": iid, "labels": label}


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
                torch.ones_like(iids_pos, dtype=torch.float),
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
