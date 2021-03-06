# -*- coding: utf-8 -*-
"""Contains functions for data processing."""


from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from scipy import sparse as sp
from torch.utils.data import DataLoader

from .datasets import BaseDataset


def read_data_lastfm(
    data_dir: Union[str, Path], use_edges_user: bool = False
) -> Tuple[np.ndarray, ...]:
    data_dir = Path(data_dir).expanduser().resolve()
    ratings = np.unique(
        np.loadtxt(data_dir.joinpath("ratings.txt"), dtype=np.int64), axis=0
    )
    triplets_kg = np.unique(
        np.loadtxt(data_dir.joinpath("triplets_kg.txt"), dtype=np.int64), axis=0
    )

    if use_edges_user:
        edges_user = np.unique(
            np.loadtxt(data_dir.joinpath("edges_uu.txt"), dtype=np.int64),
            axis=0,
        )
        return ratings, triplets_kg, edges_user
    return ratings, triplets_kg


def read_data_kgat(
    data_dir: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _read_ratings(path: Union[str, Path]) -> np.ndarray:
        ratings = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ids = list(map(int, line.strip().split()))
                uid, iids = ids[0], set(ids[1:])
                for iid in iids:
                    ratings.append((uid, iid))
        return np.asarray(ratings)

    def _read_kg(path: Union[str, Path]) -> np.ndarray:
        return np.loadtxt(path, dtype=np.int32)

    data_dir = Path(data_dir).expanduser().resolve()
    ratings_train = _read_ratings(data_dir.joinpath("train.txt"))
    ratings_test = _read_ratings(data_dir.joinpath("test.txt"))
    triplets_kg = _read_kg(data_dir.joinpath("kg_final.txt"))
    return ratings_train, ratings_test, triplets_kg


def adj_list_from_triplets(
    triplets: np.ndarray,
    reverse_triplet: bool = False,
    reverse_relation: bool = False,
) -> Dict[int, Tuple[int, int]]:
    """Constructs an adjacent lists from triplets.
    Each triplet is in the form of `(head_id, relation_id, tail_id)`.

    Args:
        triplets: A `numpy.ndarray`` object of shape `[num_triplets, 3]`, each
            row is a `(head_id, relation_id, tail_id)` triplet.
        reverse_triplet: If set to `True`, will add `head_id` to the adjacency
            list of `tail_id`.
        reverse_relation: If set to `True`, will create a new `rev_relation_id`
            for each `relation_id`.
    """
    assert triplets.shape[1] == 3

    num_relations = triplets[:, 1].max() + 1
    adj_list = defaultdict(list)
    for eid_h, rid, eid_t in triplets:
        adj_list[eid_h].append((rid, eid_t))
        if reverse_triplet:
            if reverse_relation:
                rid_rev = rid + num_relations
            else:
                rid_rev = rid
            adj_list[eid_t].append((rid_rev, eid_h))
    return adj_list


def normalize_matrix(m: sp.spmatrix, r: float) -> sp.csr_matrix:
    m = m.tocsr().astype(np.float32)
    deg_out = m.sum(axis=1).A.flatten()
    if r > 0:
        deg_l = np.power(deg_out, -r)
        deg_l[np.isinf(deg_l)] = 0.0
        deg_l = sp.diags(deg_l)
        m = deg_l.dot(m)
    r = 1 - r
    if r > 0:
        deg_r = np.power(deg_out, -r)
        deg_r[np.isinf(deg_r)] = 0.0
        deg_r = sp.diags(deg_r)
        m = m.dot(deg_r)
    return m


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    if not isinstance(dataset, BaseDataset):
        raise ValueError("'dataset' must be a 'BaseDataset' object.")
    batches = dataset.get_batches(
        batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=dataset.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
