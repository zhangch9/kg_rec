# -*- coding: utf-8 -*-
"""Implements the RippleNet model."""


import logging
import math
import multiprocessing as mp
import time
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from jsonargparse import Namespace
from jsonargparse.typing import NonNegativeInt, OpenUnitInterval, PositiveInt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import RippleDataset, create_dataloader
from ..data.utils import adj_list_from_triplets, read_data_lastfm
from .base import BaseModel

logger = logging.getLogger(__name__)


class RippleNet(BaseModel):
    def __init__(
        self,
        optim_args: Namespace,
        dataset: str,
        data_dir: str,
        max_hop: PositiveInt,
        num_neighbors: PositiveInt,
        embed_dim: PositiveInt,
        weight_kg: float = 0.0,
        val_ratio: OpenUnitInterval = 0.2,
        test_ratio: OpenUnitInterval = 0.2,
        batch_size_train: PositiveInt = 32,
        batch_size_val: Optional[PositiveInt] = None,
        batch_size_test: Optional[PositiveInt] = None,
        num_workers: NonNegativeInt = 0,
        pin_memory: bool = True,
    ):
        """Constructs a RippleNet model.

        Args:
            optim_args: Arguments for optimization.
            dataset: Name of the dataset.
            data_dir: Directory containing the dataset files.
            max_hop: Stop expanding neighbors once this many hops is reached.
            num_neighbors: Size of the neighborhood for each hop.
            embed_dim: Dimension of the embedding space.
            weight_kg: Weight of the loss on knowledge graph triplets. If set
            to a negative number, the loss on knowledge graph triplets will be
            ignored.
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
        self._weight_kg = weight_kg
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
        self._build_dataset(
            dataset, data_dir, max_hop, num_neighbors, val_ratio, test_ratio
        )

        # Model architecture
        self.embeddings_entity = nn.Parameter(
            torch.empty(self._num_entities, embed_dim)
        )
        self.embeddings_relation = nn.Parameter(
            torch.empty(self._num_relations, embed_dim, embed_dim)
        )
        self.transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def _build_dataset(
        self,
        dataset: str,
        data_dir: str,
        max_hop: int,
        num_neighbors: int,
        val_ratio: float,
        test_ratio: float,
    ):
        dataset = dataset.lower()
        if dataset == "lastfm":
            ratings, triplets_kg = read_data_lastfm(data_dir)
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported.")

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

        history = defaultdict(list)  # `uid`` -> a list of `iids`
        for uid, iid, label in ratings_train:
            if label == 1:
                history[uid].append(iid)
        uids_valid = list(history.keys())

        # `ripple_sets`: a `torch.Tensor` of shape
        # `[num_users, max_hop, 3, num_neighbors]`.
        adj_list_kg = adj_list_from_triplets(triplets_kg, reverse_triplet=True)
        logger.info("===== Building ripple sets ... ======")
        start = time.perf_counter()
        ripple_sets = torch.empty(
            self._num_users, max_hop, 3, num_neighbors, dtype=torch.long
        )

        queue_task = mp.JoinableQueue()
        queue_res = mp.JoinableQueue()
        workers = []
        for _ in range(self._num_workers):
            worker = _Worker(
                queue_task, queue_res, max_hop, num_neighbors, adj_list_kg
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        num_tasks = 0
        for uid in uids_valid:
            num_tasks += 1
            queue_task.put((uid, history[uid]))
        for _ in range(self._num_workers):
            queue_task.put(None)

        queue_task.join()

        for _ in range(len(uids_valid)):
            uid, ripple_set = queue_res.get()
            queue_res.task_done()
            ripple_sets[uid] = torch.as_tensor(ripple_set, dtype=torch.long)
        queue_res.join()
        for worker in workers:
            worker.join()

        logger.info(
            "===== Building ripple sets: Done. "
            f"({time.perf_counter() - start:.3f}s) ======"
        )

        self._dataset_train = RippleDataset(
            torch.as_tensor(
                ratings_train[np.isin(ratings_train[:, 0], uids_valid)],
                dtype=torch.long,
            ),
            ripple_sets,
        )
        self._dataset_val = RippleDataset(
            torch.as_tensor(
                ratings_val[np.isin(ratings_val[:, 0], uids_valid)],
                dtype=torch.long,
            ),
            ripple_sets,
        )
        self._dataset_test = RippleDataset(
            torch.as_tensor(
                ratings_test[np.isin(ratings_test[:, 0], uids_valid)],
                dtype=torch.long,
            ),
            ripple_sets,
        )

    def reset_parameters(self):
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

    def _split_ripple_sets(
        self, ripple_sets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs_head = []
        embs_relation = []
        embs_tail = []
        # `ripple_set`: A `torch.Tensor` object of shape
        # `[batch_size, num_hops, 3, num_neighbors]`
        num_hops = ripple_sets.size(1)
        for hop in range(num_hops):
            embs_head.append(self.embeddings_entity[ripple_sets[:, hop, 0, :]])
            embs_relation.append(
                self.embeddings_relation[ripple_sets[:, hop, 1, :]]
            )
            embs_tail.append(self.embeddings_entity[ripple_sets[:, hop, 2, :]])
        return embs_head, embs_relation, embs_tail

    def forward(
        self,
        inputs_item: torch.Tensor,
        inputs_head: Sequence[torch.Tensor],
        inputs_relation: Sequence[torch.Tensor],
        inputs_tail: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Computes logits of input data.

        Args:
            inputs_item: A `torch.Tensor` object of shape `[batch_size, dim]`
            inputs_head: A list of `torch.Tensor` objects of shape
                `[batch_size, num_memory, dim]`
            inputs_relation: A list of `torch.Tensor` objects of shape
                `[batch_size, num_memory, dim, dim]`
            inputs_tail: A list of `torch.Tensor` objects of shape
                `[batch_size, num_memory, dim]`
        """
        query = inputs_item
        num_hops = len(inputs_head)
        values = []
        for hop in range(num_hops):
            r_h = (
                torch.matmul(
                    inputs_relation[hop],
                    inputs_head[hop].unsqueeze(3).contiguous(),
                )
                .squeeze(3)
                .contiguous()
            )
            logits = torch.sum(
                r_h * query.unsqueeze(1).contiguous(), dim=2, keepdim=True
            )
            probs = F.softmax(logits, dim=1)
            values.append(torch.sum(probs * inputs_tail[hop], dim=1))
            query = self.transform(query + values[-1])
        embeddings_user = sum(values)
        logits = torch.sum(embeddings_user * query, dim=1)
        return logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, int]]:
        embs_head, embs_relation, embs_tail = self._split_ripple_sets(
            batch["ripple_sets"]
        )
        logits = self.forward(
            self.embeddings_entity[batch["iids"]],
            embs_head,
            embs_relation,
            embs_tail,
        )
        loss_cls = F.binary_cross_entropy_with_logits(
            logits, target=batch["labels"].float()
        )
        wt_cls = logits.numel()
        loss = loss_cls
        outputs = {
            "loss_cls": loss_cls.detach(),
            "wt_cls": wt_cls,
        }
        if self._weight_kg > 0:
            loss_kg = 0.0
            wt_kg = 0
            num_hops = len(embs_head)
            for hop in range(num_hops):
                r_t = (
                    torch.matmul(
                        embs_relation[hop],
                        embs_tail[hop].unsqueeze(3).contiguous(),
                    )
                    .squeeze(3)
                    .contiguous()
                )
                logits = torch.sum(embs_head[hop] * r_t, dim=2)
                loss_kg += torch.sigmoid(logits).sum()
                wt_kg += logits.numel()
            loss_kg = -loss_kg / float(wt_kg)
            loss += self._weight_kg * loss_kg
            outputs["loss_kg"] = loss_kg.detach()
            outputs["wt_kg"] = wt_kg
        outputs["loss"] = loss
        return outputs

    def training_epoch_end(
        self, outputs: Sequence[Dict[str, Union[torch.Tensor, int]]]
    ):
        loss = 0.0
        loss_cls = 0.0
        wt_cls = 0.0
        loss_kg = 0.0
        wt_kg = 0.0
        for out in outputs:
            loss += out["loss"].item() * out["wt_cls"]
            loss_cls += out["loss_cls"].item() * out["wt_cls"]
            wt_cls += out["wt_cls"]
            if self._weight_kg > 0.0:
                loss_kg += out["loss_kg"].item() * out["wt_kg"]
                wt_kg += out.get("wt_kg", 0)
        wt_cls = float(wt_cls)
        wt_kg = float(wt_kg)
        self.log("loss", loss / wt_cls)
        self.log("loss_cls", loss_cls / wt_cls)
        self.log("wt_cls", wt_cls)
        if self._weight_kg > 0.0:
            self.log("loss_kg", loss_kg / wt_kg)
            self.log("wt_kg", wt_kg)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        embs_head, embs_relation, embs_tail = self._split_ripple_sets(
            batch["ripple_sets"]
        )
        logits = self.forward(
            self.embeddings_entity[batch["iids"]],
            embs_head,
            embs_relation,
            embs_tail,
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
        preds = (probs >= 0.5).astype(np.int32)
        labels = np.hstack(labels)
        return {
            "num_samples": probs.size,
            "auc": roc_auc_score(y_true=labels, y_score=probs),
            "f1": f1_score(y_true=labels, y_pred=preds),
        }


class _Worker(mp.Process):
    def __init__(
        self,
        queue_task: mp.JoinableQueue,
        queue_res: mp.JoinableQueue,
        max_hop: int,
        num_neighbors: int,
        adj_list_kg: Dict[int, Tuple[int, int]],
    ):
        super().__init__()
        self._queue_task = queue_task
        self._queue_res = queue_res
        self._max_hop = max_hop
        self._num_neighbors = num_neighbors
        self._adj_list_kg = adj_list_kg

    def run(self):
        while True:
            task = self._queue_task.get()
            if task is None:
                self._queue_task.task_done()
                break
            uid, history = task
            ripple_set = []
            for hop in range(self._max_hop):
                nbr_h, nbr_r, nbr_t = [], [], []
                if hop == 0:
                    eids_h = history
                else:
                    eids_h = ripple_set[-1][2]
                for eid_h in eids_h:
                    for (rid, eid_t) in self._adj_list_kg[eid_h]:
                        nbr_h.append(eid_h)
                        nbr_r.append(rid)
                        nbr_t.append(eid_t)
                if len(nbr_h) == 0:
                    ripple_set.append(ripple_set[-1])
                else:
                    pop = len(nbr_h)
                    indices = np.random.choice(
                        pop,
                        size=self._num_neighbors,
                        replace=pop < self._num_neighbors,
                    )
                    nbr_h = [nbr_h[i] for i in indices]
                    nbr_r = [nbr_r[i] for i in indices]
                    nbr_t = [nbr_t[i] for i in indices]
                    ripple_set.append((nbr_h, nbr_r, nbr_t))
            self._queue_res.put((uid, ripple_set))
            self._queue_task.task_done()
