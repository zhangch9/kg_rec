# -*- coding: utf-8 -*-
"""Entrypoint for command `create`."""


import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse.actions import _ActionSubCommands

from .logging_utils import add_options_logging, initialize_logging

logger = logging.getLogger(__name__)


def _add_options_common(parser: ArgumentParser):
    add_options_logging(parser)
    parser.add_argument("--config", action=ActionConfigFile)


def generate_parser(
    sub_commands: Optional[_ActionSubCommands] = None,
) -> ArgumentParser:
    parser = ArgumentParser(description="Create a dataset from raw data.")
    if sub_commands:
        sub_commands.add_subcommand("create", parser, help=parser.description)

    sub_commands = parser.add_subcommands(
        dest="dataset", title="Available Datasets", metavar="DATASET"
    )

    parser_lastfm = ArgumentParser(description="Create the Last.fm dataset.")
    parser_lastfm.add_function_arguments(
        create_lastfm, nested_key="args", as_group=True
    )
    _add_options_common(parser_lastfm)
    sub_commands.add_subcommand(
        "lastfm", parser_lastfm, help=parser_lastfm.description
    )

    parser_yelp = ArgumentParser(description="Create the Yelp dataset.")
    parser_yelp.add_function_arguments(
        create_yelp, nested_key="args", as_group=True
    )
    _add_options_common(parser_yelp)
    sub_commands.add_subcommand(
        "yelp", parser_yelp, help=parser_yelp.description
    )
    return parser


def create_lastfm(raw_dir: str, save_dir: str, seed: int = 0):
    """Creates the Last.fm dataset.

    Args:
        raw_dir: Directory containing the raw data files.
        save_dir: Directory to save the dataset files.
        seed: An integer to seed the random state.
    """
    np.random.seed(seed)
    raw_dir = Path(raw_dir).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve().joinpath(f"seed_{seed}")
    save_dir.mkdir(parents=True, exist_ok=False)

    # We map the original item indices and the corresponding entity indices
    # to {0, 1, ..., num_items - 1}.
    item_eid, entity_eid = {}, {}
    with open(raw_dir.joinpath("item_index2entity_id.txt"), "r") as f:
        for eid, line in enumerate(f):
            item, entity = map(int, line.strip().split("\t"))
            assert item not in item_eid and entity not in entity_eid
            item_eid[item] = eid
            entity_eid[entity] = eid

    # We map the original user indices to {0, 1, ..., num_users - 1},
    # and assign a binary label to each rating according to a threshold.
    threshold = 0
    user_uid = {}
    num_users = 0
    ratings_pos, ratings_neg = defaultdict(set), defaultdict(set)
    with open(raw_dir.joinpath("user_artists.dat"), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            user, item, rating = map(int, line.strip().split("\t"))
            if item not in item_eid:
                continue
            if user not in user_uid:
                user_uid[user] = num_users
                num_users += 1
            uid = user_uid[user]
            eid = item_eid[item]
            assert eid not in ratings_pos[uid] and eid not in ratings_neg[uid]
            if rating >= threshold:
                ratings_pos[uid].add(eid)
            else:
                ratings_neg[uid].add(eid)

    # We save positive ratings and sample a set of unseen items
    # as negative examples for each user.
    eid_set = set(range(len(item_eid)))
    num_ratings = 0
    with open(save_dir.joinpath("ratings.txt"), "w", encoding="utf-8") as f:
        for uid, eids_pos in ratings_pos.items():
            for eid in eids_pos:
                f.write(f"{uid}\t{eid}\t1\n")
                num_ratings += 1
            # we remove positive and negative items from the candidate items
            unseen_eids = eid_set - eids_pos
            if uid in ratings_neg:
                unseen_eids = unseen_eids - ratings_neg[uid]
            for eid in np.random.choice(
                list(unseen_eids), size=len(eids_pos), replace=False
            ):
                f.write(f"{uid}\t{eid}\t0\n")

    # We map the original entity indices to {0, 1, ..., num_entities - 1},
    # and the original relations to {0, 1, ..., num_relations - 1}.
    # Entities associated with items are processed before.
    num_eids = len(entity_eid)
    relation_rid = {}
    num_relations = 0
    num_triplets = 0
    with open(raw_dir.joinpath("kg.txt"), "r", encoding="utf-8") as fin, open(
        save_dir.joinpath("triplets_kg.txt"), "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            head, relation, tail = line.strip().split("\t")
            head, tail = int(head), int(tail)
            if head not in entity_eid:
                entity_eid[head] = num_eids
                num_eids += 1
            if tail not in entity_eid:
                entity_eid[tail] = num_eids
                num_eids += 1
            if relation not in relation_rid:
                relation_rid[relation] = num_relations
                num_relations += 1
            eid_h, eid_t = entity_eid[head], entity_eid[tail]
            rid = relation_rid[relation]
            fout.write(f"{eid_h}\t{rid}\t{eid_t}\n")
            num_triplets += 1

    # We create a subgraph induced on users in the interaction data
    # from the original social network.
    num_edges_uu = 0
    with open(
        raw_dir.joinpath("user_friends.dat"), "r", encoding="latin-1"
    ) as fin, open(
        save_dir.joinpath("edges_uu.txt"), "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin):
            if line_no == 0:
                continue
            user_i, user_j = map(int, line.strip().split("\t"))
            if user_i > user_j:
                continue
            if user_i in user_uid and user_j in user_uid:
                fout.write(f"{user_uid[user_i]}\t{user_uid[user_j]}\n")
                num_edges_uu += 1
    logger.info(
        "==========Dataset Statistics==========\n"
        f"num_users = {len(user_uid)}\nnum_items = {len(item_eid)}\n"
        f"num_ratings = {num_ratings}\nnum_entities = {num_eids}\n"
        f"num_relations = {num_relations}\nnum_triplets = {num_triplets}\n"
        f"num_edges_uu = {num_edges_uu}"
    )


def create_yelp(raw_dir: str, save_dir: str, k: int = 10):
    """Creates the Yelp dataset.

    Args:
        raw_dir: Directory containing the raw data files.
        save_dir: Directory to save the dataset files.
        k: Order of the core.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()
    # save_dir.mkdir(parents=True, exist_ok=False)

    # We create a bipartite graph based on Yelp reviews and extract
    # its k-core subgraph.
    user_uid, item_iid = {}, {}
    num_uids, num_iids = 0, 0
    uid_iid_stars = defaultdict(list)
    with open(raw_dir.joinpath("yelp_academic_dataset_review.json"), "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj["user_id"]
            item = obj["business_id"]
            if user not in user_uid:
                user_uid[user] = num_uids
                num_uids += 1
            if item not in item_iid:
                item_iid[item] = num_iids
                num_iids += 1
            uid, iid = user_uid[user], item_iid[item]
            uid_iid_stars[(uid, iid)].append(float(obj["stars"]))

    g = nx.Graph()
    for uid, iid in uid_iid_stars:
        g.add_edge(uid, iid + num_uids)
    k_core = nx.k_core(g, k)
    print(k_core.__class__)
    logger.info(f"{k}-core: #nodes = {k_core.number_of_nodes()}")


DATASET_FUNC = {"lastfm": create_lastfm, "yelp": create_yelp}


def main(args: Namespace):
    fn = DATASET_FUNC[args.dataset]
    args = getattr(args, args.dataset)
    initialize_logging(args.verbose)
    fn(**args.args)
