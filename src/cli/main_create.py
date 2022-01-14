# -*- coding: utf-8 -*-
"""Entrypoint for command `create`."""


import ast
import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

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
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            user, item, rating = map(int, line.strip().split("\t"))
            if item not in item_eid:
                continue
            if user not in user_uid:
                user_uid[user] = num_users
                num_users += 1
            uid = user_uid[user]
            iid = item_eid[item]
            assert iid not in ratings_pos[uid] and iid not in ratings_neg[uid]
            if rating >= threshold:
                ratings_pos[uid].add(iid)
            else:
                ratings_neg[uid].add(iid)

    # We save positive ratings and sample a set of unseen items
    # as negative examples for each user.
    iid_set = set(item_eid.values())
    num_ratings = 0
    with open(save_dir.joinpath("ratings.txt"), "w", encoding="utf-8") as f:
        for uid, iids_pos in ratings_pos.items():
            for iid in iids_pos:
                f.write(f"{uid}\t{iid}\t1\n")
                num_ratings += 1
            # we remove positive and negative items from the candidate items
            unseen_iids = iid_set - iids_pos
            if uid in ratings_neg:
                unseen_iids = unseen_iids - ratings_neg[uid]
            for iid in np.random.choice(
                sorted(unseen_iids), size=len(iids_pos), replace=False
            ):
                f.write(f"{uid}\t{iid}\t0\n")
                num_ratings += 1

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
        for idx, line in enumerate(fin):
            if idx == 0:
                continue
            user_i, user_j = map(int, line.strip().split("\t"))
            if user_i > user_j:
                continue
            if user_i in user_uid and user_j in user_uid:
                fout.write(f"{user_uid[user_i]}\t{user_uid[user_j]}\n")
                num_edges_uu += 1
    logger.info(
        "==========Dataset Statistics==========\n"
        f"num_users = {len(user_uid)}\n"
        f"num_items = {len(item_eid)}\n"
        f"num_ratings (incl. negative) = {num_ratings}\n"
        f"num_entities (incl. items) = {num_eids}\n"
        f"num_relations = {num_relations}\n"
        f"num_triplets = {num_triplets}\n"
        f"num_edges_uu (undirected) = {num_edges_uu}"
    )


def create_yelp(raw_dir: str, save_dir: str, k: int = 10, seed: int = 0):
    """Creates the Yelp dataset.

    Args:
        raw_dir: Directory containing the raw data files.
        save_dir: Directory to save the dataset files.
        k: Order of the core.
        seed: An integer to seed the random state.
    """
    np.random.seed(seed)
    raw_dir = Path(raw_dir).expanduser().resolve()
    save_dir = (
        Path(save_dir).expanduser().resolve().joinpath(f"core_{k}_seed_{seed}")
    )
    save_dir.mkdir(parents=True, exist_ok=False)

    users_raw, entities_raw = [], []
    user_uid_raw, entity_eid_raw = {}, {}
    num_users, num_entities = 0, 0
    # `(uid, iid)` -> a list of ratings
    uid_eid_stars = defaultdict(list)
    with open(raw_dir.joinpath("yelp_academic_dataset_review.json"), "r") as f:
        for line in f:
            json_obj = json.loads(line)
            user = json_obj["user_id"]
            item = json_obj["business_id"]
            assert len(user) == 22 and len(item) == 22
            if user not in user_uid_raw:
                users_raw.append(user)
                user_uid_raw[user] = num_users
                num_users += 1
            if item not in entity_eid_raw:
                entities_raw.append(item)
                entity_eid_raw[item] = num_entities
                num_entities += 1
            uid, iid = user_uid_raw[user], entity_eid_raw[item]
            uid_eid_stars[(uid, iid)].append(float(json_obj["stars"]))

    # We create a bipartite graph based on Yelp reviews and extract
    # its k-core subgraph.
    g = nx.Graph()
    for uid, iid in uid_eid_stars:
        g.add_edge(uid, iid + num_users)
    k_core = nx.k_core(g, k)
    logger.info(f"{k}-core: #nodes = {k_core.number_of_nodes()}")

    users, entities = [], []
    uid_set_raw, iid_set_raw = set(), set()
    for node in k_core:
        if node < num_users:
            uid_set_raw.add(node)
            users.append(users_raw[node])
        else:
            iid_set_raw.add(node - num_users)
            entities.append(entities_raw[node - num_users])
    num_users = len(users)
    num_entities = len(entities)
    user_uid = {user: idx for idx, user in enumerate(users)}
    entity_eid = {item: idx for idx, item in enumerate(entities)}

    # `uid` -> a list of `iid`.
    ratings_pos = defaultdict(set)
    for uid_raw, iid_raw in uid_eid_stars:
        if uid_raw not in uid_set_raw or iid_raw not in iid_set_raw:
            continue
        ratings_pos[user_uid[users_raw[uid_raw]]].add(
            entity_eid[entities_raw[iid_raw]]
        )

    iid_set = set(entity_eid.values())
    num_ratings = 0
    with open(save_dir.joinpath("ratings.txt"), "w", encoding="utf-8") as f:
        for uid, iids_pos in ratings_pos.items():
            for iid in iids_pos:
                f.write(f"{uid}\t{iid}\t1\n")
                num_ratings += 1
            # we remove positive items from the candidate items
            unseen_iids = iid_set - iids_pos
            for iid in np.random.choice(
                sorted(unseen_iids), size=len(iids_pos), replace=False
            ):
                f.write(f"{uid}\t{iid}\t0\n")
                num_ratings += 1

    def flatten_json(json_obj: Dict[str, Any], parent_key="") -> Dict[str, Any]:
        json_flat = {}
        for key, value in json_obj.items():
            field = f"{parent_key}.{key}" if parent_key != "" else key
            if value is None:
                continue
            if isinstance(value, dict):
                json_flat_nested = flatten_json(value, parent_key=field)
                for k, v in json_flat_nested.items():
                    assert k not in json_flat
                    json_flat[k] = v
            else:
                assert field not in json_flat
                if isinstance(value, str):
                    value = value.strip()
                    try:
                        value = ast.literal_eval(value)
                        if isinstance(value, dict):
                            json_flat_nested = flatten_json(
                                value, parent_key=field
                            )
                            for k, v in json_flat_nested.items():
                                assert k not in json_flat
                                json_flat[k] = v
                        elif isinstance(value, (int, float, str)):
                            json_flat[field] = value
                        elif value is not None:
                            raise TypeError(
                                f"'{value.__class__}' object is not supported."
                            )
                    except (SyntaxError, ValueError):
                        json_flat[field] = value
                else:
                    json_flat[field] = value
        return json_flat

    # builds the knowledge graph
    relations = []
    relation_rid = {}
    num_relations = 0
    num_triplets = 0
    with open(
        raw_dir.joinpath("yelp_academic_dataset_business.json"), "r"
    ) as fin, open(
        save_dir.joinpath("triplets_kg.txt"), "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            json_obj = json.loads(line)
            item = json_obj.pop("business_id")
            if item not in entity_eid:
                continue
            json_obj = flatten_json(json_obj, parent_key="")
            # `relation`` -> a set of tail `entities``
            relation_tails = OrderedDict()
            for key, value in json_obj.items():
                relation = key.strip().lower()
                if not relation.startswith("attributes.") and relation not in {
                    "city",
                    "state",
                    "stars",
                    "categories",
                }:
                    continue
                if relation not in relation_tails:
                    relation_tails[relation] = set()
                if relation == "categories":
                    for cat in map(
                        lambda s: s.strip().lower(), value.split(",")
                    ):
                        relation_tails[relation].add(f"{relation}.{cat}")
                elif isinstance(value, (int, float)):
                    relation_tails[relation].add(f"{relation}.{value}")
                elif isinstance(value, str):
                    relation_tails[relation].add(
                        f"{relation}.{value.strip().lower()}"
                    )
                else:
                    raise TypeError(
                        f"'{value.__class__}' object is not supported."
                    )

            for relation, tails in relation_tails.items():
                if relation not in relation_rid:
                    relation_rid[relation] = num_relations
                    relations.append(relation)
                    num_relations += 1
                for tail in sorted(tails):
                    if tail not in entity_eid:
                        entity_eid[tail] = num_entities
                        entities.append(tail)
                        num_entities += 1
                    fout.write(
                        f"{entity_eid[item]}\t{relation_rid[relation]}\t"
                        f"{entity_eid[tail]}\n"
                    )
                    num_triplets += 1

    with open(save_dir.joinpath("entity_eid.txt"), "w", encoding="utf-8") as f:
        f.write("entity\teid\n")
        for eid, entity in enumerate(entities):
            f.write(f"{entity}\t{eid}\n")
    with open(
        save_dir.joinpath("relation_rid.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("relation\trid\n")
        for rid, relation in enumerate(relations):
            f.write(f"{relation}\t{rid}\n")

    # builds the social network
    num_edges_uu = 0
    with open(
        raw_dir.joinpath("yelp_academic_dataset_user.json"),
        "r",
        encoding="utf-8",
    ) as fin, open(
        save_dir.joinpath("edges_uu.txt"), "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            obj = json.loads(line)
            user = obj["user_id"]
            if user not in user_uid:
                continue
            uid_u = user_uid[user]
            friends = obj.get("friends", None)
            if friends is not None:
                for user_f in map(lambda s: s.strip(), friends.split(",")):
                    if user_f in user_uid:
                        uid_v = user_uid[user_f]
                        assert uid_u != uid_v
                        if uid_u < uid_v:
                            fout.write(f"{uid_u}\t{uid_v}\n")
                            num_edges_uu += 1
    with open(save_dir.joinpath("user_uid.txt"), "w", encoding="utf-8") as f:
        f.write("user\tuid\n")
        for uid, user in enumerate(users):
            f.write(f"{user}\t{uid}\n")

    logger.info(
        "==========Dataset Statistics==========\n"
        f"num_users = {num_users}\n"
        f"num_items = {len(iid_set)}\n"
        f"num_ratings (incl. negative) = {num_ratings}\n"
        f"num_entities (incl. items) = {num_entities}\n"
        f"num_relations = {num_relations}\n"
        f"num_triplets = {num_triplets}\n"
        f"num_edges_uu (undirected) = {num_edges_uu}"
    )


DATASET_FUNC = {"lastfm": create_lastfm, "yelp": create_yelp}


def main(args: Namespace):
    fn = DATASET_FUNC[args.dataset]
    args = getattr(args, args.dataset)
    initialize_logging(args.verbose)
    fn(**args.args)
