import pdb
from collections import defaultdict
import networkx as nx
import pandas as pd
import heapq
import time

def build_equivalence_dict(r, lhs, rhs):
    eq_dict = defaultdict(lambda: defaultdict(list))
    for idx, t in enumerate(r):
        lhs_val = tuple(t[k] for k in lhs)
        rhs_val = tuple(t[k] for k in rhs)
        eq_dict[lhs_val][rhs_val].append(idx)
    return eq_dict

def maximal_afd_tuples(r, afd_list,n):
    G = nx.Graph()
    for afd in afd_list:
        lhs, rhs = afd["lhs"], afd["rhs"]
        eq_dict = build_equivalence_dict(r, lhs, rhs)
        for rhs_groups in eq_dict.values():
            group_lists = list(rhs_groups.values())
            if len(group_lists) <= 1:
                continue
            for i in range(len(group_lists)):
                for j in range(i + 1, len(group_lists)):
                    G.add_edges_from((a, b) for a in group_lists[i] for b in group_lists[j])

    V = set(range(n))
    T_prime = set()
    conflict_nodes = set(G.nodes)
    degree_map = dict(G.degree)
    heap = [(deg, node) for node, deg in degree_map.items()]
    heapq.heapify(heap)
    deleted = set()
    while heap:
        deg, u = heapq.heappop(heap)
        if u in deleted:
            continue
        T_prime.add(u)
        deleted.add(u)
        deleted.update(G.neighbors(u))
    T_prime.update(V - conflict_nodes)
    conflict_nodes=V - T_prime
    return conflict_nodes

def PartialRD(r, r_all, afd_list, relation_schemas,n):
    conflict_nodes = maximal_afd_tuples(r, afd_list,n)
    result = {}
    r0_df = r_all[r_all.index.isin(conflict_nodes)] # 包含 index_list 的子集
    r_df = r_all[~r_all.index.isin(conflict_nodes)]
    for idx, schema in enumerate(relation_schemas):
        sub_df = r_df[schema].drop_duplicates()
        result[f"r{idx + 1}"] = sub_df.to_dict(orient='records')
    result["r0"] = r0_df.to_dict(orient="records")
    return result