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

def maximal_afd_tuples(r, afd_list):
    G = nx.Graph()
    n = len(r)
    start_time = time.time()

    # Step 1: Build conflict graph
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

    end_time = time.time()
    print(f"Time to construct conflict graph: {end_time - start_time:.4f} s")

    # Step 2: Select maximal AFD-consistent tuples
    start_time = time.time()
    V = set(range(n))
    T_prime = set()
    conflict_nodes = set(G.nodes)
    print(f"Number of Tuples: {n}")
    print(f"Conflict Graph Nodes: {len(conflict_nodes)}")
    print(f"Proportion: {len(conflict_nodes) / n:.2%}")

    # Greedy deletion (minimum-degree-first)
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

    # Add tuples not in conflict
    T_prime.update(V - conflict_nodes)
    r_prime = [r[i] for i in sorted(T_prime)]
    end_time = time.time()
    print(f"Time to identify maximal AFD tuples: {end_time - start_time:.4f} s")
    return r_prime

def PartialRD(r, afd_list, relation_schemas):
    r_prime = maximal_afd_tuples(r, afd_list)
    result = {}

    # Project into subrelations
    r_df = pd.DataFrame(r_prime)
    for idx, schema in enumerate(relation_schemas):
        sub_df = r_df[schema].drop_duplicates()
        result[f"r{idx + 1}"] = sub_df.to_dict(orient='records')

    # Extract residual tuples (r0)
    r_all = pd.DataFrame(r)
    r_prime_df = pd.DataFrame(r_prime)
    r0_df = pd.concat([r_all, r_prime_df]).drop_duplicates(keep=False)
    result["r0"] = r0_df.to_dict(orient="records")

    return result