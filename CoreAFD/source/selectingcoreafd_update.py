import pdb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import product
import heapq
import time
from itertools import chain
from collections import defaultdict
import pandas as pd
from collections import defaultdict

def build_equivalence_dict_vectorized(df, lhs, rhs):
    early_stop_threshold_ratio = 0.05
    rhs_col = rhs[0]
    grouped = df.groupby(lhs + [rhs_col]).size().reset_index(name='count')
    lhs_rhs_counts = grouped.groupby(lhs).size().reset_index(name='rhs_count')
    valid_lhs = lhs_rhs_counts[lhs_rhs_counts['rhs_count'] > 1]
    grouped = grouped.merge(valid_lhs[lhs], on=lhs, how='inner')
    total_count = grouped['count'].sum()
    if total_count>early_stop_threshold_ratio * len(df):
        return 0, set()
    grouped['max_count'] = grouped.groupby(lhs)['count'].transform('max')
    grouped['is_max'] = grouped['count'] == grouped['max_count']
    grouped = grouped.reset_index(drop=False).rename(columns={'index': 'row_id'})
    first_max_rows = grouped[grouped['is_max']].groupby(lhs)['row_id'].first().reset_index()
    first_max_rows['keep'] = True
    grouped = grouped.merge(first_max_rows, on=lhs + ['row_id'], how='left')
    grouped['keep'] = grouped['keep'].fillna(False).astype(bool)
    conflict_rhs_rows = grouped[~grouped['keep']]
    df['orig_index'] = df.index
    merge_cols = lhs + [rhs_col]
    conflict_matches = df.merge(conflict_rhs_rows[merge_cols], on=merge_cols, how='inner')
    t_set = set(conflict_matches['orig_index'])
    return 1, t_set

def build_equivalence_dict(r, lhs, rhs):
    eq_dict = defaultdict(lambda: defaultdict(list))
    t_set = set()
    node_count = set()
    flag=1
    for idx, t in enumerate(r):
        lhs_val = tuple(t[k] for k in lhs)
        rhs_val = tuple(t[k] for k in rhs)
        eq_dict[lhs_val][rhs_val].append(idx)
        if len(eq_dict[lhs_val]) > 1:
            for rhs_key in eq_dict[lhs_val]:
                node_count.update(eq_dict[lhs_val][rhs_key])
        if idx > 20000 and len(node_count) > 0.3 * idx:
            flag=0
            eq_dict = defaultdict(lambda: defaultdict(list))
            break
        if len(node_count) > 0.05 * len(r):
            flag=0
            eq_dict = defaultdict(lambda: defaultdict(list))
            break
    if flag==1:
        for rhs_groups in eq_dict.values():
            if len(rhs_groups) <= 1:
                continue
            max_rhs_val = max(rhs_groups.items(), key=lambda x: len(x[1]))[0]
            for rhs_val, idx_list in rhs_groups.items():
                if rhs_val != max_rhs_val:
                    t_set.update(idx_list)
    return flag, t_set
def estimate_merged_cost(df, afd_set, index_list):
    """Estimates total cost of a merged FD set on a subset of tuples."""
    merged = defaultdict(set)
    for lhs, rhs in afd_set:
        merged[lhs].add(rhs)
    dg = 0
    mf=len(index_list)
    for lhs, rhs_set in merged.items():
        dX = df.loc[index_list, list(lhs)].drop_duplicates().shape[0]
        dg+=(mf-dX)*len(rhs_set)-dX*len(lhs)
    return dg

def computedg(df, afd_set, T_prime):
    """Computes total size after applying FD set and removing conflicting tuples."""
    dg = estimate_merged_cost(df, afd_set, list(T_prime))
    return dg

def get_t_set_for_fd_set(fd_set, afd_sets, total_count):
    conflict_union = set()
    for lhs, rhs in fd_set:
        key = (tuple(lhs), rhs)
        if key in afd_sets:
            conflict_union.update(afd_sets[key])
    all_indices = set(range(total_count))
    return all_indices - conflict_union

def cache_and_index(input_fd_sets, df, cols):
    """Main loop to evaluate each FD set and return the one with maximum gain."""
    r = df.to_dict(orient="records")
    n = len(r)
    best_afd = []
    best_dg = 0
    original_fds = set(fd for fd_set in input_fd_sets for fd in fd_set)
    laji_fds = []

    fd_t_sets = {}
    t1= time.time()
    for lhs, rhs in original_fds:
        flag, t_set = build_equivalence_dict_vectorized(df, list(lhs), [rhs])
        if flag==0:
            laji_fds.append((lhs, rhs))
        else:
            fd_t_sets[(lhs, rhs)] = t_set
    print(time.time() - t1)
    print('construct over')

    for fd_set in input_fd_sets:
        t2= time.time()
        effective_fd_set = [fd for fd in fd_set if fd not in laji_fds]
        if not effective_fd_set:
            print("All FDs skipped as laji.")
            continue
        T_prime = get_t_set_for_fd_set(effective_fd_set, fd_t_sets, n)
        dg = computedg(df, effective_fd_set, T_prime)
        # print(dg, effective_fd_set)
        if dg > best_dg:
            best_dg = dg
            best_afd = effective_fd_set
        print(f"AFD Set i: dg = {dg}, time = {round(time.time() - t2, 2)}s")
    return best_afd, best_dg


if __name__ == "__main__":
    df = pd.read_csv('../../dataset/Indusdata_running_example.csv')
    input = [((('device',), 'pctr'), (('docid',), 'subcategory'), (('duration',), 'click'),
              (('subcategory',), 'category'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('subcategory',), 'category'),
              (('timestamp',), 'duration'), (('timestamp',), 'pctr'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('age',), 'pctr'), (('docid',), 'subcategory'), (('duration',), 'click'), (('subcategory',), 'category'),
              (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'), (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('duration',), 'pctr'),
              (('subcategory',), 'category'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             (
             (('age',), 'pctr'), (('docid',), 'subcategory'), (('subcategory',), 'category'), (('timestamp',), 'click'),
             (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'), (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('subcategory',), 'category'),
              (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'), (('uin',), 'pctr'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('sex',), 'pctr'), (('subcategory',), 'category'),
              (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'), (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('subcategory',), 'category'),
              (('timestamp',), 'duration'), (('timestamp_laqu',), 'pctr'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('click',), 'pctr'), (('docid',), 'subcategory'), (('duration',), 'click'),
              (('subcategory',), 'category'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('category',), 'pctr'), (('docid',), 'subcategory'), (('subcategory',), 'category'),
              (('timestamp',), 'click'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('subcategory',), 'category'), (('subcategory',), 'pctr'),
              (('timestamp',), 'click'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'pctr'), (('subcategory',), 'category'),
              (('timestamp',), 'click'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             (
             (('docid',), 'subcategory'), (('sex',), 'pctr'), (('subcategory',), 'category'), (('timestamp',), 'click'),
             (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'), (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('subcategory',), 'category'), (('timestamp',), 'click'),
              (('timestamp',), 'duration'), (('timestamp_laqu',), 'pctr'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('click',), 'pctr'), (('docid',), 'subcategory'), (('subcategory',), 'category'),
              (('timestamp',), 'click'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('subcategory',), 'category'), (('timestamp',), 'click'),
              (('timestamp',), 'duration'), (('timestamp',), 'pctr'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('category',), 'pctr'), (('docid',), 'subcategory'), (('duration',), 'click'),
              (('subcategory',), 'category'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('docid',), 'subcategory'), (('duration',), 'click'), (('subcategory',), 'category'),
              (('subcategory',), 'pctr'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex')),
             ((('device',), 'pctr'), (('docid',), 'subcategory'), (('subcategory',), 'category'),
              (('timestamp',), 'click'), (('timestamp',), 'duration'), (('uin',), 'age'), (('uin',), 'device'),
              (('uin',), 'sex'))
             ]
    t1= time.time()
    best_afd, best_dg=cache_and_index(input, df, df.columns)
    print(time.time() - t1)
    print(best_afd, best_dg)