import numpy as np
import igraph as ig

def topo_features_igraph(pruned_indices, graph):
    """NetworkX版と同値（検証済み）・30〜66倍速い topological features."""
    num = len(pruned_indices)
    default = {'sum_edge_weights':0.0,'mean_edge_weights':0.0,'std_edge_weights':0.0,'n_edges_pruned':0,
               'graph_density':0.0,'n_connected_components':float(num),'mean_degree_centrality':0.0,
               'max_degree_centrality':0.0,'mean_betweenness_centrality':0.0,'max_betweenness_centrality':0.0,
               'avg_shortest_path_length':-1.0,'input_to_logit_shortest_path':-1.0}
    if num < 2: return default
    f = {}
    pruned = graph.adjacency_matrix[np.ix_(pruned_indices, pruned_indices)]
    A = pruned.numpy() if hasattr(pruned, 'numpy') else np.asarray(pruned)
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    edges = list(zip(rows.tolist(), cols.tolist())) # This line was missing
    weights = A[rows, cols].astype(float).tolist()
    # 決定ログ 2026-07-17: 媒介中心性は絶対値重みで定義(Path B)。
    # 負重み(~48%)による負閉路で重み付き最短経路が未定義となるため。
    processed_weights = [abs(w) for w in weights]
    g = ig.Graph(n=n, edges=edges, directed=True)
    if processed_weights: g.es['weight'] = processed_weights
    WA = 'weight' if g.ecount() > 0 else None
    ew = np.array(weights)
    if ew.size > 0:
        f['sum_edge_weights']=float(np.sum(ew)); f['mean_edge_weights']=float(np.mean(ew)); f['std_edge_weights']=float(np.std(ew))
    else:
        f['sum_edge_weights']=0.0; f['mean_edge_weights']=0.0; f['std_edge_weights']=0.0
    f['n_edges_pruned'] = g.ecount()
    f['graph_density'] = g.ecount()/(n*(n-1)) if n > 1 else 0.0
    wcc = g.connected_components(mode='weak'); f['n_connected_components'] = len(wcc)
    deg = np.array(g.degree(mode='all', loops=True), dtype=float); dc = deg/(n-1) if n > 1 else deg*0.0
    f['mean_degree_centrality']=float(np.mean(dc)); f['max_degree_centrality']=float(np.max(dc)) if dc.size>0 else 0.0
    bc_raw = np.array(g.betweenness(weights=WA, directed=True), dtype=float)
    bc = bc_raw/((n-1)*(n-2)) if n > 2 else bc_raw
    f['mean_betweenness_centrality']=float(np.mean(bc)); f['max_betweenness_centrality']=float(np.max(bc)) if bc.size>0 else 0.0
    memb=np.array(wcc.membership); sizes=np.array(wcc.sizes()); ln=np.where(memb==int(np.argmax(sizes)))[0].tolist()
    if len(ln) > 1:
        sub = g.induced_subgraph(ln)
        if sub.is_connected(mode='strong'):
            D = np.array(sub.distances(weights=('weight' if sub.ecount()>0 else None)), dtype=float)
            m = sub.vcount(); mask = ~np.eye(m, dtype=bool); f['avg_shortest_path_length']=float(D[mask].mean())
        else: f['avg_shortest_path_length']=-1.0
    else: f['avg_shortest_path_length']=-1.0
    n_feat=len(graph.selected_features); n_err=graph.cfg.n_layers*graph.n_pos
    st=n_feat+n_err; et=st+graph.n_pos; sl=et
    g2l={gi:li for li,gi in enumerate(pruned_indices)}
    src=[g2l[i] for i in pruned_indices if st<=i<et]; tgt=[g2l[i] for i in pruned_indices if i>=sl]
    if src and tgt:
        D=np.array(g.distances(source=src, target=tgt, weights=WA), dtype=float); fin=D[np.isfinite(D)]
        f['input_to_logit_shortest_path']=float(fin.min()) if fin.size>0 else -1.0
    else: f['input_to_logit_shortest_path']=-1.0
    for k in default:
        if k not in f: f[k]=default[k]
    return f
