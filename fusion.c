/*
 * fusion.c — Fusion scheduler.
 * BitBake analogy: group do_configure + do_compile only when the intermediate
 * build artefacts are private to that recipe AND fit in /tmp scratch space.
 */

#include <string.h>

#include "mlsys.h"

int has_direct_edge(const int *ops_a, int na,
                    const int *ops_b, int nb,
                    const Problem *p, const TensorInfo *info) {
    int in_b[MAX_OPS] = {0};
    for (int i = 0; i < nb; i++) in_b[ops_b[i]] = 1;

    for (int i = 0; i < na; i++) {
        int op = ops_a[i];
        for (int n = 0; n < p->ops[op].num_outputs; n++) {
            int t = p->ops[op].outputs[n];
            for (int c = 0; c < info[t].num_consumers; c++)
                if (in_b[info[t].consumers[c]]) return 1;
        }
    }
    return 0;
}

int intermediates_are_safe(const int *ops_a, int na,
                           const int *ops_b, int nb,
                           const Problem *p, const TensorInfo *info) {
    int in_ab[MAX_OPS] = {0};
    for (int i = 0; i < na; i++) in_ab[ops_a[i]] = 1;
    for (int i = 0; i < nb; i++) in_ab[ops_b[i]] = 1;

    int in_b[MAX_OPS] = {0};
    for (int i = 0; i < nb; i++) in_b[ops_b[i]] = 1;

    for (int i = 0; i < na; i++) {
        int op = ops_a[i];
        for (int n = 0; n < p->ops[op].num_outputs; n++) {
            int t = p->ops[op].outputs[n];
            int flows_to_b = 0;
            for (int c = 0; c < info[t].num_consumers; c++)
                if (in_b[info[t].consumers[c]]) { flows_to_b = 1; break; }
            if (!flows_to_b) continue;
            for (int c = 0; c < info[t].num_consumers; c++)
                if (!in_ab[info[t].consumers[c]]) return 0;
        }
    }
    return 1;
}

long long working_set(const int *ops, int nops,
                      const Problem *p, const TensorInfo *info,
                      Granularity gran) {
    long long w = gran.w, h = gran.h, k = gran.k;
    long long ws = 0;

    int in_grp[MAX_OPS]     = {0};
    int counted[MAX_TENSORS] = {0};
    for (int i = 0; i < nops; i++) in_grp[ops[i]] = 1;

    for (int i = 0; i < nops; i++) {
        int op = ops[i];

        for (int n = 0; n < p->ops[op].num_inputs; n++) {
            int t = p->ops[op].inputs[n];
            if (counted[t]) continue;
            if (info[t].producer >= 0 && in_grp[info[t].producer]) continue;
            counted[t] = 1;
            ws += (p->ops[op].op_type == OP_MATMUL)
                  ? ((n == 0) ? h * k : k * w)
                  : (w * h);
        }

        for (int n = 0; n < p->ops[op].num_outputs; n++) {
            int t = p->ops[op].outputs[n];
            if (counted[t]) continue;
            int ext = info[t].is_graph_output;
            for (int c = 0; !ext && c < info[t].num_consumers; c++)
                ext = !in_grp[info[t].consumers[c]];
            if (!ext) continue;
            counted[t] = 1;
            ws += w * h;
        }
    }
    return ws;
}

Granularity pick_gran(const int *ops, int nops, const Problem *p) {
    Granularity g;
    g.w = p->native_granularity.w;
    g.h = p->native_granularity.h;
    g.k = 1;
    for (int i = 0; i < nops; i++)
        if (p->ops[ops[i]].op_type == OP_MATMUL) {
            g.k = p->tensors[p->ops[ops[i]].inputs[0]].width;
            break;
        }
    return g;
}

/*
 * Walk ops in topological order.  Maintain an open subgraph and attempt to
 * absorb each successive op.  Merge when all three guards pass; otherwise
 * emit the current subgraph and open a fresh one.
 */
Solution schedule_fusion(const Problem *p, const TensorInfo *info) {
    Solution sol;
    memset(&sol, 0, sizeof(sol));

    int topo_order[MAX_OPS];
    int num_ordered = 0;
    topo_sort(p, info, topo_order, &num_ordered);

    Subgraph *sg = &sol.subgraphs[0];
    sg->ops[0]          = topo_order[0];
    sg->num_ops         = 1;
    sg->gran            = pick_gran(sg->ops, 1, p);
    sg->num_retain      = 0;
    sg->traversal_order = NULL;
    sol.num_subgraphs   = 1;

    for (int i = 1; i < num_ordered; i++) {
        int next = topo_order[i];

        /* Build tentative merged op list */
        int merged[MAX_SG_OPS];
        int nm = sg->num_ops;
        for (int j = 0; j < nm; j++) merged[j] = sg->ops[j];
        merged[nm] = next;

        Granularity mg = pick_gran(merged, nm + 1, p);

        int can_merge =
            (nm < MAX_SG_OPS - 1)
         && has_direct_edge(sg->ops, nm, &next, 1, p, info)
         && intermediates_are_safe(sg->ops, nm, &next, 1, p, info)
         && working_set(merged, nm + 1, p, info, mg) <= p->fast_memory_capacity;

        if (can_merge) {
            sg->ops[sg->num_ops++] = next;
            sg->gran = mg;
        } else {
            sg->latency = evaluate_subgraph(p, info, sg, NULL, 0);
            sg = &sol.subgraphs[sol.num_subgraphs++];
            sg->ops[0]          = next;
            sg->num_ops         = 1;
            sg->gran            = pick_gran(sg->ops, 1, p);
            sg->num_retain      = 0;
            sg->traversal_order = NULL;
        }
    }

    sg->latency = evaluate_subgraph(p, info, sg, NULL, 0);
    return sol;
}
