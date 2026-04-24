/*
 * baseline.c — Topological sort (Kahn's algorithm) + one-op-per-subgraph
 *              baseline scheduler.
 *
 * BitBake analogy: the simplest possible build plan — each recipe runs alone,
 * no task grouping, no sstate reuse.  Every intermediate is spilled to disk.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlsys.h"

void topo_sort(const Problem *p, const TensorInfo *info,
               int *topo_order, int *num_ordered) {
    int indegree[MAX_OPS] = {0};
    int queue[MAX_OPS];
    int head = 0, tail = 0;

    for (int i = 0; i < p->num_ops; i++)
        for (int n = 0; n < p->ops[i].num_inputs; n++) {
            int t = p->ops[i].inputs[n];
            if (info[t].producer >= 0)
                indegree[i]++;
        }

    for (int i = 0; i < p->num_ops; i++)
        if (indegree[i] == 0)
            queue[tail++] = i;

    *num_ordered = 0;
    while (head < tail) {
        int op = queue[head++];
        topo_order[(*num_ordered)++] = op;
        for (int n = 0; n < p->ops[op].num_outputs; n++) {
            int t = p->ops[op].outputs[n];
            for (int c = 0; c < info[t].num_consumers; c++) {
                int consumer = info[t].consumers[c];
                if (--indegree[consumer] == 0)
                    queue[tail++] = consumer;
            }
        }
    }

    if (*num_ordered != p->num_ops) {
        fprintf(stderr, "ERROR: Topological sort failed — cycle detected? "
                "Ordered %d of %d ops.\n", *num_ordered, p->num_ops);
        exit(1);
    }
}

Solution schedule_baseline(const Problem *p, const TensorInfo *info) {
    Solution sol;
    memset(&sol, 0, sizeof(sol));

    int topo_order[MAX_OPS];
    int num_ordered = 0;
    topo_sort(p, info, topo_order, &num_ordered);

    for (int i = 0; i < num_ordered; i++) {
        int op = topo_order[i];
        Subgraph *sg = &sol.subgraphs[sol.num_subgraphs];

        sg->ops[0]  = op;
        sg->num_ops = 1;

        sg->gran.w = p->native_granularity.w;
        sg->gran.h = p->native_granularity.h;
        sg->gran.k = (p->ops[op].op_type == OP_MATMUL)
                     ? p->tensors[p->ops[op].inputs[0]].width  /* full K */
                     : 1;

        sg->num_retain      = 0;    /* evict everything — no sstate */
        sg->traversal_order = NULL; /* raster order */

        sg->latency = evaluate_subgraph(p, info, sg, NULL, 0);
        sol.num_subgraphs++;
    }

    return sol;
}
