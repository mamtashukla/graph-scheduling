/*
 * mlsys — BitBake-inspired DAG Scheduler for MLSys 2026 Track A
 *
 * Philosophy: Just like BitBake decides how to order and group build tasks
 * to avoid re-fetching sstate caches, we decide how to order and group
 * tensor ops to minimize slow-memory traffic.
 *
 *   BitBake DEPENDS edges      == tensor data edges
 *   Task grouping (configure+compile) == op fusion (ephemeral intermediates)
 *   sstate cache hit           == tensors_to_retain (data stays warm)
 *   Build artifact too large for /tmp == working set > fast_memory_capacity
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cjson/cJSON.h>

#define MAX_TENSORS   512
#define MAX_OPS       512
#define MAX_SUBGRAPHS 1024
#define MAX_FANIN     16    /* max inputs to one op  */
#define MAX_FANOUT    16    /* max outputs from one op */
#define MAX_SG_OPS    256   /* max ops fused into one subgraph */

#define OP_POINTWISE  0
#define OP_MATMUL     1

/*
 * Structs for Tensor and Ops inspired by mlsys.h
 * https://github.com/yarongmu-google/MLSys/blob/main/mlsys.h
 */

typedef struct {
    long long width;
    long long height;
} Tensor;

typedef struct {
    int       op_type;               /* OP_POINTWISE or OP_MATMUL */
    long long base_cost;
    int       inputs[MAX_FANIN];
    int       num_inputs;
    int       outputs[MAX_FANOUT];
    int       num_outputs;
} Op;

typedef struct {
    long long w;
    long long h;
    long long k;
} Granularity;

typedef struct {
    Tensor      tensors[MAX_TENSORS];
    int         num_tensors;
    Op          ops[MAX_OPS];
    int         num_ops;
    long long   fast_memory_capacity;
    long long   slow_memory_bandwidth;
    Granularity native_granularity;
} Problem;

typedef struct {
    int         ops[MAX_SG_OPS];
    int         num_ops;
    Granularity gran;
    int         tensors_to_retain[MAX_TENSORS];
    int         num_retain;
    long long  *traversal_order;    /* NULL = default raster order */
    int         traversal_len;
    double      latency;
} Subgraph;

typedef struct {
    Subgraph subgraphs[MAX_SUBGRAPHS];
    int      num_subgraphs;
} Solution;

/*
 * cJSON for reading Input.json
 *
 */

static char *read_file_to_string(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *buf = malloc(size + 1);
    if (!buf) { fprintf(stderr, "ERROR: OOM reading file\n"); exit(1); }
    size_t nread = fread(buf, 1, size, f);
    buf[nread] = '\0';
    fclose(f);
    return buf;
}

static Problem parse_problem(const char *filename) {
    Problem p;
    memset(&p, 0, sizeof(p));

    char *buf = read_file_to_string(filename);
    cJSON *root = cJSON_Parse(buf);
    free(buf);

    if (!root) {
        const char *err = cJSON_GetErrorPtr();
        fprintf(stderr, "ERROR: JSON parse failed near: %s\n", err ? err : "?");
        exit(1);
    }

    cJSON *j_widths  = cJSON_GetObjectItem(root, "widths");
    cJSON *j_heights = cJSON_GetObjectItem(root, "heights");
    p.num_tensors = cJSON_GetArraySize(j_widths);
    for (int i = 0; i < p.num_tensors; i++) {
        p.tensors[i].width  = (long long)cJSON_GetArrayItem(j_widths,  i)->valuedouble;
        p.tensors[i].height = (long long)cJSON_GetArrayItem(j_heights, i)->valuedouble;
    }

    cJSON *j_inputs     = cJSON_GetObjectItem(root, "inputs");
    cJSON *j_outputs    = cJSON_GetObjectItem(root, "outputs");
    cJSON *j_base_costs = cJSON_GetObjectItem(root, "base_costs");
    cJSON *j_op_types   = cJSON_GetObjectItem(root, "op_types");
    p.num_ops = cJSON_GetArraySize(j_inputs);

    for (int i = 0; i < p.num_ops; i++) {
        cJSON *in_arr   = cJSON_GetArrayItem(j_inputs,  i);
        cJSON *out_arr  = cJSON_GetArrayItem(j_outputs, i);
        const char *tp  = cJSON_GetArrayItem(j_op_types, i)->valuestring;

        p.ops[i].base_cost = (long long)cJSON_GetArrayItem(j_base_costs, i)->valuedouble;
        p.ops[i].op_type   = (strcmp(tp, "MatMul") == 0) ? OP_MATMUL : OP_POINTWISE;

        p.ops[i].num_inputs = cJSON_GetArraySize(in_arr);
        for (int n = 0; n < p.ops[i].num_inputs; n++)
            p.ops[i].inputs[n] = (int)cJSON_GetArrayItem(in_arr, n)->valuedouble;

        p.ops[i].num_outputs = cJSON_GetArraySize(out_arr);
        for (int n = 0; n < p.ops[i].num_outputs; n++)
            p.ops[i].outputs[n] = (int)cJSON_GetArrayItem(out_arr, n)->valuedouble;
    }

    p.fast_memory_capacity  = (long long)cJSON_GetObjectItem(root, "fast_memory_capacity")->valuedouble;
    p.slow_memory_bandwidth = (long long)cJSON_GetObjectItem(root, "slow_memory_bandwidth")->valuedouble;

    cJSON *ng = cJSON_GetObjectItem(root, "native_granularity");
    p.native_granularity.w = (long long)cJSON_GetArrayItem(ng, 0)->valuedouble;
    p.native_granularity.h = (long long)cJSON_GetArrayItem(ng, 1)->valuedouble;
    p.native_granularity.k = p.native_granularity.w;

    cJSON_Delete(root);
    return p;
}

typedef struct {
    int producer;              
    int consumers[MAX_OPS];
    int num_consumers;
    int is_graph_output;
} TensorInfo;

static void build_tensor_info(const Problem *p, TensorInfo *info) {
    memset(info, 0, sizeof(TensorInfo) * MAX_TENSORS);
    for (int t = 0; t < p->num_tensors; t++)
        info[t].producer = -1;  
    for (int i = 0; i < p->num_ops; i++) {
      
        for (int n = 0; n < p->ops[i].num_outputs; n++) {
            int t = p->ops[i].outputs[n];
            info[t].producer = i;
        }
    
        for (int n = 0; n < p->ops[i].num_inputs; n++) {
            int t = p->ops[i].inputs[n];
            info[t].consumers[info[t].num_consumers++] = i;
        }
    }

    for (int t = 0; t < p->num_tensors; t++)
        info[t].is_graph_output = (info[t].num_consumers == 0) ? 1 : 0;
}

static void print_problem_summary(const Problem *p, const TensorInfo *info) {
    fprintf(stderr, "--------------------------------------\n");
    fprintf(stderr, " Problem summary\n");
    fprintf(stderr, "--------------------------------------\n");
    fprintf(stderr, " Tensors : %d\n", p->num_tensors);
    fprintf(stderr, " Ops     : %d\n", p->num_ops);
    fprintf(stderr, " Fast mem: %lld\n", p->fast_memory_capacity);
    fprintf(stderr, " Bandwidth: %lld\n", p->slow_memory_bandwidth);
    fprintf(stderr, " Native gran: [%lld, %lld]\n",
            p->native_granularity.w, p->native_granularity.h);
    fprintf(stderr, "\n Graph inputs (in slow DRAM at start):\n");
    for (int t = 0; t < p->num_tensors; t++)
        if (info[t].producer < 0)
            fprintf(stderr, "   Tensor[%d]  %lld×%lld\n",
                    t, p->tensors[t].width, p->tensors[t].height);
    fprintf(stderr, " Graph outputs (must end in slow DRAM):\n");
    for (int t = 0; t < p->num_tensors; t++)
        if (info[t].is_graph_output)
            fprintf(stderr, "   Tensor[%d]  %lld×%lld\n",
                    t, p->tensors[t].width, p->tensors[t].height);
    fprintf(stderr, " Ops:\n");
    for (int i = 0; i < p->num_ops; i++) {
        fprintf(stderr, "   Op[%d] %s cost=%lld  in=[",
                i, p->ops[i].op_type == OP_MATMUL ? "MatMul    " : "Pointwise ",
                p->ops[i].base_cost);
        for (int n = 0; n < p->ops[i].num_inputs; n++)
            fprintf(stderr, "%d%s", p->ops[i].inputs[n],
                    n+1 < p->ops[i].num_inputs ? "," : "");
        fprintf(stderr, "] out=[");
        for (int n = 0; n < p->ops[i].num_outputs; n++)
            fprintf(stderr, "%d%s", p->ops[i].outputs[n],
                    n+1 < p->ops[i].num_outputs ? "," : "");
        fprintf(stderr, "]\n");
    }

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: mlsys <problem.json> \n");
        fprintf(stderr, "  If solution.json is omitted, writes to stdout.\n");
        return 1;
    }

    const char *problem_file  = argv[1];

    Problem prob = parse_problem(problem_file);

    /*Build graph metadata */
    TensorInfo tinfo[MAX_TENSORS];
    build_tensor_info(&prob, tinfo);

    print_problem_summary(&prob, tinfo);

    return 0;
}
