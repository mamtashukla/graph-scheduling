/*
 * mlsys.h — shared types, constants, and function declarations
 *
 * All .c files include this header. Structs are inspired by:
 * https://github.com/yarongmu-google/MLSys/blob/main/mlsys.h
 */

#ifndef MLSYS_H
#define MLSYS_H

#define MAX_TENSORS   512
#define MAX_OPS       512
#define MAX_SUBGRAPHS 1024
#define MAX_FANIN     16
#define MAX_FANOUT    16
#define MAX_SG_OPS    256

#define OP_POINTWISE  0
#define OP_MATMUL     1

typedef struct {
    long long width;
    long long height;
} Tensor;

typedef struct {
    int       op_type;
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
    long long  *traversal_order;
    int         traversal_len;
    double      latency;
} Subgraph;

typedef struct {
    Subgraph subgraphs[MAX_SUBGRAPHS];
    int      num_subgraphs;
} Solution;

typedef struct {
    int producer;
    int consumers[MAX_OPS];
    int num_consumers;
    int is_graph_output;
} TensorInfo;

Problem  parse_problem(const char *filename);
void     build_tensor_info(const Problem *p, TensorInfo *info);
void     print_problem_summary(const Problem *p, const TensorInfo *info);
void     write_solution(const Solution *sol, const char *filename);
long long ceil_div(long long a, long long b);
double   evaluate_subgraph(const Problem *p, const TensorInfo *info,
                            const Subgraph *sg,
                            const int *retained, int num_retained);

void     topo_sort(const Problem *p, const TensorInfo *info,
                   int *topo_order, int *num_ordered);
Solution schedule_baseline(const Problem *p, const TensorInfo *info);

int       has_direct_edge(const int *ops_a, int na,
                          const int *ops_b, int nb,
                          const Problem *p, const TensorInfo *info);
int       intermediates_are_safe(const int *ops_a, int na,
                                 const int *ops_b, int nb,
                                 const Problem *p, const TensorInfo *info);
long long working_set(const int *ops, int nops,
                      const Problem *p, const TensorInfo *info,
                      Granularity gran);
Granularity pick_gran(const int *ops, int nops, const Problem *p,
                      const TensorInfo *info);
Solution    schedule_fusion(const Problem *p, const TensorInfo *info);

#endif /* MLSYS_H */
