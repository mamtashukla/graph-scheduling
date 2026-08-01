# graph-scheduling 

A BitBake-inspired DAG scheduler for minimising slow-memory traffic in tensor
workloads.  The core analogy: BitBake groups build tasks to avoid re-fetching
sstate caches. This scheduler groups tensor ops into fused subgraphs to avoid
spilling intermediates to slow DRAM.

---

## Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| **[cJSON](https://github.com/davegamble/cjson)** | JSON I/O | bundled under `../cJSON/` |
| **libm** | math | bundled with gcc |

For the default static build no system packages are needed — cJSON source
is already in this repo under `../cJSON/`.

---

## Building

`make` produces a **fully static binary** — cJSON, libc, and libm are all
compiled in.  No runtime dependencies; the binary can be copied to any
x86-64 Linux machine and run directly.

```sh
make
```

First build the static cJSON library from the bundled source (one-time):

```sh
cmake -S ../cJSON -B ../cJSON/build-static \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_CJSON_TEST=OFF
cmake --build ../cJSON/build-static
make
```

### Dynamic build (requires cJSON installed at runtime)

If you prefer to link against the system cJSON (faster compile, not portable):

```sh
sudo apt install libcjson-dev
make dynamic
```

---

## Running

```sh
# Write solution to stdout
./mlsys path/to/problem.json

# Write solution to a file
./mlsys path/to/problem.json path/to/solution.json
```

```sh
make test
```

### Run benchmarks

```sh
make bench
```

Benchmarks are expected under `benchmarks/*.json`.
