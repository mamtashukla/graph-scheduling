# graph-scheduling 

A BitBake-inspired DAG scheduler for minimising slow-memory traffic in tensor
workloads.  The core analogy: BitBake groups build tasks to avoid re-fetching
sstate caches; this scheduler groups tensor ops into fused subgraphs to avoid
spilling intermediates to slow DRAM.

---

## Dependencies

| Library | Purpose | Package (Ubuntu/Debian) |
|---------|---------|------------------------|
| **[cJSON](https://github.com/davegamble/cjson)** | JSON parsing (problem input) and serialisation (solution output) | `libcjson-dev` |
| **libm** | `math.h` (`ceil`, `log2`) | bundled with gcc |

Install cJSON on Ubuntu/Debian:

```sh
sudo apt install libcjson-dev
```

On Fedora/RHEL:

```sh
sudo dnf install cjson-devel
```

---

## Building

### Dynamic build (requires cJSON installed at runtime)

```sh
make
```

Produces `./mlsys` linked against the system `libcjson.so`.  Users must have
`libcjson-dev` (or the equivalent runtime package) installed.

### Fully static build (no runtime dependencies)

```sh
make static
```

Produces `./mlsys` binary with cJSON compiled in.  This is
what you want for submitting to the contest — the grader machine may not have
cJSON installed.  Requires `libcjson.a` from the dev package:

```sh
sudo apt install libcjson-dev   # provides both .so and .a
make static
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
