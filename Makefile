CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -g
LDFLAGS = -lcjson -lm
CJSON_A = ../cJSON/build-static/libcjson.a

TARGET  = mlsys
SRC     = main.c baseline.c fusion.c

.PHONY: all clean dynamic test bench

# Default: fully static binary — no runtime dependencies, safe to ship
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -static -o $@ $^ $(CJSON_A) -lm

# Dynamic build (requires libcjson installed on the runner)
dynamic: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $^ $(LDFLAGS)

test: $(TARGET)
	./$(TARGET) example_problem.json

bench: $(TARGET)
	@for f in benchmarks/*.json; do \
            echo ""; \
            echo "=== $$f ==="; \
            ./$(TARGET) $$f; \
        done


clean:
	rm -f $(TARGET)
