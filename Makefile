CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -g
LDFLAGS = -lcjson -lm

TARGET  = mlsys
SRC     = main.c

.PHONY: all clean static test bench

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

static: $(SRC)
	$(CC) $(CFLAGS) -static -o $(TARGET) $^ -l:libcjson.a -lm

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
