CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -g
LDFLAGS = -lcjson -lm

TARGET  = mlsys
SRC     = main.c

.PHONY: all clean static test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

static: $(SRC)
	$(CC) $(CFLAGS) -static -o $(TARGET) $^ -l:libcjson.a -lm

test: $(TARGET)
	./$(TARGET) example_problem.json

clean:
	rm -f $(TARGET)
