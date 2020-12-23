CPP=g++
CPPFLAGS=-std=c++0x \
				 -Wall -g \

PROG=main
SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)

all: $(PROG)

main: $(OBJS)
		$(CPP) -o $@ $^

clean:
		rm -f $(PROG) $(OBJS)

main.o: $(wildcard data/*.h) cnn.h config.h
cnn.o: cnn.h config.h
config.o: config.h

train:
		py/mnist.py

train_and_run: main train
		./main

.PHONY: all clean train train_and_run
