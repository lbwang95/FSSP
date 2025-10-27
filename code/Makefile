CC = gcc
CXX = g++
LIBS = -lrt
CFLAGS = --std=c++11 -W -Wall -Wno-sign-compare -O3 -s -pipe -mmmx -msse -msse2 -msse3 -mcmodel=large
TARGET = ./all

all: DROMC ROSE FSSP
	
DROMC: spf.h spf.cpp h2h.h h2h.cpp DROMC.cpp EmbeddingService.h EmbeddingService.cpp
	$(CXX) $(CFLAGS) -o DROMC h2h.cpp spf.cpp DROMC.cpp EmbeddingService.cpp $(LIBS)

ROSE: spf.h spf.cpp h2h.h h2h.cpp ROSE.cpp EmbeddingService.h EmbeddingService.cpp
	$(CXX) $(CFLAGS) -o ROSE h2h.cpp spf.cpp ROSE.cpp EmbeddingService.cpp $(LIBS)

FSSP: spf.h spf.cpp h2h.h h2h.cpp FSSP.cpp EmbeddingService.h EmbeddingService.cpp
	$(CXX) $(CFLAGS) -o FSSP h2h.cpp spf.cpp FSSP.cpp EmbeddingService.cpp $(LIBS)

.PHONY: clean
clean:
	-@rm *.o *.gcno *~ 2> /dev/null || true
	-@rm SOR shortestPath SRH Statistic buildIndex 2> /dev/null || true