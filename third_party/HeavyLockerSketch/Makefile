CPPFLAGS = -Wall -O3 -std=c++14 -lm -w -mcmodel=medium -g
PROGRAMS = merge

all: $(PROGRAMS)

merge: mergetest.cpp BaseSketch.h\
	BOBHASH32.h BOBHASH64.h params.h ssummary.h CMSketch.h ElasticSketch.h \
	MVSketch.h Uss.h DASketch.h newMSketch.h goodMSketch.h
	g++ -o merge_add mergetest.cpp $(CPPFLAGS)

clean:
	rm -f *.o $(PROGRAMS)
