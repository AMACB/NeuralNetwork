CC=g++
CFLAGS=-c -O2 --std=c++11 -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused
LDFLAGS=
SOURCES=src/main.cpp src/matrix.cpp src/mnist.cpp src/network.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=network

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm $(OBJECTS)
	    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
