make:
	g++ network.cpp -o network.out
exec:
	make
	./network.out
clean:
	rm network.out
