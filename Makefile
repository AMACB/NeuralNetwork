make:
	g++ main.cpp -o network.out
exec:
	make
	./network.out
clean:
	rm network.out
