g++ -c main.cpp -I$RATROOT/include -I$ROOTSYS/include -I/usr/local/cuda/include
nvcc -c beta14.cu
nvcc -o main main.o beta14.o -L$ROOTSYS/lib -lCore -lHist -lTree -L$RATROOT/lib -lRATEvent_Linux-g++

