#ifndef TESTER_H
#define TESTER_H

template <class T>
class Tester {
    public:
        Tester() {
            cudaMalloc(&gpu_p, sizeof(T));
        };
        ~Tester() {
            cudaFree(gpu_p);
        };

    private:
        T* gpu_p;
};

#endif