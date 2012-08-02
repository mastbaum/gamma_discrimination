/*
 * Compute beta parameters.
 *
 * The so-called beta parameters were developed in SNO as a measure of event
 * isotropy:
 *
 *     The lth beta parameter, beta_l, is defined as the average value of the
 *     Legendre polynomial, P_l, of the cosine of the angle between each pair
 *     PMT hits in the event.
 *
 *         beta_l = <P_l(cos(theta_ik)> where i != k
 *
 *     Again, the angle is taken with respect to the fitted vertex position.
 *     The combination beta_14 = beta_1 + 4 * beta_4 was selected by the SNO
 *     collaboration for use in signal extraction due to the good separability
 *     it provides and the ease of parameterisation of the Gaussian-like
 *     distribution.
 *
 *     - Measurement of the 8B Solar Neutrino Energy Spectrum at the Sudbury
 *       Neutrino Observatory, Jeanne R. Wilson, p. 179 (Ph.D. Thesis)
*/

#include <map>
#include <string>
#include <cuda.h>
#include <stdio.h>

__global__ void VecAdd(float* a, float* b, float* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

extern "C" std::map<std::string, float> calculate_betas() {
    std::map<std::string, float> betas;
    betas["foo"] = 42.0;
    return betas;
}

extern "C" int stuff(int argc, char* argv[]) {
    const int N = 4;
    const size_t size = N * sizeof(float);

    float* a = (float*) malloc(size);
    float* b = (float*) malloc(N * sizeof(float));
    float* c = (float*) malloc(N * sizeof(float));

    float *ad, *bd, *cd;
    cudaMalloc((void**) &ad, size);
    cudaMalloc((void**) &bd, size);
    cudaMalloc((void**) &cd, size);

   for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = 2 * i;
        printf("%u %f %f\n", i, a[i], b[i]);
    }

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

    VecAdd<<<1,N>>>(ad, bd, cd);

    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

    for (unsigned i=0; i<N; i++) {
        printf("%u %f\n", i, c[i]);
    }

    free(a);
    free(b);
    free(c);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    return 0;
}

