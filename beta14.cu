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

__global__ void get_coeffs(int2* pairs, float3* hit_pos, float3* fit_pos, float* p1, float* p4, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        int2 pair = pairs[i];

        float3 A = hit_pos[pair.x];
        float3 B = hit_pos[pair.y];
        float3 C = *fit_pos;

        float c = sqrtf((A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z));
        float a = sqrtf((C.x-B.x)*(C.x-B.x) + (C.y-B.y)*(C.y-B.y) + (C.z-B.z)*(C.z-B.z));
        float b = sqrtf((C.x-A.x)*(C.x-A.x) + (C.y-A.y)*(C.y-A.y) + (C.z-A.z)*(C.z-A.z));

        // dust off that trig!
        p1[i] = (float)(-0.5) * (c*c - a*a - b*b) / (a*b);

        // Legendre P4(x) = (1/8) (25x**4 - 30x**2 + 3)
        p4[i] = (float)(0.125) * (25*p1[i]*p1[i]*p1[i]*p1[i] - 30*p1[i]*p1[i] + 3);
    }
}

float get_beta14(float3 &fit_pos, float3* hit_pos, int nhits) {
    // compute list of pmt pairs (this isn't really necessary)
    const int npairs = nhits * (nhits - 1) / 2;
    int2* pairs = (int2*) malloc(npairs * sizeof(int2));
    unsigned pidx = 0;
    for (unsigned i=0; i<nhits-1; i++) {
        for (unsigned j=i+1; j<nhits; j++) {
            pairs[pidx] = make_int2(i, j);
            pidx++;
        }
    }

    // allocate device memory and copy over
    int2* pairs_device;
    float* p1_device;
    float* p4_device;
    float3* hit_pos_device;
    float3* fit_pos_device;

    cudaMalloc(&pairs_device, npairs * sizeof(int2));
    cudaMalloc(&p1_device, npairs * sizeof(float));
    cudaMalloc(&p4_device, npairs * sizeof(float));
    cudaMalloc(&hit_pos_device, nhits * sizeof(float3));
    cudaMalloc(&fit_pos_device, sizeof(float3));

    cudaMemcpy(pairs_device, pairs, npairs * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(hit_pos_device, hit_pos, nhits * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(fit_pos_device, &fit_pos, sizeof(float3), cudaMemcpyHostToDevice);

    // execute kernel and retreive results
    int blocksize = 512;
    int nblocks = npairs / blocksize + 1;
    get_coeffs<<<nblocks, blocksize>>>(pairs_device, hit_pos_device, fit_pos_device, p1_device, p4_device, npairs);

    float* p1 = (float*) malloc(npairs * sizeof(float));
    float* p4 = (float*) malloc(npairs * sizeof(float));
    cudaMemcpy(p1, p1_device, npairs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p4, p4_device, npairs * sizeof(float), cudaMemcpyDeviceToHost);

    // compute average
    float p1ave = 0.0;
    float p4ave = 0.0;
    for (unsigned i=0; i<npairs; i++) {
        p1ave += p1[i];
        p4ave += p4[i];
    }

    p1ave /= npairs;
    p4ave /= npairs;

    float beta14 = p1ave + 4.0 * p4ave;

    free(pairs);
    free(p1);
    free(p4);
    cudaFree(pairs_device);
    cudaFree(hit_pos_device);
    cudaFree(fit_pos_device);
    cudaFree(p1_device);
    cudaFree(p4_device);

    return beta14;
}

