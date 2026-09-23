#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "sim_constants.hpp"
#include <format>

__global__ void calc_site_energies_kernel(int N, int num_realizations, int t,
                        const double* uX, const double* uY,
                        const double* pX, const double* pY,
                        const int* neighbors,
                        const double* ideal_dx, const double* ideal_dy,
                        double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                        double* d_M_site, double* d_P_site) {

    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (r >= num_realizations) return;

    int global_id = r * N + i;
    int r_offset = r * N;

    __shared__ double s_T_kin[256];
    __shared__ double s_V_M[256];
    __shared__ double s_V_A[256];

    s_T_kin[tid] = 0.0;
    s_V_M[tid] = 0.0;
    s_V_A[tid] = 0.0;

    if (i < N) {
        double T_kin = (pX[global_id] * pX[global_id] + pY[global_id] * pY[global_id]) / (2.0 * PHYS_M_C);
        double V_M = 0.0;
        double local_V_M = 0.0;
        double V_A = 0.0;

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            int n_global = r_offset + n_local;
            double dx = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
            double dy = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);
            double r_dist = sqrt(dx * dx + dy * dy);
            double exp_term = exp(-PHYS_A * (r_dist - PHYS_R0));
            double bond_E = PHYS_D * (exp_term - 1.0) * (exp_term - 1.0);

            if (i < n_local) {
                V_M += bond_E;
            }
            local_V_M += 0.5 * bond_E;
        }

        auto get_angle_energy = [&](int l1_local, int l1_k, int l2_local, int l2_k) {
            int l1_global = r_offset + l1_local;
            int l2_global = r_offset + l2_local;

            double dx1 = -ideal_dx[l1_k * N + i] + (uX[l1_global] - uX[global_id]);
            double dy1 = -ideal_dy[l1_k * N + i] + (uY[l1_global] - uY[global_id]);
            double r1 = sqrt(dx1*dx1 + dy1*dy1);
            double dx2 = -ideal_dx[l2_k * N + i] + (uX[l2_global] - uX[global_id]);
            double dy2 = -ideal_dy[l2_k * N + i] + (uY[l2_global] - uY[global_id]);
            double r2 = sqrt(dx2*dx2 + dy2*dy2);

            if (r1 == 0.0 || r2 == 0.0) return 0.0;
            double cos_phi = (dx1*dx2 + dy1*dy2) / (r1 * r2);
            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
            double phi = acos(cos_phi);
            double dphi = phi - PHYS_PHI0;

            return 0.5 * PHYS_D_ANG * dphi * dphi - (1.0 / 3.0) * PHYS_D_PRIME * dphi * dphi * dphi;
        };

        int n0 = neighbors[0 * N + i];
        int n1 = neighbors[1 * N + i];
        int n2 = neighbors[2 * N + i];

        V_A += get_angle_energy(n0, 0, n1, 1);
        V_A += get_angle_energy(n1, 1, n2, 2);
        V_A += get_angle_energy(n2, 2, n0, 0);

        // Store site values for M (kinetic) and P (potential)
        d_M_site[global_id] = T_kin;
        d_P_site[global_id] = local_V_M + V_A;

        s_T_kin[tid] = T_kin;
        s_V_M[tid] = V_M;
        s_V_A[tid] = V_A;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_T_kin[tid] += s_T_kin[tid + s];
            s_V_M[tid] += s_V_M[tid + s];
            s_V_A[tid] += s_V_A[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int obs_index = t * num_realizations + r;
        atomicAdd(&d_obs_T_kin[obs_index], s_T_kin[0]);
        atomicAdd(&d_obs_V_M[obs_index], s_V_M[0]);
        atomicAdd(&d_obs_V_A[obs_index], s_V_A[0]);
    }
}

__global__ void calc_matrix_corr_kernel(int N, int num_realizations, int t,
                        const double* d_M_site_t0, const double* d_P_site_t0,
                        const double* d_M_site_t, const double* d_P_site_t,
                        double* d_obs_C_MM, double* d_obs_C_PP,
                        double* d_obs_C_MP, double* d_obs_C_PM) {
    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (r >= num_realizations) return;

    int global_id = r * N + i;

    __shared__ double s_MM[256];
    __shared__ double s_PP[256];
    __shared__ double s_MP[256];
    __shared__ double s_PM[256];

    s_MM[tid] = 0.0;
    s_PP[tid] = 0.0;
    s_MP[tid] = 0.0;
    s_PM[tid] = 0.0;

    if (i < N) {
        double my_M_t0 = d_M_site_t0[global_id];
        double my_P_t0 = d_P_site_t0[global_id];
        double my_M_t  = d_M_site_t[global_id];
        double my_P_t  = d_P_site_t[global_id];

        s_MM[tid] = my_M_t0 * my_M_t;
        s_PP[tid] = my_P_t0 * my_P_t;
        s_MP[tid] = my_M_t0 * my_P_t;
        s_PM[tid] = my_P_t0 * my_M_t;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_MM[tid] += s_MM[tid + s];
            s_PP[tid] += s_PP[tid + s];
            s_MP[tid] += s_MP[tid + s];
            s_PM[tid] += s_PM[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int obs_index = t * num_realizations + r;
        atomicAdd(&d_obs_C_MM[obs_index], s_MM[0]);
        atomicAdd(&d_obs_C_PP[obs_index], s_PP[0]);
        atomicAdd(&d_obs_C_MP[obs_index], s_MP[0]);
        atomicAdd(&d_obs_C_PM[obs_index], s_PM[0]);
    }
}

inline void compute_observables(int N, int num_realizations, dim3 blocksPerGrid, dim3 threadsPerBlock, int t,
                         const double* d_uX, const double* d_uY,
                         const double* d_pX, const double* d_pY,
                         const int* d_neighbors,
                         const double* d_ideal_dx, const double* d_ideal_dy,
                         double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                         double* d_obs_C_MM, double* d_obs_C_PP, double* d_obs_C_MP, double* d_obs_C_PM,
                         double* d_M_site, double* d_P_site,
                         double* d_M_site_t0, double* d_P_site_t0,
                         bool capture_ref) {

    calc_site_energies_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, num_realizations, t,
        d_uX, d_uY, d_pX, d_pY,
        d_neighbors, d_ideal_dx, d_ideal_dy,
        d_obs_T_kin, d_obs_V_M, d_obs_V_A,
        d_M_site, d_P_site);

    if (capture_ref) {
        size_t bytes = N * num_realizations * sizeof(double);
        cudaMemcpy(d_M_site_t0, d_M_site, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_site_t0, d_P_site, bytes, cudaMemcpyDeviceToDevice);
    }

    calc_matrix_corr_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, num_realizations, t,
        d_M_site_t0, d_P_site_t0,
        d_M_site, d_P_site,
        d_obs_C_MM, d_obs_C_PP, d_obs_C_MP, d_obs_C_PM);
}

inline void save_observables(const std::map<std::string, std::vector<double>>& obs_histories, SimParams params) {
    for (const auto& pair : obs_histories) {
        std::string filename = std::format("../data/{}_w-{}_h-{}_H-{:.2f}_s-{}_tt-{}_eqt-{:.3f}_dt-{:.3f}_r-{}_of-{}.bin",
            pair.first, params.W, params.H, params.target_E,
            params.seed, params.totaltime, params.eqltime, params.dt, params.num_realizations, params.obs_freq);

        std::ofstream out_file(filename, std::ios::binary);
        if (out_file) {
            out_file.write(reinterpret_cast<const char*>(pair.second.data()), pair.second.size() * sizeof(double));
            out_file.close();
        } else {
            std::cerr << "Error: Could not open " << filename << " for writing.\n";
        }
    }
}
