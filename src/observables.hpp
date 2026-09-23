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

__global__ void calc_observables_kernel(int N, int num_realizations, int t,
                        const double* uX, const double* uY,
                        const double* uX_ref, const double* uY_ref,
                        const double* pX, const double* pY,
                        const double* pX_ref, const double* pY_ref,
                        const int* neighbors,
                        const double* ideal_dx, const double* ideal_dy,
                        double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                        double* d_obs_u_dot_u0, double* d_obs_Et_mult_E0,
                        double* d_obs_rel_u_dot_u0,
                        double* d_obs_v_dot_v0, double* d_obs_len_t,
                        double* d_E_ref,
                        double* d_S_ref, double* d_B_ref,
                        double* d_obs_S_mean, double* d_obs_B_mean,
                        double* d_obs_C_SS, double* d_obs_C_SB, double* d_obs_C_BS, double* d_obs_C_BB,
                        double* d_obs_G_SS, double* d_obs_G_SB, double* d_obs_G_BB,
                        bool capture_ref) {

    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (r >= num_realizations) return;

    int global_id = r * N + i;
    int r_offset = r * N;

    // Single shared array to be recycled (drops shared memory usage from 34.8 KB to 2 KB)
    __shared__ double s_reduce[256];

    double val_T_kin = 0.0, val_V_M = 0.0, val_V_A = 0.0;
    double val_u_dot_u0 = 0.0, val_Et_mult_E0 = 0.0;
    double val_rel_u_dot_u0 = 0.0, val_v_dot_v0 = 0.0, val_len_t = 0.0;
    double val_S = 0.0, val_B = 0.0;
    double val_C_SS = 0.0, val_C_SB = 0.0, val_C_BS = 0.0, val_C_BB = 0.0;
    double val_G_SS = 0.0, val_G_SB = 0.0, val_G_BB = 0.0;

    if (i < N) {
        val_T_kin = (pX[global_id] * pX[global_id] + pY[global_id] * pY[global_id]) / (2.0 * PHYS_M_C);

        double local_V_M = 0.0;
        double local_dS_i = 0.0;
        double local_dB_i = 0.0;
        double vX_i = pX[global_id] / PHYS_M_C;
        double vY_i = pY[global_id] / PHYS_M_C;

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            int n_global = r_offset + n_local;
            double dx = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
            double dy = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);
            double r_dist = sqrt(dx * dx + dy * dy);

            if (r_dist > 0) {
                double exp_term = exp(-PHYS_A * (r_dist - PHYS_R0));
                double bond_E = PHYS_D * (exp_term - 1.0) * (exp_term - 1.0);

                if (i < n_local) {
                    val_V_M += bond_E;
                }
                local_V_M += 0.5 * bond_E;

                double force_mag = 2.0 * PHYS_A * PHYS_D * exp_term * (exp_term - 1.0);
                double Fx_i = force_mag * (dx / r_dist);
                double Fy_i = force_mag * (dy / r_dist);
                double vX_n = pX[n_global] / PHYS_M_C;
                double vY_n = pY[n_global] / PHYS_M_C;

                local_dS_i += -0.5 * (Fx_i * (vX_i - vX_n) + Fy_i * (vY_i - vY_n));
            }
        }

        auto add_angle_energy_and_dot = [&](int l1_local, int l1_k, int l2_local, int l2_k, double& ang_E, double& ang_dot) {
            int l1_global = r_offset + l1_local;
            int l2_global = r_offset + l2_local;

            double dx1_id = -ideal_dx[l1_k * N + i];
            double dy1_id = -ideal_dy[l1_k * N + i];
            double dx2_id = -ideal_dx[l2_k * N + i];
            double dy2_id = -ideal_dy[l2_k * N + i];

            double dx1 = dx1_id + (uX[l1_global] - uX[global_id]);
            double dy1 = dy1_id + (uY[l1_global] - uY[global_id]);
            double r1 = sqrt(dx1*dx1 + dy1*dy1);

            double dx2 = dx2_id + (uX[l2_global] - uX[global_id]);
            double dy2 = dy2_id + (uY[l2_global] - uY[global_id]);
            double r2 = sqrt(dx2*dx2 + dy2*dy2);

            if (r1 == 0.0 || r2 == 0.0) return;
            double cos_phi = (dx1*dx2 + dy1*dy2) / (r1 * r2);
            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
            double phi = acos(cos_phi);
            double dphi = phi - PHYS_PHI0;

            ang_E += 0.5 * PHYS_D_ANG * dphi * dphi - (1.0 / 3.0) * PHYS_D_PRIME * dphi * dphi * dphi;

            double sin_phi = sqrt(1.0 - cos_phi * cos_phi);
            if (sin_phi < 1e-16) return;

            double dV_dphi = PHYS_D_ANG * dphi - PHYS_D_PRIME * dphi * dphi;
            double P = -dV_dphi / sin_phi;

            double u1x = dx1 / r1, u1y = dy1 / r1;
            double u2x = dx2 / r2, u2y = dy2 / r2;

            double grad_l1_x = (u2x - cos_phi * u1x) / r1;
            double grad_l1_y = (u2y - cos_phi * u1y) / r1;
            double grad_l2_x = (u1x - cos_phi * u2x) / r2;
            double grad_l2_y = (u1y - cos_phi * u2y) / r2;
            double grad_v_x = -(grad_l1_x + grad_l2_x);
            double grad_v_y = -(grad_l1_y + grad_l2_y);

            double vX_l1 = pX[l1_global] / PHYS_M_C;
            double vY_l1 = pY[l1_global] / PHYS_M_C;
            double vX_l2 = pX[l2_global] / PHYS_M_C;
            double vY_l2 = pY[l2_global] / PHYS_M_C;

            ang_dot += P * grad_l1_x * vX_l1 + P * grad_l1_y * vY_l1
                     + P * grad_l2_x * vX_l2 + P * grad_l2_y * vY_l2
                     + P * grad_v_x * vX_i + P * grad_v_y * vY_i;
        };

        int n0 = neighbors[0 * N + i];
        int n1 = neighbors[1 * N + i];
        int n2 = neighbors[2 * N + i];

        add_angle_energy_and_dot(n0, 0, n1, 1, val_V_A, local_dB_i);
        add_angle_energy_and_dot(n1, 1, n2, 2, val_V_A, local_dB_i);
        add_angle_energy_and_dot(n2, 2, n0, 0, val_V_A, local_dB_i);

        double local_E = val_T_kin + local_V_M + val_V_A;
        val_S = local_V_M;
        val_B = val_V_A;

        if (capture_ref) {
            d_E_ref[global_id] = local_E;
            d_S_ref[global_id] = val_S;
            d_B_ref[global_id] = val_B;
        }

        val_u_dot_u0 = (uX[global_id] * uX_ref[global_id]) + (uY[global_id] * uY_ref[global_id]);
        val_Et_mult_E0 = local_E * d_E_ref[global_id];
        val_v_dot_v0 = (pX[global_id] * pX_ref[global_id] + pY[global_id] * pY_ref[global_id]) / (PHYS_M_C * PHYS_M_C);

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            if (i < n_local) {
                int n_global = r_offset + n_local;
                double rx_t = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
                double ry_t = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);
                double rx_ref = ideal_dx[k * N + i] + (uX_ref[global_id] - uX_ref[n_global]);
                double ry_ref = ideal_dy[k * N + i] + (uY_ref[global_id] - uY_ref[n_global]);

                double len_t = sqrt(rx_t * rx_t + ry_t * ry_t);
                double len_ref = sqrt(rx_ref * rx_ref + ry_ref * ry_ref);

                val_rel_u_dot_u0 += len_t * len_ref;
                val_len_t += len_t;
            }
        }

        val_C_SS = val_S * d_S_ref[global_id];
        val_C_SB = val_S * d_B_ref[global_id];
        val_C_BS = val_B * d_S_ref[global_id];
        val_C_BB = val_B * d_B_ref[global_id];

        val_G_SS = local_dS_i * local_dS_i;
        val_G_SB = local_dS_i * local_dB_i;
        val_G_BB = local_dB_i * local_dB_i;
    }

    int obs_index = t * num_realizations + r;

    // Streamlined block reduction targeting the recycled 2KB shared array
    auto reduce_and_add = [&](double local_val, double* d_out) {
        s_reduce[tid] = local_val;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_reduce[tid] += s_reduce[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(&d_out[obs_index], s_reduce[0]);
        }
        __syncthreads(); // Prevent the next variable reduction from trampling the data
    };

    reduce_and_add(val_T_kin, d_obs_T_kin);
    reduce_and_add(val_V_M, d_obs_V_M);
    reduce_and_add(val_V_A, d_obs_V_A);
    reduce_and_add(val_u_dot_u0, d_obs_u_dot_u0);
    reduce_and_add(val_Et_mult_E0, d_obs_Et_mult_E0);
    reduce_and_add(val_rel_u_dot_u0, d_obs_rel_u_dot_u0);
    reduce_and_add(val_v_dot_v0, d_obs_v_dot_v0);
    reduce_and_add(val_len_t, d_obs_len_t);
    reduce_and_add(val_S, d_obs_S_mean);
    reduce_and_add(val_B, d_obs_B_mean);
    reduce_and_add(val_C_SS, d_obs_C_SS);
    reduce_and_add(val_C_SB, d_obs_C_SB);
    reduce_and_add(val_C_BS, d_obs_C_BS);
    reduce_and_add(val_C_BB, d_obs_C_BB);
    reduce_and_add(val_G_SS, d_obs_G_SS);
    reduce_and_add(val_G_SB, d_obs_G_SB);
    reduce_and_add(val_G_BB, d_obs_G_BB);
}

inline void compute_observables(int N, int num_realizations, dim3 blocksPerGrid, dim3 threadsPerBlock, int t,
                         const double* d_uX, const double* d_uY,
                         const double* d_uX_ref, const double* d_uY_ref,
                         const double* d_pX, const double* d_pY,
                         const double* d_pX_ref, const double* d_pY_ref,
                         const int* d_neighbors,
                         const double* d_ideal_dx, const double* d_ideal_dy,
                         double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                         double* d_obs_u_dot_u0, double* d_obs_Et_mult_E0,
                         double* d_obs_rel_u_dot_u0,
                         double* d_obs_v_dot_v0, double* d_obs_len_t,
                         double* d_E_ref,
                         double* d_S_ref, double* d_B_ref,
                         double* d_obs_S_mean, double* d_obs_B_mean,
                         double* d_obs_C_SS, double* d_obs_C_SB, double* d_obs_C_BS, double* d_obs_C_BB,
                         double* d_obs_G_SS, double* d_obs_G_SB, double* d_obs_G_BB,
                         bool capture_ref) {

    calc_observables_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, num_realizations, t,
        d_uX, d_uY, d_uX_ref, d_uY_ref, d_pX, d_pY, d_pX_ref, d_pY_ref,
        d_neighbors, d_ideal_dx, d_ideal_dy,
        d_obs_T_kin, d_obs_V_M, d_obs_V_A, d_obs_u_dot_u0, d_obs_Et_mult_E0,
        d_obs_rel_u_dot_u0, d_obs_v_dot_v0, d_obs_len_t,
        d_E_ref, d_S_ref, d_B_ref,
        d_obs_S_mean, d_obs_B_mean,
        d_obs_C_SS, d_obs_C_SB, d_obs_C_BS, d_obs_C_BB,
        d_obs_G_SS, d_obs_G_SB, d_obs_G_BB,
        capture_ref);
}

inline void save_observables(const std::map<std::string, std::vector<double>>& obs_histories, SimParams params) {
    for (const auto& pair : obs_histories) {
        std::string filename = std::format("../data/{}_w-{}_h-{}_H-{:.2f}_s-{}_tt-{}_eqt-{:.3f}_dt-{:.3f}_r-{}_of-{}.bin",
            pair.first, params.W, params.H, params.target_E,
            params.seed, params.totaltime,params.eqltime , params.dt, params.num_realizations, params.obs_freq);

        std::ofstream out_file(filename, std::ios::binary);
        if (out_file) {
            out_file.write(reinterpret_cast<const char*>(pair.second.data()), pair.second.size() * sizeof(double));
            out_file.close();
        } else {
            std::cerr << "Error: Could not open " << filename << " for writing.\n";
        }
    }
}
