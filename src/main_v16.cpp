#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>
#include "sim_constants.hpp"
#include "auxiliary.hpp"
#include "observables.hpp"

int main(int argc, char* argv[]) {
    SimParams params;
    get_parameters(argc, argv, params);

    int T_steps = int(params.totaltime / params.dt);
    int N = params.W * params.H;
    int N_total = N * params.num_realizations;

    int T_equilibration = static_cast<int>(params.eqltime/params.dt);
    if (T_equilibration % params.obs_freq != 0) {
        T_equilibration = (T_equilibration / params.obs_freq) * params.obs_freq;
        std::cout << "Adjusted Equilibration Time to align with obs_freq (New step: " << T_equilibration << ")\n";
    }

    int num_records = (T_steps + params.obs_freq - 1) / params.obs_freq;

    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Target energy: H =  " << params.target_E << std::endl;
    std::cout << "Lattice Width:  W = " << params.W << ";\tLattice Height:  H = " << params.H << std::endl;
    std::cout << "Total Time: " << params.totaltime << "\t Equilibration Time: " <<params.eqltime  << "(time steps: " << T_equilibration <<")" <<   std::endl;
    std::cout << "Time step size: dt =  " << params.dt << "\t (# of sim steps: " << T_steps <<")" << std::endl;
    std::cout << "Observation Freq: " << params.obs_freq << " steps\t (Total records: " << num_records << ")" << std::endl;
    std::cout << "Realizations (Batched): " << params.num_realizations << " (Total threads: " << N_total << ")" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    generate_lattice(params.W, params.H);

    std::vector<double> posX_eq(N, 0.0), posY_eq(N, 0.0);
    std::vector<int> neighbors(N * 3, -1);
    load_lattice(params.W, params.H, posX_eq, posY_eq, neighbors);

    std::vector<double> uX(N_total, 0.0), uY(N_total, 0.0);
    std::vector<double> pX(N_total, 0.0), pY(N_total, 0.0);

    double Lx_eq = params.W * (std::sqrt(3.0) / 2.0) * 1.0;
    double Ly_eq = params.H * 1.5 * 1.0;

    std::vector<double> ideal_dx(N * 3, 0.0);
    std::vector<double> ideal_dy(N * 3, 0.0);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            int n_idx = neighbors[k * N + i];
            double dx = posX_eq[i] - posX_eq[n_idx];
            double dy = posY_eq[i] - posY_eq[n_idx];

            dx -= Lx_eq * std::round(dx / Lx_eq);
            dy -= Ly_eq * std::round(dy / Ly_eq);

            double r = std::sqrt(dx * dx + dy * dy);
            if (r > 0) {
                ideal_dx[k * N + i] = (dx / r) * PHYS_R0;
                ideal_dy[k * N + i] = (dy / r) * PHYS_R0;
            }
        }
    }

    std::vector<int> rev_idx(N * 3, -1);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            int n_idx = neighbors[k * N + i];
            for (int m = 0; m < 3; ++m) {
                if (neighbors[m * N + n_idx] == i) {
                    rev_idx[k * N + i] = m;
                    break;
                }
            }
        }
    }

    if (params.target_E > 0.0) {
        double tol = 1e-6;
        int max_iter = 100;

        double com_x = 0.0, com_y = 0.0;
        for (int i = 0; i < N; ++i) {
            com_x += posX_eq[i];
            com_y += posY_eq[i];
        }
        com_x /= N; com_y /= N;

        for (int r = 0; r < params.num_realizations; ++r) {
            int r_offset = r * N;
            std::mt19937 gen(params.seed + r);
            std::normal_distribution<double> dist(0.0, 1.0);

            std::vector<double> base_uX(N, 0.0), base_uY(N, 0.0);
            double sum_ux = 0.0, sum_uy = 0.0;

            for (int i = 0; i < N; ++i) {
                base_uX[i] = dist(gen);
                base_uY[i] = dist(gen);
                sum_ux += base_uX[i];
                sum_uy += base_uY[i];
            }

            for (int i = 0; i < N; ++i) {
                base_uX[i] -= sum_ux / N;
                base_uY[i] -= sum_uy / N;
            }

            double Lz = 0.0;
            double I = 0.0;
            for (int i = 0; i < N; ++i) {
                double rx = posX_eq[i] - com_x;
                double ry = posY_eq[i] - com_y;
                Lz += rx * base_uY[i] - ry * base_uX[i];
                I += rx * rx + ry * ry;
            }
            double d_theta = Lz / I;
            for (int i = 0; i < N; ++i) {
                double rx = posX_eq[i] - com_x;
                double ry = posY_eq[i] - com_y;
                base_uX[i] -= (-d_theta * ry);
                base_uY[i] -= ( d_theta * rx);
            }

            std::vector<double> test_uX(N), test_uY(N);
            auto eval_E = [&](double alpha) {
                for(int i = 0; i < N; ++i) {
                    test_uX[i] = alpha * base_uX[i];
                    test_uY[i] = alpha * base_uY[i];
                }
                return compute_potential_energy_host(N, test_uX, test_uY, neighbors, ideal_dx, ideal_dy);
            };

            double alpha_min = 0.0;
            double alpha_max = 0.001;
            double E_max = eval_E(alpha_max);

            while (E_max < params.target_E) {
                alpha_max *= 2.0;
                E_max = eval_E(alpha_max);
                if (alpha_max > 1e4) {
                    std::cerr << "Warning: Bisection bracket expansion exceeded sensible limits.\n";
                    break;
                }
            }

            double alpha_mid = 0.0;
            for (int iter = 0; iter < max_iter; ++iter) {
                alpha_mid = 0.5 * (alpha_min + alpha_max);
                double E_mid = eval_E(alpha_mid);
                if (std::abs(E_mid - params.target_E) < tol) break;
                if (E_mid < params.target_E) alpha_min = alpha_mid;
                else alpha_max = alpha_mid;
            }

            for (int i = 0; i < N; ++i) {
                uX[r_offset + i] = alpha_mid * base_uX[i];
                uY[r_offset + i] = alpha_mid * base_uY[i];
                pX[r_offset + i] = 0.0;
                pY[r_offset + i] = 0.0;
            }
        }
    }

    double *d_uX, *d_uY, *d_pX, *d_pY;
    double *d_ideal_dx, *d_ideal_dy;
    double *d_obs_T_kin, *d_obs_V_M, *d_obs_V_A;
    double *d_obs_C_MM, *d_obs_C_PP, *d_obs_C_MP, *d_obs_C_PM;
    double *d_M_site, *d_P_site, *d_M_site_t0, *d_P_site_t0;
    int *d_neighbors, *d_rev_idx;

    size_t obs_size = num_records * params.num_realizations * sizeof(double);
    size_t site_size = N_total * sizeof(double);

    cudaMalloc(&d_uX, site_size); cudaMalloc(&d_uY, site_size);
    cudaMalloc(&d_pX, site_size); cudaMalloc(&d_pY, site_size);

    cudaMalloc(&d_ideal_dx, N * 3 * sizeof(double));
    cudaMalloc(&d_ideal_dy, N * 3 * sizeof(double));
    cudaMalloc(&d_neighbors, N * 3 * sizeof(int));
    cudaMalloc(&d_rev_idx, N * 3 * sizeof(int));

    cudaMalloc(&d_M_site, site_size); cudaMalloc(&d_P_site, site_size);
    cudaMalloc(&d_M_site_t0, site_size); cudaMalloc(&d_P_site_t0, site_size);

    cudaMalloc(&d_obs_T_kin, obs_size); cudaMalloc(&d_obs_V_M, obs_size);
    cudaMalloc(&d_obs_V_A, obs_size);
    cudaMalloc(&d_obs_C_MM, obs_size); cudaMalloc(&d_obs_C_PP, obs_size);
    cudaMalloc(&d_obs_C_MP, obs_size); cudaMalloc(&d_obs_C_PM, obs_size);

    cudaMemset(d_M_site_t0, 0, site_size);
    cudaMemset(d_P_site_t0, 0, site_size);

    cudaMemset(d_obs_T_kin, 0, obs_size);
    cudaMemset(d_obs_V_M, 0, obs_size);
    cudaMemset(d_obs_V_A, 0, obs_size);
    cudaMemset(d_obs_C_MM, 0, obs_size);
    cudaMemset(d_obs_C_PP, 0, obs_size);
    cudaMemset(d_obs_C_MP, 0, obs_size);
    cudaMemset(d_obs_C_PM, 0, obs_size);

    cudaMemcpy(d_uX, uX.data(), site_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uY, uY.data(), site_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pX, pX.data(), site_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pY, pY.data(), site_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ideal_dx, ideal_dx.data(), N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ideal_dy, ideal_dy.data(), N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors.data(), N * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rev_idx, rev_idx.data(), N * 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid1D = (N_total + threadsPerBlock - 1) / threadsPerBlock;
    dim3 obsThreads(256);
    dim3 obsBlocks((N + obsThreads.x - 1) / obsThreads.x, params.num_realizations);

    if (T_steps <= T_equilibration) {
        std::cerr << "Error: Total simulation steps must be greater than equilibration steps.\n";
        return 1;
    }

    size_t host_obs_count = num_records * params.num_realizations;
    std::vector<double> h_obs_T_kin(host_obs_count, 0.0);
    std::vector<double> h_obs_V_M(host_obs_count, 0.0);
    std::vector<double> h_obs_V_A(host_obs_count, 0.0);
    std::vector<double> h_obs_C_MM(host_obs_count, 0.0);
    std::vector<double> h_obs_C_PP(host_obs_count, 0.0);
    std::vector<double> h_obs_C_MP(host_obs_count, 0.0);
    std::vector<double> h_obs_C_PM(host_obs_count, 0.0);

    std::map<std::string, std::vector<double>> obs_hist;
    obs_hist["temperature"].assign(host_obs_count, 0.0);
    obs_hist["relative_error"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_MM"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_PP"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_MP"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_PM"].assign(host_obs_count, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now();

    compute_observables(N, params.num_realizations, obsBlocks, obsThreads, 0,
        d_uX, d_uY, d_pX, d_pY, d_neighbors, d_ideal_dx, d_ideal_dy,
        d_obs_T_kin, d_obs_V_M, d_obs_V_A, d_obs_C_MM, d_obs_C_PP, d_obs_C_MP, d_obs_C_PM,
        d_M_site, d_P_site, d_M_site_t0, d_P_site_t0, false);

    double dt = params.dt;

    for (int t = 0; t < T_steps - 1; ++t) {
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a1 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b1 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a2 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b2 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a3 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b3 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a4 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b4 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a4 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b3 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a3 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b2 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a2 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b1 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a1 * dt, N_total, d_uX, d_uY, d_pX, d_pY);

        int next_t = t + 1;
        bool capture_ref = (next_t == T_equilibration);

        if (next_t % params.obs_freq == 0) {
            int record_idx = next_t / params.obs_freq;
            compute_observables(N, params.num_realizations, obsBlocks, obsThreads, record_idx,
                d_uX, d_uY, d_pX, d_pY, d_neighbors, d_ideal_dx, d_ideal_dy,
                d_obs_T_kin, d_obs_V_M, d_obs_V_A, d_obs_C_MM, d_obs_C_PP, d_obs_C_MP, d_obs_C_PM,
                d_M_site, d_P_site, d_M_site_t0, d_P_site_t0, capture_ref);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_obs_T_kin.data(), d_obs_T_kin, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_V_M.data(), d_obs_V_M, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_V_A.data(), d_obs_V_A, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_C_MM.data(), d_obs_C_MM, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_C_PP.data(), d_obs_C_PP, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_C_MP.data(), d_obs_C_MP, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_C_PM.data(), d_obs_C_PM, obs_size, cudaMemcpyDeviceToHost);

    std::vector<double> h_H0(params.num_realizations, 0.0);
    std::vector<double> ref_mean_M(params.num_realizations, 0.0);
    std::vector<double> ref_mean_P(params.num_realizations, 0.0);
    int eq_idx = T_equilibration / params.obs_freq;

    for (int r = 0; r < params.num_realizations; ++r) {
        h_H0[r] = h_obs_T_kin[r] + h_obs_V_M[r] + h_obs_V_A[r];
        int ref_idx = eq_idx * params.num_realizations + r;
        ref_mean_M[r] = h_obs_T_kin[ref_idx] / N;
        ref_mean_P[r] = (h_obs_V_M[ref_idx] + h_obs_V_A[ref_idx]) / N;
    }

    for (int record_idx = 0; record_idx < num_records; ++record_idx) {
        for (int r = 0; r < params.num_realizations; ++r) {
            int idx = record_idx * params.num_realizations + r;
            int actual_t = record_idx * params.obs_freq;

            double H_t = h_obs_T_kin[idx] + h_obs_V_M[idx] + h_obs_V_A[idx];
            obs_hist["temperature"][idx] = (h_obs_T_kin[idx] / (N * PHYS_K_B));
            obs_hist["relative_error"][idx] = std::abs(H_t - h_H0[r]) / h_H0[r];

            if (actual_t >= T_equilibration) {
                obs_hist["acf_C_MM"][idx] = (h_obs_C_MM[idx] / N) - (ref_mean_M[r] * ref_mean_M[r]);
                obs_hist["acf_C_PP"][idx] = (h_obs_C_PP[idx] / N) - (ref_mean_P[r] * ref_mean_P[r]);
                obs_hist["acf_C_MP"][idx] = (h_obs_C_MP[idx] / N) - (ref_mean_M[r] * ref_mean_P[r]);
                obs_hist["acf_C_PM"][idx] = (h_obs_C_PM[idx] / N) - (ref_mean_P[r] * ref_mean_M[r]);
            }
        }
    }

    std::vector<double> norm_C_MM(params.num_realizations, 0.0);
    std::vector<double> norm_C_PP(params.num_realizations, 0.0);

    for (int r = 0; r < params.num_realizations; ++r) {
        int ref_idx = eq_idx * params.num_realizations + r;
        norm_C_MM[r] = obs_hist["acf_C_MM"][ref_idx];
        norm_C_PP[r] = obs_hist["acf_C_PP"][ref_idx];
    }

    for (int record_idx = eq_idx; record_idx < num_records; ++record_idx) {
        for (int r = 0; r < params.num_realizations; ++r) {
            int idx = record_idx * params.num_realizations + r;

            double v_M = norm_C_MM[r];
            double v_P = norm_C_PP[r];
            double v_Cross = std::sqrt(std::max(0.0, v_M * v_P));

            obs_hist["acf_C_MM"][idx] = (v_M != 0.0) ? (obs_hist["acf_C_MM"][idx] / v_M) : 0.0;
            obs_hist["acf_C_PP"][idx] = (v_P != 0.0) ? (obs_hist["acf_C_PP"][idx] / v_P) : 0.0;
            obs_hist["acf_C_MP"][idx] = (v_Cross != 0.0) ? (obs_hist["acf_C_MP"][idx] / v_Cross) : 0.0;
            obs_hist["acf_C_PM"][idx] = (v_Cross != 0.0) ? (obs_hist["acf_C_PM"][idx] / v_Cross) : 0.0;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Simulation computation time: " << diff.count() << " seconds.\n";

    save_observables(obs_hist, params);
    std::cout << "Observables saved to ../data/ directory.\n";

    cudaFree(d_uX); cudaFree(d_uY);
    cudaFree(d_pX); cudaFree(d_pY);
    cudaFree(d_neighbors); cudaFree(d_rev_idx);
    cudaFree(d_ideal_dx); cudaFree(d_ideal_dy);
    cudaFree(d_M_site); cudaFree(d_P_site);
    cudaFree(d_M_site_t0); cudaFree(d_P_site_t0);
    cudaFree(d_obs_T_kin); cudaFree(d_obs_V_M); cudaFree(d_obs_V_A);
    cudaFree(d_obs_C_MM); cudaFree(d_obs_C_PP);
    cudaFree(d_obs_C_MP); cudaFree(d_obs_C_PM);

    return 0;
}
