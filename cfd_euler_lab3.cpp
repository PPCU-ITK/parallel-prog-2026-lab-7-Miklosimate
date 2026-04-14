#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cstdlib>

// -------- Timing and openMP ----------
#include <omp.h>
#include <chrono>

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4; // Ratio of specific heats
const double CFL = 0.5;       // CFL number

// ------------------------------------------------------------
// per-loop timing helpers:
// ------------------------------------------------------------
struct LoopStats
{
    double total_seconds = 0.0;
    long long calls = 0;
};
// RAII-style timer utalizing constructor and destructor to automatically accumulate time spent in a loop
struct ScopedTimer
{
    double t0;
    double &accum;
    long long &calls;
    ScopedTimer(double &accum_, long long &calls_) : t0(omp_get_wtime()), accum(accum_), calls(calls_)
    {
        calls++;
    }
    ~ScopedTimer()
    {
        accum += (omp_get_wtime() - t0);
    }
};

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E)
{
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E,
           double &frho, double &frhou, double &frhov, double &fE)
{
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double &frho, double &frhou, double &frhov, double &fE)
{
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(int argc, char **argv)
{
    // ----- Grid and domain parameters -----
    int Nx = 200;          // Number of cells in x (excluding ghost cells)
    int Ny = 100;          // Number of cells in y
    int nSteps = 2000;
    if (argc > 1)
        Nx = atoi(argv[1]);
    if (argc > 2)
        Ny = atoi(argv[2]);
    if (argc > 3)
        nSteps = atoi(argv[3]);

    const double Lx = 2.0; // Domain length in x
    const double Ly = 1.0; // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);

    vector<double> rho(total_size);
    vector<double> rhou(total_size);
    vector<double> rhov(total_size);
    vector<double> E(total_size);

    vector<double> rho_new(total_size);
    vector<double> rhou_new(total_size);
    vector<double> rhov_new(total_size);
    vector<double> E_new(total_size);

    // A mask to mark solid cells (inside the cylinder)
    vector<bool> solid(total_size, false);

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;     // Cylinder center x
    const double cy = 0.5;     // Cylinder center y
    const double radius = 0.1; // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0 / (gamma_val - 1.0) + 0.5 * rho0 * (u0 * u0 + v0 * v0);

    // ----- Initialize grid and obstacle mask -----
    for (int i = 0; i < Nx + 2; i++)
    {
        for (int j = 0; j < Ny + 2; j++)
        {
            // Compute cell center coordinates
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            // Mark cell as solid if inside the cylinder
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius)
            {
                solid[i * (Ny + 2) + j] = true;
                // For a wall, we set zero velocity
                rho[i * (Ny + 2) + j] = rho0;
                rhou[i * (Ny + 2) + j] = 0.0;
                rhov[i * (Ny + 2) + j] = 0.0;
                E[i * (Ny + 2) + j] = p0 / (gamma_val - 1.0);
            }
            else
            {
                solid[i * (Ny + 2) + j] = false;
                rho[i * (Ny + 2) + j] = rho0;
                rhou[i * (Ny + 2) + j] = rho0 * u0;
                rhov[i * (Ny + 2) + j] = rho0 * v0;
                E[i * (Ny + 2) + j] = E0;
            }
        }
    }

    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0) / 2.0;

    // ---- Timing stats (per loop) ---- 
    // Lab 2 introducted timing, lab 3 re-use the same timers to also print bandwidth estimates
    //----------------------------------
    LoopStats t_bc_left, t_bc_right, t_bc_bottom, t_bc_top;
    LoopStats t_update, t_copyback, t_kinetic;
    double main_loop_seconds = 0.0;

    // ------------------------------------------------------------------------------------------------
    // Lab 3 extension: Bandwidth estimation
    // ------------------------------------------------------------------------------------------------
    // Grid sizes INCLUDING ghost cells (because boundary loops run over ghost lines)
    const int NYG = Ny + 2; // j = 0..Ny+1  -> (Ny+2) elements
    const int NXG = Nx + 2; // i = 0..Nx+1  -> (Nx+2) elements
    // Size of one array element in bytes. -> vector<double> uses doubles which are 8 bytes each
    const double B = 8.0; // bytes per double

    double bytes_bc_left = (double)NYG * (4.0 * B);
    double bytes_bc_right = (double)NYG * (8.0 * B);  
    double bytes_bc_bottom = (double)NXG * (8.0 * B); 
    double bytes_bc_top = (double)NXG * (8.0 * B);

    double bytes_update = (double)Nx * (double)Ny * (20.0*B); 
    double bytes_copyback = (double)Nx * (double)Ny * (8.0 * B); 
    double bytes_kinetic = (double)Nx * (double)Ny * (3.0 * B);

    // ----- Main time-stepping loop -----
    double main_t0 = omp_get_wtime(); // start main loop timer
    for (int n = 0; n < nSteps; n++)
    {
        // --- Apply boundary conditions on ghost cells ---
        // Left boundary (inflow): fixed free-stream state
        { // added local scope to ensure timer is only active during the loop
            // call scope timer to accumulate time spent in this loop
            ScopedTimer timer(t_bc_left.total_seconds, t_bc_left.calls);

            #pragma omp parallel for
            for (int j = 0; j < Ny + 2; j++)
            {
                rho[0 * (Ny + 2) + j] = rho0;       //(wtite) 
                rhou[0 * (Ny + 2) + j] = rho0 * u0; //(wtite)  
                rhov[0 * (Ny + 2) + j] = rho0 * v0; //(wtite) 
                E[0 * (Ny + 2) + j] = E0;           //(wtite) 
            }
            // Bandwidth calculation:
            //   writes: 4 doubles  -> 4 * 8 bytes
            //   reads:  0 counted (we assign constants; scalars neglected)
            // Bytes per call = (Ny+2) * (4 * 8)
            // double bytes_bc_left = (double)NYG * (4.0 * B);

        }
        // Right boundary (outflow): copy from the interior
        {                                                                  // local scope wrapper for timer to only measure the loop
            ScopedTimer timer(t_bc_right.total_seconds, t_bc_right.calls); // timer

            #pragma omp parallel for
            for (int j = 0; j < Ny + 2; j++)
            {
                rho[(Nx + 1) * (Ny + 2) + j] = rho[Nx * (Ny + 2) + j];   //(read+write)
                rhou[(Nx + 1) * (Ny + 2) + j] = rhou[Nx * (Ny + 2) + j]; //(read+write)
                rhov[(Nx + 1) * (Ny + 2) + j] = rhov[Nx * (Ny + 2) + j]; //(read+write)
                E[(Nx + 1) * (Ny + 2) + j] = E[Nx * (Ny + 2) + j];       //(read+write)
            }
            //
            //   reads:  4 doubles (rho,rhou,rhov,E from interior) -> 4 * 8
            //   writes: 4 doubles (rho,rhou,rhov,E to ghost)     -> 4 * 8
            // Total per j = 8 doubles -> 8 * 8 bytes
            // Bytes per call = (Ny+2) * (8 * 8)

            // double bytes_bc_right = (double)NYG * (8.0 * B);

        }
        // Bottom boundary: reflective
        {
            ScopedTimer timer(t_bc_bottom.total_seconds, t_bc_bottom.calls);

            #pragma omp parallel for
            for (int i = 0; i < Nx + 2; i++)
            {
                rho[i * (Ny + 2) + 0] = rho[i * (Ny + 2) + 1];    //(read+write)
                rhou[i * (Ny + 2) + 0] = rhou[i * (Ny + 2) + 1];  //(read+write)
                rhov[i * (Ny + 2) + 0] = -rhov[i * (Ny + 2) + 1]; //(read+write)
                E[i * (Ny + 2) + 0] = E[i * (Ny + 2) + 1];        //(read+write)
            }
            // Per i-element: 4 reads + 4 writes = 8 doubles -> 8 * 8 bytes
            // Bytes per call = (Nx+2) * (8 * 8)

            //double bytes_bc_bottom = (double)NXG * (8.0 * B);
        }
        // Top boundary: reflective
        {
            ScopedTimer timer(t_bc_top.total_seconds, t_bc_top.calls);

            #pragma omp parallel for
            for (int i = 0; i < Nx + 2; i++)
            {
                rho[i * (Ny + 2) + (Ny + 1)] = rho[i * (Ny + 2) + Ny];    //(read+write)
                rhou[i * (Ny + 2) + (Ny + 1)] = rhou[i * (Ny + 2) + Ny];  //(read+write)
                rhov[i * (Ny + 2) + (Ny + 1)] = -rhov[i * (Ny + 2) + Ny]; //(read+write)
                E[i * (Ny + 2) + (Ny + 1)] = E[i * (Ny + 2) + Ny];        //(read+write)
            }
            // Per i-element: 4 reads + 4 writes = 8 doubles -> 8 * 8 bytes

            //double bytes_bc_top = (double)NXG * (8.0 * B);
        }

        // --- Update interior cells using a Lax-Friedrichs scheme ---
        {                                                              // timer scope
            ScopedTimer timer(t_update.total_seconds, t_update.calls); // timer

            #pragma omp parallel for collapse(2) // safe paralell - collapse 2
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    // If the cell is inside the solid obstacle, do not update it
                    if (solid[i * (Ny + 2) + j])
                    {
                        rho_new[i * (Ny + 2) + j]  = rho[i * (Ny + 2) + j];   //(read+write) [ignored in simplified model]
                        rhou_new[i * (Ny + 2) + j] = rhou[i * (Ny + 2) + j];  //(read+write) [ignored in simplified model]
                        rhov_new[i * (Ny + 2) + j] = rhov[i * (Ny + 2) + j];  //(read+write) [ignored in simplified model]
                        E_new[i * (Ny + 2) + j]    = E[i * (Ny + 2) + j];     //(read+write) [ignored in simplified model]
                        continue;
                    }

                    // Compute a Lax averaging of the four neighboring cells
                    rho_new[i * (Ny + 2) + j] = 0.25 * (rho[(i + 1) * (Ny + 2) + j] + rho[(i - 1) * (Ny + 2) + j] +     //1 read + 4 reads
                                                        rho[i * (Ny + 2) + (j + 1)] + rho[i * (Ny + 2) + (j - 1)]);
                    rhou_new[i * (Ny + 2) + j] = 0.25 * (rhou[(i + 1) * (Ny + 2) + j] + rhou[(i - 1) * (Ny + 2) + j] +  //1 read + 4 reads
                                                         rhou[i * (Ny + 2) + (j + 1)] + rhou[i * (Ny + 2) + (j - 1)]);
                    rhov_new[i * (Ny + 2) + j] = 0.25 * (rhov[(i + 1) * (Ny + 2) + j] + rhov[(i - 1) * (Ny + 2) + j] +  //1 read + 4 reads
                                                         rhov[i * (Ny + 2) + (j + 1)] + rhov[i * (Ny + 2) + (j - 1)]);
                    E_new[i * (Ny + 2) + j] = 0.25 * (E[(i + 1) * (Ny + 2) + j] + E[(i - 1) * (Ny + 2) + j] +           //1 read + 4 reads
                                                      E[i * (Ny + 2) + (j + 1)] + E[i * (Ny + 2) + (j - 1)]);           //-----------------
                                                                                                                        // 4 + 16
                    // Compute fluxes
                    double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                    double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                    double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                    double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                    fluxX(rho[(i + 1) * (Ny + 2) + j], rhou[(i + 1) * (Ny + 2) + j], rhov[(i + 1) * (Ny + 2) + j], E[(i + 1) * (Ny + 2) + j],
                          fx_rho1, fx_rhou1, fx_rhov1, fx_E1); // 4 reads -> ignored
                    fluxX(rho[(i - 1) * (Ny + 2) + j], rhou[(i - 1) * (Ny + 2) + j], rhov[(i - 1) * (Ny + 2) + j], E[(i - 1) * (Ny + 2) + j],
                          fx_rho2, fx_rhou2, fx_rhov2, fx_E2); // 4 reads -> ignored
                    fluxY(rho[i * (Ny + 2) + (j + 1)], rhou[i * (Ny + 2) + (j + 1)], rhov[i * (Ny + 2) + (j + 1)], E[i * (Ny + 2) + (j + 1)],
                          fy_rho1, fy_rhou1, fy_rhov1, fy_E1); // 4 reads -> ignored
                    fluxY(rho[i * (Ny + 2) + (j - 1)], rhou[i * (Ny + 2) + (j - 1)], rhov[i * (Ny + 2) + (j - 1)], E[i * (Ny + 2) + (j - 1)],
                          fy_rho2, fy_rhou2, fy_rhov2, fy_E2); // 4 reads -> ignored
                                                           

                    // Apply flux differences
                    double dtdx = dt / (2 * dx);
                    double dtdy = dt / (2 * dy);

                    rho_new[i * (Ny + 2) + j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);       //read+write -> ignored
                    rhou_new[i * (Ny + 2) + j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);  //read+write -> ignored
                    rhov_new[i * (Ny + 2) + j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);  //read+write -> ignored
                    E_new[i * (Ny + 2) + j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);                 //read+write -> ignored
                }                                                                                                 
            }       
        }              // double bytes_update = (double)Nx * (double)Ny * (20.0*B); 


        // Copy updated values back
        {
            ScopedTimer timer(t_copyback.total_seconds, t_copyback.calls);

            #pragma omp parallel for collapse(2)
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    rho[i * (Ny + 2) + j] = rho_new[i * (Ny + 2) + j];      //(read+write)
                    rhou[i * (Ny + 2) + j] = rhou_new[i * (Ny + 2) + j];    //(read+write) 
                    rhov[i * (Ny + 2) + j] = rhov_new[i * (Ny + 2) + j];    //(read+write)
                    E[i * (Ny + 2) + j] = E_new[i * (Ny + 2) + j];          //(read+write)
                }
            } //    double bytes_copyback = (double)Nx * (double)Ny * (8.0 * B); 

        }

        // Calculate total kinetic energy
        double total_kinetic = 0.0;
        {
            ScopedTimer timer(t_kinetic.total_seconds, t_kinetic.calls);

            #pragma omp parallel for collapse(2) reduction(+ : total_kinetic)
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    double u = rhou[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j]; // 2 reads 
                    double v = rhov[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j]; // 2 reads rho -> dupliceate read, ==> 1 read
                    total_kinetic += 0.5 * rho[i * (Ny + 2) + j] * (u * u + v * v); // duplicate read 
                } //double bytes_kinetic = (double)Nx * (double)Ny * (3.0 * B);

            }
        }

        if (n % 50 == 0)
        {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
        }
    }
    main_loop_seconds = omp_get_wtime() - main_t0; // end main loop timer

    // ------------------------------------------------------------------------------------------------
    // Lab 2: Print timing summary per loop
    // ------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------
    // ----- Print timing summary -----
    // ------------------------------------------------------------------------------------------------
    // cout << "\n================ TIMING SUMMARY ================\n";
    // cout << "Main time-stepping loop elapsed time: " << fixed << setprecision(6)
    //      << main_loop_seconds << " s\n\n";

    // auto print_loop = [](const string &name, const LoopStats &s)
    // {
    //     cout << left << setw(30) << name
    //          << " calls=" << setw(8) << s.calls
    //          << " total=" << setw(12) << fixed << setprecision(6) << s.total_seconds << " s"
    //          << " avg=" << setw(12) << (s.calls ? (s.total_seconds / (double)s.calls) : 0.0) << " s\n";
    // };

    // print_loop("BC left (inflow)", t_bc_left);
    // print_loop("BC right (outflow)", t_bc_right);
    // print_loop("BC bottom (refl)", t_bc_bottom);
    // print_loop("BC top (refl)", t_bc_top);
    // print_loop("Update interior", t_update);
    // print_loop("Copyback", t_copyback);
    // print_loop("Total kinetic (red)", t_kinetic);
    // cout << "================================================\n";

    // ------------------------------------------------------------------------------------------------
    // Lab 3 extension: Print memory bandwidth estimates
    // ------------------------------------------------------------------------------------------------
    auto print_loop_bw = [](const string &name, const LoopStats &s, double bytes_per_call)
    {
        double total_bytes = bytes_per_call * (double)s.calls;
        double gbs = (s.total_seconds > 0.0) ? (total_bytes / s.total_seconds / 1e9) : 0.0;

        cout << left << setw(30) << name
             << " Count=" << setw(8) << s.calls
             << " Time=" << setw(12) << fixed << setprecision(6) << s.total_seconds << " s"
             << " GB/s=" << setw(10) << fixed << setprecision(3) << gbs
             << "\n";
    };

    cout << "\nName" << setw(26) << " " << "Count" << setw(6) << " " << "Time" << setw(7) << " " << "GB/s\n";
    print_loop_bw("BC left (inflow)", t_bc_left, bytes_bc_left);
    print_loop_bw("BC right (outflow)", t_bc_right, bytes_bc_right);
    print_loop_bw("BC bottom (refl)", t_bc_bottom, bytes_bc_bottom);
    print_loop_bw("BC top (refl)", t_bc_top, bytes_bc_top);
    print_loop_bw("Update interior", t_update, bytes_update);
    print_loop_bw("Copyback", t_copyback, bytes_copyback);
    print_loop_bw("Total kinetic (red)", t_kinetic, bytes_kinetic);

    double final_total_kinetic = 0.0;
    for (int i = 1; i <= Nx; i++)
    {
        for (int j = 1; j <= Ny; j++)
        {
            double u = rhou[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j];
            double v = rhov[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j];
            final_total_kinetic += 0.5 * rho[i * (Ny + 2) + j] * (u * u + v * v);
        }
    }
    cout << "CPU_RUNTIME_SECONDS," << fixed << setprecision(6) << main_loop_seconds << "\n";
    cout << "CPU_FINAL_KE," << fixed << setprecision(6) << final_total_kinetic << "\n";

    return 0;
}