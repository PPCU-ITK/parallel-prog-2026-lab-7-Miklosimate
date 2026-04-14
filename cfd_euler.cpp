#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <omp.h>

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4; // Ratio of specific heats
const double CFL = 0.5;       // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
// Lab 7: compile helper routines for both host and GPU so target kernels can call them.
#pragma omp declare target
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

// Lab 7: end the GPU-compilation region for the helper functions above.
#pragma omp end declare target

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(int argc, char **argv)
{
    int Nx = 200; // Number of cells in x (excluding ghost cells)
    int Ny = 100; // Number of cells in y
    int nSteps = 2000;

    // Lab 7: command line arg: let the runner script sweep Nx, Ny, and nSteps for CPU/GPU comparisons.
    if (argc > 1)
        Nx = atoi(argv[1]);
    if (argc > 2)
        Ny = atoi(argv[2]);
    if (argc > 3)
        nSteps = atoi(argv[3]);

    // ----- Grid and domain parameters -----
    const double Lx = 2.0; // Domain length in x
    const double Ly = 1.0; // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);

    double *rho = (double *)malloc(total_size * sizeof(double));
    double *rhou = (double *)malloc(total_size * sizeof(double));
    double *rhov = (double *)malloc(total_size * sizeof(double));
    double *E = (double *)malloc(total_size * sizeof(double));
    double *rho_new = (double *)malloc(total_size * sizeof(double));
    double *rhou_new = (double *)malloc(total_size * sizeof(double));
    double *rhov_new = (double *)malloc(total_size * sizeof(double));
    double *E_new = (double *)malloc(total_size * sizeof(double));

    // Boolean mask for solid cells
    bool *solid = (bool *)malloc(total_size * sizeof(bool));

    // Remember to initialize if needed
    for (int i = 0; i < total_size; i++)
    {
        rho[i] = 0.0;
        rhou[i] = 0.0;
        rhov[i] = 0.0;
        E[i] = 0.0;
        rho_new[i] = 0.0;
        rhou_new[i] = 0.0;
        rhov_new[i] = 0.0;
        E_new[i] = 0.0;
        solid[i] = false;
    }

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

    double main_t0 = omp_get_wtime();

    double total_kinetic = 0.0;

// Lab 7: keep the main arrays on the GPU across the time-stepping loop.
#pragma omp target data map(tofrom : rho[0 : total_size], rhou[0 : total_size], rhov[0 : total_size], E[0 : total_size], rho_new[0 : total_size], rhou_new[0 : total_size], rhov_new[0 : total_size], E_new[0 : total_size]) map(to : solid[0 : total_size])
    {
        // ----- Main time-stepping loop -----
        for (int n = 0; n < nSteps; n++)
        {
// --- Apply boundary conditions on ghost cells ---
// Left boundary (inflow): fixed free-stream state

// Lab 7: offload each boundary loop to the GPU, matching the CPU OpenMP structure.
#pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny + 2; j++)
            {
                rho[0 * (Ny + 2) + j] = rho0;
                rhou[0 * (Ny + 2) + j] = rho0 * u0;
                rhov[0 * (Ny + 2) + j] = rho0 * v0;
                E[0 * (Ny + 2) + j] = E0;
            }
// Right boundary (outflow): copy from the interior
#pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny + 2; j++)
            {
                rho[(Nx + 1) * (Ny + 2) + j] = rho[Nx * (Ny + 2) + j];
                rhou[(Nx + 1) * (Ny + 2) + j] = rhou[Nx * (Ny + 2) + j];
                rhov[(Nx + 1) * (Ny + 2) + j] = rhov[Nx * (Ny + 2) + j];
                E[(Nx + 1) * (Ny + 2) + j] = E[Nx * (Ny + 2) + j];
            }
// Bottom boundary: reflective
#pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx + 2; i++)
            {
                rho[i * (Ny + 2) + 0] = rho[i * (Ny + 2) + 1];
                rhou[i * (Ny + 2) + 0] = rhou[i * (Ny + 2) + 1];
                rhov[i * (Ny + 2) + 0] = -rhov[i * (Ny + 2) + 1];
                E[i * (Ny + 2) + 0] = E[i * (Ny + 2) + 1];
            }
// Top boundary: reflective
#pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx + 2; i++)
            {
                rho[i * (Ny + 2) + (Ny + 1)] = rho[i * (Ny + 2) + Ny];
                rhou[i * (Ny + 2) + (Ny + 1)] = rhou[i * (Ny + 2) + Ny];
                rhov[i * (Ny + 2) + (Ny + 1)] = -rhov[i * (Ny + 2) + Ny];
                E[i * (Ny + 2) + (Ny + 1)] = E[i * (Ny + 2) + Ny];
            }

// --- Update interior cells using a Lax-Friedrichs scheme ---

// Lab 7: run the interior update on the GPU inside the persistent data region.
#pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    // If the cell is inside the solid obstacle, do not update it
                    if (solid[i * (Ny + 2) + j])
                    {
                        rho_new[i * (Ny + 2) + j] = rho[i * (Ny + 2) + j];
                        rhou_new[i * (Ny + 2) + j] = rhou[i * (Ny + 2) + j];
                        rhov_new[i * (Ny + 2) + j] = rhov[i * (Ny + 2) + j];
                        E_new[i * (Ny + 2) + j] = E[i * (Ny + 2) + j];
                        continue;
                    }

                    // Compute a Lax averaging of the four neighboring cells
                    rho_new[i * (Ny + 2) + j] = 0.25 * (rho[(i + 1) * (Ny + 2) + j] + rho[(i - 1) * (Ny + 2) + j] +
                                                        rho[i * (Ny + 2) + (j + 1)] + rho[i * (Ny + 2) + (j - 1)]);
                    rhou_new[i * (Ny + 2) + j] = 0.25 * (rhou[(i + 1) * (Ny + 2) + j] + rhou[(i - 1) * (Ny + 2) + j] +
                                                         rhou[i * (Ny + 2) + (j + 1)] + rhou[i * (Ny + 2) + (j - 1)]);
                    rhov_new[i * (Ny + 2) + j] = 0.25 * (rhov[(i + 1) * (Ny + 2) + j] + rhov[(i - 1) * (Ny + 2) + j] +
                                                         rhov[i * (Ny + 2) + (j + 1)] + rhov[i * (Ny + 2) + (j - 1)]);
                    E_new[i * (Ny + 2) + j] = 0.25 * (E[(i + 1) * (Ny + 2) + j] + E[(i - 1) * (Ny + 2) + j] +
                                                      E[i * (Ny + 2) + (j + 1)] + E[i * (Ny + 2) + (j - 1)]);

                    // Compute fluxes
                    double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                    double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                    double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                    double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                    fluxX(rho[(i + 1) * (Ny + 2) + j], rhou[(i + 1) * (Ny + 2) + j], rhov[(i + 1) * (Ny + 2) + j], E[(i + 1) * (Ny + 2) + j],
                          fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                    fluxX(rho[(i - 1) * (Ny + 2) + j], rhou[(i - 1) * (Ny + 2) + j], rhov[(i - 1) * (Ny + 2) + j], E[(i - 1) * (Ny + 2) + j],
                          fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                    fluxY(rho[i * (Ny + 2) + (j + 1)], rhou[i * (Ny + 2) + (j + 1)], rhov[i * (Ny + 2) + (j + 1)], E[i * (Ny + 2) + (j + 1)],
                          fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                    fluxY(rho[i * (Ny + 2) + (j - 1)], rhou[i * (Ny + 2) + (j - 1)], rhov[i * (Ny + 2) + (j - 1)], E[i * (Ny + 2) + (j - 1)],
                          fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                    // Apply flux differences
                    double dtdx = dt / (2 * dx);
                    double dtdy = dt / (2 * dy);

                    rho_new[i * (Ny + 2) + j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                    rhou_new[i * (Ny + 2) + j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                    rhov_new[i * (Ny + 2) + j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                    E_new[i * (Ny + 2) + j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
                }
            }
// Lab 7: copy the new state back so the next step uses the updated solution.
// Copy updated values back
#pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    rho[i * (Ny + 2) + j] = rho_new[i * (Ny + 2) + j];
                    rhou[i * (Ny + 2) + j] = rhou_new[i * (Ny + 2) + j];
                    rhov[i * (Ny + 2) + j] = rhov_new[i * (Ny + 2) + j];
                    E[i * (Ny + 2) + j] = E_new[i * (Ny + 2) + j];
                }
            }
            // Lab 7: compute kinetic energy on the GPU so the runtime comparison is fair.

            // Calculate total kinetic energy
            total_kinetic = 0.0;
#pragma omp target teams distribute parallel for collapse(2) reduction(+ : total_kinetic)
            for (int i = 1; i <= Nx; i++)
            {
                for (int j = 1; j <= Ny; j++)
                {
                    double u = rhou[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j];
                    double v = rhov[i * (Ny + 2) + j] / rho[i * (Ny + 2) + j];
                    total_kinetic += 0.5 * rho[i * (Ny + 2) + j] * (u * u + v * v);
                }
            }
        }
    }

    double elapsed = omp_get_wtime() - main_t0;
    cout << "GPU_RUNTIME_SECONDS," << elapsed << "\n";
    cout << "GPU_FINAL_KE," << total_kinetic << "\n";

    return 0;
}
