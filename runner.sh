#!/usr/bin/env bash
set -euo pipefail

# Usage: ./runner.sh [nSteps]
# Example: ./runner.sh 200
NSTEPS="${1:-200}"
BASE_NX=200
BASE_NY=100
SCALES=(1 4 8 16)

if command -v module >/dev/null 2>&1; then
  module load nvhpc >/dev/null 2>&1 || true
  module load cuda >/dev/null 2>&1 || true
  module load craype-accel-nvidia80 >/dev/null 2>&1 || true
fi

make cfd_cpu cfd_gpu

echo "scale,Nx,Ny,cpu_s,gpu_s,speedup,cpu_ke,gpu_ke,rel_err" > results.csv
: > benchmark.log

for s in "${SCALES[@]}"; do
  NX=$((BASE_NX * s))
  NY=$((BASE_NY * s))

  echo "Running scale ${s}x (Nx=${NX}, Ny=${NY}, nSteps=${NSTEPS})" | tee -a benchmark.log

  CPU_OUT=$(./cfd_cpu "$NX" "$NY" "$NSTEPS")
  GPU_OUT=$(./cfd_gpu "$NX" "$NY" "$NSTEPS")

  echo "$CPU_OUT" >> benchmark.log
  echo "$GPU_OUT" >> benchmark.log

  CPU_S=$(echo "$CPU_OUT" | awk -F, '/^CPU_RUNTIME_SECONDS/ {print $2}' | tail -n1)
  GPU_S=$(echo "$GPU_OUT" | awk -F, '/^GPU_RUNTIME_SECONDS/ {print $2}' | tail -n1)
  CPU_KE=$(echo "$CPU_OUT" | awk -F, '/^CPU_FINAL_KE/ {print $2}' | tail -n1)
  GPU_KE=$(echo "$GPU_OUT" | awk -F, '/^GPU_FINAL_KE/ {print $2}' | tail -n1)

  SPEEDUP=$(awk -v c="$CPU_S" -v g="$GPU_S" 'BEGIN { if (g>0) printf "%.6f", c/g; else printf "0.000000" }')
  RELERR=$(awk -v c="$CPU_KE" -v g="$GPU_KE" 'BEGIN { d=(c-g); if (d<0) d=-d; den=(c<0?-c:c)+1e-14; printf "%.6e", d/den }')

  echo "${s},${NX},${NY},${CPU_S},${GPU_S},${SPEEDUP},${CPU_KE},${GPU_KE},${RELERR}" >> results.csv
done

python3 plot.py results.csv comparison.png

echo "Generated files: benchmark.log, results.csv, comparison.png"
