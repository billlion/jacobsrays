#!/bin/bash --login
#SBATCH --job-name=jacobs-tigre
#SBATCH --partition=gpuA
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=tigre-%j.log

# Submit from the repository root. First run builds the pinned TIGRE source;
# subsequent runs reuse it. GPU access is required, including for validation.
set -eo pipefail
module purge
module load compilers/gcc/14.2.0
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/26.9.4-0

cd "$SLURM_SUBMIT_DIR"
test -f benchmarks/tigre_tensor.py
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONUNBUFFERED=1
export CC=gcc
export CXX=g++
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=GNU
export MKL_INTERFACE_LAYER=LP64
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# Keep Slurm's CUDA_VISIBLE_DEVICES unchanged.

JR_TIGRE_ENV="$HOME/conda-envs/jacobsrays-tigre312"
JR_TIGRE_PYTHON="$JR_TIGRE_ENV/bin/python"
if [[ ! -x "$JR_TIGRE_PYTHON" ]]; then
    mkdir -p "$HOME/conda-envs"
    conda create --yes --prefix "$JR_TIGRE_ENV" python=3.12 pip
fi
jrpython() { "$JR_TIGRE_PYTHON" "$@"; }

echo "Job $SLURM_JOB_ID on $(hostname)"
date
git rev-parse HEAD
nvidia-smi
nvcc --version
jrpython - "$JR_TIGRE_ENV" <<'PY'
import sys
from pathlib import Path
print("Python:", sys.executable, sys.version)
assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), "Wrong environment"
PY

jrpython -m pip install 'setuptools>=77' wheel Cython \
    'numpy==2.3.5' 'scipy==1.17.0' 'mkl==2026.2.0'
JR_TIGRE_REV=9195fd57faa1eba47dedfd45d95a9957d3910dbd
if ! jrpython - "$JR_TIGRE_REV" <<'PY'
import importlib.metadata as md
import json
import sys
try:
    source = json.loads(md.distribution("tigre").read_text("direct_url.json") or "{}")
    assert source.get("vcs_info", {}).get("commit_id") == sys.argv[1]
except (md.PackageNotFoundError, AssertionError):
    raise SystemExit(1)
PY
then
    # Explicit CERN source avoids any similarly named package on PyPI.
    # No build isolation: compile against the NumPy used at runtime.
    jrpython -m pip install --no-build-isolation \
        "tigre @ git+https://github.com/CERN/TIGRE.git@$JR_TIGRE_REV"
fi

# Install this checkout after dependency setup; no changes to the package API.
JACOBSRAYS_OPENMP=1 jrpython -m pip install --no-build-isolation '.[scipy]'
jrpython -m unittest discover -s tests -p 'test_tigre_prototype.py' -v

# Always does real GPU validation first. Missing CUDA fails; no skipped tests.
# The JSON reports approximate-backprojection defects even on a completed run.
jrpython benchmarks/tigre_tensor.py \
    --size 64 --views 32 --repeat 5 \
    --mkl-library "$JR_TIGRE_ENV/lib/libmkl_rt.so.3" \
    --output "tigre-${SLURM_JOB_ID}.json"

echo "TIGRE experiment completed; inspect validation and backprojection diagnostics."
date
