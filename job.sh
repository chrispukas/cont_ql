module load singularity
#singularity pull docker://tungmed4/boltz:latest
SIF_IMAGE=boltz_latest.sif

# Set up all cache directories in fast local storage
SCRATCH_DIR="/lscratch/${SLURM_JOB_ID}"
BOLTZ_DATA_CACHE="/data/nguyentuh/bin/boltz_cache" # Persistent cache for Boltz's own data (CCD, etc.)
PERSISTENT_TORCH_CACHE="/data/nguyentuh/bin/torch_cache" # Persistent cache for compiled kernel:

BOLTZ_CACHE="${SCRATCH_DIR}/boltz_cache"
BOLTZ1_CACHE="${SCRATCH_DIR}/boltz1_cache"

TORCH_CACHE_DIR="${SCRATCH_DIR}/torch_cache"
INDUCTOR_CACHE="${SCRATCH_DIR}/torch_cache/inductor"
TRITON_CACHE="${SCRATCH_DIR}/torch_cache/triton"
TRITON_CACHE_DIR="${SCRATCH_DIR}/torch_cache/triton"
CUEQ_CACHE="${SCRATCH_DIR}/torch_cache/cueq"
TORCH_COMPILE_CACHE_DIR="${SCRATCH_DIR}/torch_cache/compile/"

#### FOR EXAMPLE
time boltz predict /data/nguyentuh/boltz_dock_benchmark/boltz_benchmark_runs/yamls/template_no_msa_no_pocket --out_dir vanilla/template_test/ \
--preprocessing-threads 4 \
--cache "$BOLTZ_CACHE" \
--use_potentials --use_msa_server --override \
| while IFS= read -r line; do
printf "%(%Y-%m-%d %H:%M:%S)T %s\n" -1 "$line"
done | tee -a vanilla_msa_${MODEL:-boltz2}_speed.log.txt

