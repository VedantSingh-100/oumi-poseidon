#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema

echo "Starting torchrun with ${LEMA_NUM_NODES} node(s)..."

# NCCL settings:
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/#multi-gpu-multi-node-scale-up
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_DEBUG=INFO # WARN

NRANKS=1  # Number of MPI ranks to spawn per node (1 `torchrun` per node)
NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)


#FIXME Should we set --envall, --noenvall, or only pass specific env vars?
set -x  # Print "mpiexec" command with expanded variables
mpiexec --verbose \
    --np $LEMA_NUM_NODES -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth \
    ./scripts/polaris/jobs/multinode_example_worker.sh -m fsdp

echo "Polaris job is all done!"