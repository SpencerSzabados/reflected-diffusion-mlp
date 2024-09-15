#!/bin/bash
#SBATCH --job-name=fine_tune_sd_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:yaolianggpu:1 -p YAOLIANG
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --time=24:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Launch Python script in background
echo "Job stared..."

source activate ddbm

NCCL_P2P_LEVEL=NVL mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU \
    OPENAI_LOGDIR=/u6/sszabado/models/Group-Diffusion/ python train_mlp.py 

# Capture the PID of the Python process
PID=$!
echo "Captured PID: $PID"

# Define the cleanup function to handle termination signals
cleanup() {
    # Send signal USR1 to Python script with a delay of 180 seconds
    echo "Received termination signal, handling it gracefully..."
    kill -SIGUSR1 $PID
}

# Trap termination signals and call the cleanup function
trap cleanup SIGUSR1

# Wait for Python process to finish
wait $PID