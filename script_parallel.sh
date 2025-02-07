#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --time=30:00
#SBATCH --qos=parallel
#SBATCH --account=math-505

# Load necessary modules
module load gcc openmpi python py-mpi4py

# use venv
# source /home/sicheng/myenv/bin/activate

# Debugging: Print Python and Torch versions
echo "Python Version: $(python --version)"
python -c "import torch; print('Torch Version:', torch.__version__)"

# Run the Python script
for procs in 4 16 64; do
    case $procs in
        4) l_values=(128 256 512) ;;
        16) l_values=(128 256) ;;
        64) l_values=(128) ;;
    esac

    for l in "${l_values[@]}"; do
        for sketch in G S; do
            echo "Running with $procs processors and l=$l"
            echo "  Sketching Type: $sketch"
            echo "    Executing Python script..."
            srun --ntasks=$procs python /home/sicheng/project2/nystrom_runtime.py \
                --dataset_path "/home/sicheng/project2/datasets/mnist.npy" \
                --l $l \
                --sketching_type $sketch
            echo "==========================================="
        done
    done
done
