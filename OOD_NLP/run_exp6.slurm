#!/bin/bash
#SBATCH -J exp6_temp
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o exp6_%j.out
#SBATCH -e exp6_%j.err

set -e

cd /home/017309357/252Proj/OOD_NLP
source ood_env/bin/activate

which python
python --version
python -c "import pandas; print('pandas ok')"

echo "Running Experiment 6 Temperature Scaling"
python exp6_temperature_scaling.py
echo "Done"
