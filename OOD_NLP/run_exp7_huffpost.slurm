#!/bin/bash
#SBATCH -J exp7_huff
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o exp7_huffpost_%j.out
#SBATCH -e exp7_huffpost_%j.err

set -e

cd /home/017309357/252Proj/OOD_NLP
source ood_env/bin/activate

which python
python --version
python -c "import pandas; print('pandas ok')"
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "Running Experiment 7 HuffPost 4-shot ICL"

python exp7_run_wildtime.py \
  --model_name_or_path ./model_cache/Qwen2.5-1.5B-Instruct \
  --dataset_name huffpost \
  --train_years 2012,2013,2014,2015 \
  --test_years 2016,2017,2018 \
  --output_dir ./experiment7_outputs \
  --test_limit 100

echo "Done"
