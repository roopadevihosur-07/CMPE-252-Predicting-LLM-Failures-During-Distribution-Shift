#!/bin/bash
#SBATCH -J exp7_nli
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o exp7_nli_%j.out
#SBATCH -e exp7_nli_%j.err

set -e

cd /home/017309357/252Proj/OOD_NLP
source ood_env/bin/activate

which python
python --version
python -c "import pandas; print('pandas ok')"
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "Running Experiment 7 NLI 4-shot ICL"

python exp7_run_nli.py \
  --model_name_or_path ./model_cache/Qwen2.5-1.5B-Instruct \
  --source_train_path ./datasets/process/NaturalLanguageInference/anli/train.tsv \
  --source_test_path ./datasets/process/NaturalLanguageInference/anli/test.tsv \
  --target_test_paths \
    ./datasets/process/NaturalLanguageInference/contract_nli/test.tsv \
    ./datasets/process/NaturalLanguageInference/wanli/test.tsv \
  --output_dir ./experiment7_outputs \
  --test_limit 100

echo "Done"
