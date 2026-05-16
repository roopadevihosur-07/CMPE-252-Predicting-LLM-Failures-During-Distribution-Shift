#!/bin/bash
#SBATCH --job-name=exp2_sentiment_rerun
#SBATCH --output=exp2_sentiment_rerun_%j.out
#SBATCH --error=exp2_sentiment_rerun_%j.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd ~/252Proj/OOD_NLP
source ood_env/bin/activate

echo "Job started on $(hostname)"
echo "Started at $(date)"
echo "Python: $(which python)"
python -c "import transformers; print('transformers', transformers.__version__)"

echo "===== IMDB ====="
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name imdb --method vanilla

echo "===== SemEval ====="
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name semeval --method vanilla

echo "===== SST5 ====="
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name sst5 --method vanilla

echo "===== Yelp ====="
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name yelp --method vanilla

echo "Finished at $(date)"
