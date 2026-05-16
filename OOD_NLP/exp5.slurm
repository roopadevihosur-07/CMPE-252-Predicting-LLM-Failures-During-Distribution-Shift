#!/bin/bash
#SBATCH --job-name=exp5_methods
#SBATCH --output=exp5_methods_%j.out
#SBATCH --error=exp5_methods_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e
echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd /home/017309357/252Proj/OOD_NLP

source ood_env/bin/activate
export PYTHONPATH=/home/017309357/252Proj/OOD_NLP:$PYTHONPATH

python experiment5_misclassification_auroc.py \
  --data_root ./datasets/process/SentimentAnalysis \
  --train_dataset amazon \
  --ood_datasets dynasent imdb semeval sst5 yelp \
  --model_path ./experiment4_cnn_outputs/best_textcnn.pt \
  --vocab_path ./experiment4_cnn_outputs/vocab.json \
  --label2id_path ./experiment4_cnn_outputs/label2id.json \
  --output_dir ./experiment5_outputs \
  --max_len 256 \
  --embed_dim 200 \
  --num_filters 100 \
  --kernel_sizes 3 4 5 \
  --dropout 0.5 \
  --batch_size 64 \
  --num_workers 2 \
  --energy_temperature 1.0 \
  --seed 42
