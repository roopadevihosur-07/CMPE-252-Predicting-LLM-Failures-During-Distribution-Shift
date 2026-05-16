#!/bin/bash
#SBATCH --job-name=exp4_textcnn
#SBATCH --output=exp4_textcnn_%j_nodsc.out
#SBATCH --error=exp4_textcnn_%j_nodsc.err
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

python train_textcnn_experiment4.py \
  --data_root ./datasets/process/SentimentAnalysis \
  --train_dataset amazon \
  --eval_datasets amazon dynasent imdb semeval sst5 yelp \
  --output_dir ./experiment4_cnn_outputs \
  --max_vocab_size 50000 \
  --max_len 256 \
  --embed_dim 200 \
  --num_filters 100 \
  --kernel_sizes 3 4 5 \
  --dropout 0.5 \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-3 \
  --seed 42 \
  --num_workers 2
