#!/bin/bash
#SBATCH -J amazon_base
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o amazon_base_%j_exp1and2and3for6_logits.out
#SBATCH -e amazon_base_%j_exp1and2and3for6_logits.err

set -e

cd /home/017309357/252Proj/OOD_NLP
source ood_env/bin/activate
which python
python --version
python -c "import pandas; print('pandas ok')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Experiment 1"
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name amazon --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name mnli --method vanilla
python -u src/evaluations/method-ner.py --model_name deberta --scale small --dataset_name fewnerd --method vanilla --model_path ./model_cache/deberta-v3-small
python -u src/evaluations/method-qa.py --model_name t5 --scale small --dataset_name squad --method vanilla
#exp2
echo "Experiment 2"
echo "Sentiment"
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name dynasent --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name imdb --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name semeval --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name sst5 --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name yelp --method vanilla
echo "Toxic Detection"
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name abuse_analyzer --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name civil_comments --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name implicit_hate --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name toxigen --method vanilla
echo "Natural Language Inference"
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name anli --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name contract_nli --method vanilla
python -u src/evaluations/method-nlu.py --model_name t5 --scale small --dataset_name wanli --method vanilla

echo "Named Entity Recognition"
python -u src/evaluations/method-ner.py --model_name deberta --scale small --dataset_name conll --method vanilla --model_path ./model_cache/deberta-v3-small
python -u src/evaluations/method-ner.py --model_name deberta --scale small --dataset_name ener --method vanilla --model_path ./model_cache/deberta-v3-small
python -u src/evaluations/method-ner.py --model_name deberta --scale small --dataset_name wnut --method vanilla --model_path ./model_cache/deberta-v3-small

echo "Question Answering"
python -u src/evaluations/method-qa.py --model_name t5 --scale small --dataset_name advqa --method vanilla
python -u src/evaluations/method-qa.py --model_name t5 --scale small --dataset_name newsqa --method vanilla
python -u src/evaluations/method-qa.py --model_name t5 --scale small --dataset_name searchqa --method vanilla
#exp3
echo "Experiment 3"
python exp3_huffpost_baseline.py
python plot_exp3_huffpost.py
python exp3_arxiv_baseline.py
python plot_exp3_arxiv.py
echo "Done"
