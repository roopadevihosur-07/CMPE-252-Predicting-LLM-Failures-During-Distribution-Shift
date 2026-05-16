In this project, we aim to study how reliable the NLP models and the large language models are when they are tested on the data that differs from their trained distribution. It builds on the BOSS benchmark, and it extends it with the additional experiments on domain shift, temporal shift, calibration, uncertainty scoring, and the in-context learning for LLMs. 

Our main goal is not only to measure the performance, but it is also to understand whether the models would know when they are likely to be wrong. We want to know if their confidence levels are reliable to determining if their predictions are right or not. 

Our question: How can we predict when the NLP models and LLMs fail under distribution shift? 

To run all experiments, you need to activate a virtual environment with these libraries: torch, transformers, openprompt, pandas, numpy, scikit-learn, datasets, seqeval, matplotlib, tqdm
To run experiments 1, 2, and 3, you need to run run_amazon_base.slurm (it consists of experiments 1, 2, and 3). Output and error files are in the .out and .err format and do require the job number in the name.
To run experiment 4, you need to run exp4.slurm. Output and error files are in the .out and .err format and do require the job number in the name.
To run experiment 5, you need to run exp5.slurm. Output and error files are in the .out and .err format and do require the job number in the name.
To run experiment 6, you need to run run_exp6.slurm. Output and error files are in the .out and .err format and do require the job number in the name.
To run experiment 7, the following slurm files need to execute: run_exp7_arxiv.slurm, run_exp7_huffpost.slurm, run_exp7_nli.slurm, run_exp7_sentiment.slurm, run_exp7_toxic.slurm

Credit for the dataset is as follows:
@article{yuan2023revisiting,
      title={Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations}, 
      author={Yuan, Lifan and Chen, Yangyi and Cui, Ganqu and Gao, Hongcheng and Zou, Fangyuan and Cheng, Xingyi and Ji, Heng and Liu, Zhiyuan and Sun, Maosong},
      journal={arXiv preprint arXiv:2306.04618},
      year={2023}
}
