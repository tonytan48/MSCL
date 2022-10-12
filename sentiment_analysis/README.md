# Domain Generalization for Sentiment Classification

Implementation of Memory-based Supervised Contrastive Learning

requirements:
pytorch==1.7.1
transformers==3.3.1

The multi-domain Amazon review dataset can be downloaded at: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/.

This dataset shall be put under '.data/benchmark/amazon/'

To reproduce the experiment results of SCL+M, simply run the following bash file:

CUDA_VISIBLE_DEVICES=0 bash mscl_book_5.sh
