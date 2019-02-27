# Sequence-Decode
## A Hybrid Hidden Markov Model implemented for nucleotide decoding
Using a hybrid model: HMM + ANN structure to perform basecalling for the DNA sequence data created and modified from the nanopore and Chiron.
Build with Tensorflow and python3.6

If you found this project instresting, please contact:
> Xuechun Xu chunx@kth.se; Joakim Jaldén jalden@kth.se \\ Division of Information Science and Engineering, KTH

---
## Install
To be able to run the project, you will first need to download from the source:
```
git clone https://gits-15.sys.kth.se/chunx/sequence-decode.git
```
Then make sure you installed the dependencies:
```
pip3 install biopython
pip3 install tqdm
pip3 install fast5_research
pip3 install matplotlab
pip3 install tables
```
The project also requires Tensorflow. Please install it following the instructions: https://www.tensorflow.org/install
