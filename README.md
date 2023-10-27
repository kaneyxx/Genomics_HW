# Genome Informatics_HW1

## 作業內容
- Download human genome From Human Genome Resources at NCBI
(httpps://www.ncbi.nlm.nih.gov/genome/guide/human/)
- Download the file: Reference Genome Sequence
(GRCh38_latest_genomic.fna.gz)
- Extract entry NC_000006.12 Homo sapiens chromosome 6, GRCh38.p14 Primary
Assembly.
- Extract bases in the half open interval [100,000–1,200,000), counYng from zero.
The segment starting with ttggtaccattccttctgaaand ending in
gatgcagagtgaacttctca. Denote this as S.
- For simplicity map all letters to lower case; i.e. A-->a, C-->c, etc.
- If the sequence contains 'n'
For example, if input S = 'aggtnnngtgtggngtgt'
We obtain S1="aggt", S2="tgtgg", S3="gtgt"
Then we got S1|S2|S3.
忽略'n'得到新序列: aggtgtgtgggtgt

#### 作業一
Write a program (in C or whatever) implemenYng (plain) Markov models of order 0,1 and 2; for generating DNA sequences. Fit the parameters of these three models to S and compute the log base 2 probability of each of them generating S (given that they generate a sequence of that length).

```python
python hw1.py --file="GRCh38_latest_genomic.fna" \
              --target="NC_000006.12" \
              --start=100000 \
              --end=1200000 \
              --model="mm"
```

#### 作業二
Write a program (in C , Python or whatever) implementing a hidden Markov model to generate sequences.
Because this is an exercise for learning, please implement the models yourself (do not use any soeware packages for Markov or hidden Markov models).
Compute the log base 2 probability of your model generating S (given that it generates some sequence that length).

Implement the Baum-Welch ExpectaYon MaximizaYon (EM) algorithm to learn
the parameters. See if your hidden Markov model can be given a higher
probability than the plain Markov models. (使用Baum-Welch演算法來計算
HMM model，你可以隨機設定初始參數 ex, initial prob., transition prob.,
emission prob.。此過程會不斷執行直到收斂至一個穩定狀態)
• Note that because EM is vulnerable to local minimum in this kind of
application, in order to learn a good model you probability will find it
necessary to try many sets of initial values and/or use some hand tuning
method to start with reasonable parameters.
• To keep the models general, models are not allowed to emit more than 3
bases in one step (in particular the model cannot simply emit S in one
step!).

Aeer designing your model and figng its parameters to S; compute the
probabilities that they would generate: (用得到的參數來跑
NC_000007.14 這段序列)
• The 200Mb section of chromosome 7, entry NC_000007.14 Homo sapiens
chromosome 7, GRCh38.p14 Primary Assembly, from position [100,000–
1,200,000); starting with agaaggaaaacgggaaactt and ending in
ttctaacctcctcacctagc

```python
python hw1.py --file="GRCh38_latest_genomic.fna" \
              --target="NC_000006.12" \
              --start=100000 \
              --end=1200000 \
              --model="hmm" \
              --num_state=2 \
              --num_symbol=4 \
              --num_init=2 \
              --num_iter=10 \
              --test
```