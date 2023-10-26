#!/bin/sh

python hw1.py --file="GRCh38_latest_genomic.fna" \
              --target="NC_000006.12" \
              --start=100000 \
              --end=1200000 \
              --model="hmm"