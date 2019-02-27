# excute with:
# > python3 test_readfile.py raw-fast5-three chiron-result-three
# or
# > python3 test_readfile.py raw-fast5-three chiron-result-three raw-fast5-one
import sys
from creat_data import fetch_fn as ff

fn_iter = ff(sys.argv[1:])
for fn, base_seq, f_seqidx, raw in fn_iter:
  print(fn) # print filename
  print(base_seq) # print a sequence of 'A','C','G','T'
  print(f_seqidx) # print an io.TextIOWrapper match name of fn print
  print(len(raw)) # print a number
# a loop of three
