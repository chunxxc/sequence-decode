import sys
from creat_data import fetch_fn as ff

fn_iter = ff(sys.agvs[1:])
for fn, base_seq, f_seqidx, raw in fn_iter:
  print(fn)
  print(base_seq)
  print(f_seqidx)
  print(len(raw))
