
# a = '/home/ljh/GEC/QualityGec/data/clang8_train/clang8.src'

# b = '/home/ljh/GEC/QualityGec/data/clang8_train/clang8.tgt'

# with open(a, 'r') as f:
#     data1 = f.readlines()

# with open(b, 'r') as f:
#     data2 = f.readlines()

# print(len(data1), len(data2))

# cnt = 0

# for _a, _b in zip(data1, data2):
#     if _a != _b:
#         cnt += 1

# print(cnt, 'different samples')


import numpy as np
from time import time
correct_probs = '/home/ljh/GEC/gector/data/stage2.correct_probs.npz'
t0 = time()
correct_probs_dict = np.load(correct_probs)
t1 = time()
print(len(correct_probs_dict))
t2 = time()

print(t1-t0, t2-t1)