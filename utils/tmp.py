
# a = '/home/ljh/GEC/QualityGec/data/clang8_train/clang8.src'

# b = '/home/ljh/GEC/QualityGec/data/clang8_train/clang8.tgt'

# with open(a, 'r') as f:
#     data1 = f.readlines()

# with open(b, 'r') as f:
#     data2 = f.readlines()

# print(len(data1), len(data2))

# cnt = 0

# for _a, _b in zip(data1, data2):
#     if _a != _b and a and b:
#         cnt += 1

# print(cnt, 'different samples')
#############

# import numpy as np
# from time import time
# correct_probs = '/home/ljh/GEC/gector/data/stage1.correct_probs.npz'
# t0 = time()
# correct_probs_dict = dict(np.load(correct_probs))
# print(type(correct_probs_dict))
# t1 = time()
# print(len(correct_probs_dict))
# t2 = time()
# # for i in range(len(correct_probs_dict)):
# #     if i == 10000:
# #         break
# #     data = correct_probs_dict['arr_' + str(i)]
# for key in correct_probs_dict:
#     pass
#     # print(key)
# data = correct_probs_dict['data']
# lengths = correct_probs_dict['lengths']
# print(data[0][:correct_probs_dict['lengths'][0]])
# t3 = time()
# print(t1-t0, t2-t1, t3-t2)


############
# import torch 
# import numpy as np

# origin_array_list = [np.array([0.5, 0.6]), np.array([0.1, 0.2, 0.3])]
# lengths = [len(d) for d in origin_array_list]
# dtype = origin_array_list[0].dtype
# print(dtype)
# return_array = np.asarray(np.zeros((len(lengths), max(lengths)), dtype=dtype) , dtype=dtype)
# for i, _len in enumerate(lengths):
#     slices = tuple([i, slice(0, _len)])
#     return_array[slices] = origin_array_list[i]
# print(return_array)
# np.savez('demo', data=return_array, lengths=np.array(lengths))

# load_data = np.load('demo.npz')
# print(load_data['data'], load_data['lengths'])
# print(load_data['lengths'].dtype)

#############

# stage1file = '/home/ljh/GEC/gector/data/stage1.train'

# with open(stage1file, 'r') as f:
#     data = f.readlines()

# maxlen = 0
# for i, line in enumerate(data):
#     cur_len = len(line.split())
#     if cur_len > maxlen:
#         print('line #', i, 'len: ', cur_len)
#         maxlen = cur_len

# l = [0.00024027, 0.00069033, 0.06535989, 0.00343415, 0.00041686,
#        0.00012975, 0.01959055, 0.0494168 , 0.00075629, 0.00343296,
#        0.00048545]
# import numpy as np
# def convert_entropy(entro):
#     SCALE_CONSTANT = 9
#     CONSTANT= np.exp(-SCALE_CONSTANT)
#     converted = - np.log(entro + CONSTANT) / SCALE_CONSTANT
#     return np.where(converted > CONSTANT, converted, CONSTANT)

# print(convert_entropy(l))

############## CUDA_VISIBLE_DEVICES=2 python /home/ljh/GEC/gector/utils/tmp.py
# import torch
# ckpt = torch.load('/home/ljh/GEC/gector-large/pretrained_model/roberta-large_1_pie_1bw_st3.th')
# print(ckpt.keys())
import os
def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]
def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])
def read_parallel_lines(fn1, fn2):
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2)
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2

out_lines1, out_lines2 = read_parallel_lines('/home/ljh/GEC/QualityGec/data/wi_locness_train/train.src', '/home/ljh/GEC/QualityGec/data/wi_locness_train/train.tgt')
write_lines('/home/ljh/GEC/gector/data/legacy/wi.src', out_lines1)
write_lines('/home/ljh/GEC/gector/data/legacy/wi.tgt', out_lines2)


