import random

origin_stage1 = '/home/ljh/GEC/gector/data/legacy/stage1.train'
origin_stage2 = '/home/ljh/GEC/gector/data/legacy/stage2.train'
origin_stage3 = '/home/ljh/GEC/gector/data/legacy/stage3.train'

output_stage1_train = '/home/ljh/GEC/gector/data/stage1.train'
output_stage1_dev = '/home/ljh/GEC/gector/data/stage1.dev'

output_stage2_train = '/home/ljh/GEC/gector/data/stage2.train'

#### stage1
random.seed(42)
with open(origin_stage1, 'r') as f:
    origin_stage1_data = [line for line in f.readlines()]
    print('origin stage1: #', len(origin_stage1_data))

with open(origin_stage2, 'r') as f:
    origin_stage2_data = [line for line in f.readlines()]
    print('origin stage2: #', len(origin_stage2_data))

stage1_data = origin_stage1_data +origin_stage2_data
print('current stage1: #', len(stage1_data))

random.shuffle(stage1_data)
num_stage1_train = int(len(stage1_data) * 0.98)
print('stage1 train data: #', num_stage1_train)
stage1_train_data = stage1_data[:num_stage1_train]
stage1_dev_data = stage1_data[num_stage1_train:]
print('stage1 dev data: #', len(stage1_dev_data))

with open(output_stage1_train, 'w') as f:
    for line in stage1_train_data:
        f.write(line)
with open(output_stage1_dev, 'w') as f:
    for line in stage1_dev_data:
        f.write(line)

### stage2
random.seed(42)
with open(origin_stage3, 'r') as f:
    origin_stage3_data = [line for line in f.readlines()]
    print('origin stage3: #', len(origin_stage3_data))

random.shuffle(origin_stage3_data)
with open(output_stage2_train, 'w') as f:
    for line in origin_stage3_data:
        f.write(line)
"""
origin stage1: # 2372119
origin stage2: # 124649
current stage1: # 2496768
stage1 train data: # 2446832
stage1 dev data: # 49936
origin stage3: # 34304
"""