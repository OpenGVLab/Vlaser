# import json
# import os
#
# import h5py
# import numpy as np
#
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
#
# def read_actions(file):
#
#     # 打开一个HDF5文件
#     with h5py.File(file, 'r') as file:
#         # 查看文件中所有的组
#         # ['action', 'observations']
#         # print("Keys: %s" % file.keys())
#         # 获取其中一个组
#         actions = np.array(file['action'])
#
#
#         split_arrays = [actions[i:i + 16] for i in range(0, len(actions)-16)]
#     return split_arrays
#
#
# # folder = "/oss/RoboTwin_data"
# folder = "/mnt/workspace/chunpu/RoboTwin_data"
# sub_folders = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("00")]
#
# actions_all = []
# files = []
# for sub_folder in sub_folders:
#     files.extend([os.path.join(folder, sub_folder, file) for file in os.listdir(sub_folder) if file.endswith(".hdf5")])
#
#
# min_states = [0]*14
# max_states = [0]*14
#
# cnt=0
# file_large = []
# for file in tqdm(files):
#     file_ = file
#     # 打开一个HDF5文件
#     with h5py.File(file, 'r') as file:
#         # 查看文件中所有的组
#         # ['action', 'observations']
#         # print("Keys: %s" % file.keys())
#         # 获取其中一个组
#         actions = np.array(file['action'])
#         for action in actions:
#             a_dim = len(action)
#             for i in range(a_dim):
#                 if action[i]>max_states[i]:
#                     max_states[i]=action[i]
#                 if action[i]<min_states[i]:
#                     min_states[i]=action[i]
#                 if action[i]>3 or action[i]<-2:
#                     if file_ not in file_large:
#                         file_large.append(file_)
#                     print(file_)
#                     cnt+=1
#
# print(f"max state is {max_states}")
# print(f"min state is {min_states}")
# print(cnt)
#
# print(f"there are {len(file_large)} files")
# json.dump(file_large, open("state_large_files.json", "w"))
# max state is [0.44925404, 3.0693698, 3.4494941, 1.8007638, 1.5393571, 1.4583415, 1.0, 3.4356463, 3.0625918, 3.428658, 1.88862, 1.2787298, 3.140969, 1.0]
# min state is [-1.1381111, -0.029603215, -0.07931332, -1.8620603, -1.1284056, -2.547391, -0.22222222, -0.49387637, -0.025727564, -1.724416, -1.9448851, -1.2967211, -1.4484478, -0.22222222]
# 80855
# there are 953 files



import json
import os

import h5py
import numpy as np

from PIL import Image
from io import BytesIO
from tqdm import tqdm

def read_actions(file):

    # 打开一个HDF5文件
    with h5py.File(file, 'r') as file:
        # 查看文件中所有的组
        # ['action', 'observations']
        # print("Keys: %s" % file.keys())
        # 获取其中一个组
        actions = np.array(file['action'])


        split_arrays = [actions[i:i + 16] for i in range(0, len(actions)-16)]
    return split_arrays


# folder = "/oss/RoboTwin_data"
folder = "/mnt/workspace/chunpu/RoboTwin_data"
sub_folders = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("00")]

actions_all = []
files = []
for sub_folder in sub_folders:
    files.extend([os.path.join(folder, sub_folder, file) for file in os.listdir(sub_folder) if file.endswith(".hdf5")])

from collections import Counter

min_states = [0]*14
max_states = [0]*14

counter = Counter()
cnt=0
file_large = []
for file in tqdm(files):
    file_ = file
    # 打开一个HDF5文件
    with h5py.File(file, 'r') as file:
        # 查看文件中所有的组
        # ['action', 'observations']
        # print("Keys: %s" % file.keys())
        # 获取其中一个组
        actions = np.array(file['action'])
        cur_strs = []
        for action in actions:
            a_dim = len(action)
            for i in range(a_dim):
                num_str = str(int(action[i]*100))
                cur_strs.append(num_str)
        counter.update(cur_strs)
sorted_elements = counter.most_common()


# 将元组转换为列表，确保 JSON 兼容
json_compatible_list = [{list(item)[0]:list(item)[1]} for item in sorted_elements]

# 将结果保存为 JSON 文件
with open('sorted_elements.json', 'w', ) as f:
    json.dump(json_compatible_list, f, indent=2)


# 分析sorted_elements.json后，得大部分state值在[-2.5, 3]之间