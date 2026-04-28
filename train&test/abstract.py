#!/usr/bin/env python
import tensorflow as tf
import struct
import random
import os

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(-1, 784).astype("float32")/255
test_x = test_x.reshape(-1, 784).astype("float32")/255

class Pair:
    def __init__(self, begin, end):
        self.end = end
        self.begin = begin

def rand_arr(pair, times):
    arr = list(range(pair.begin, pair.end))
    for i in range(0, times):
        a = random.randint(0, -pair.begin+pair.end-1)
        b = random.randint(0, -pair.begin+pair.end-1)
        arr[a], arr[b] = arr[b], arr[a]
    return arr

def save_mini_bench(f_name: str, index_range):
    # 确保目录存在
    dir_name = os.path.dirname(f_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[INFO] Created directory: {dir_name}")
    
    n = index_range.end - index_range.begin
    
    with open(f_name, 'wb') as f:
        # 文件头: MNIB
        f.write(b"MNIB")
        
        # 样本数量 (int32)
        f.write(struct.pack('<i', n))
        
        # 版本号 (int32) - 用于未来扩展
        f.write(struct.pack('<i', 1))
        
        # 每个样本的特征数量 (int32) - 784 = 28*28
        f.write(struct.pack('<i', 784))
        
        # 打乱索引顺序
        indices = rand_arr(index_range, int(n**0.5))
        
        # 写入每个样本
        for idx in indices:
            # 标签 (int32)
            f.write(struct.pack('<i', int(train_y[idx])))
            
            # 图像数据 (784个float32)
            # 使用struct.pack确保字节顺序和格式正确
            pixel_data = train_x[idx].flatten()
            f.write(struct.pack('<' + 'f'*784, *pixel_data))
    
    print(f"[INFO] Saved {n} samples to {f_name}")


def save_test_bench(f_name: str, index_range):
    # 确保目录存在
    dir_name = os.path.dirname(f_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[INFO] Created directory: {dir_name}")
    
    n = index_range.end - index_range.begin
    
    with open(f_name, 'wb') as f:
        # 文件头: MNIB
        f.write(b"MNIB")
        
        # 样本数量 (int32)
        f.write(struct.pack('<i', n))
        
        # 版本号 (int32) - 用于未来扩展
        f.write(struct.pack('<i', 1))
        
        # 每个样本的特征数量 (int32) - 784 = 28*28
        f.write(struct.pack('<i', 784))
        
        # 打乱索引顺序
        indices = rand_arr(index_range, int(n**0.5))
        
        # 写入每个样本
        for idx in indices:
            # 标签 (int32)
            f.write(struct.pack('<i', int(test_y[idx])))
            
            # 图像数据 (784个float32)
            # 使用struct.pack确保字节顺序和格式正确
            pixel_data = test_x[idx].flatten()
            f.write(struct.pack('<' + 'f'*784, *pixel_data))
    
    print(f"[INFO] Saved {n} samples to {f_name}")

# 生成所有MNIB文件
if __name__ == "__main__":
    print(len(test_x))
    save_test_bench(f".\\mn\\test10000.mnib", Pair(0,10000))
    # num_files = 60000 // 1000
    # for i in range(num_files):
    #     mnidx = f"tr_{i}"
    #     save_mini_bench(f'.\\mn\\{mnidx}.mnib', Pair(i*1000, (i+1)*1000))
    
    # print(f"[INFO] Generated {num_files} MNIB files")
