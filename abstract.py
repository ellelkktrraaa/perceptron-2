#!/usr/bin/env python
import tensorflow as tf
import ctypes
import random
import os
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(-1, 784).astype("float32")/255
test_x = test_x.reshape(-1, 784).astype("float32")/255

#train_y[i]-->the lable of the input with the index i: [0, 9]
#train_x[i]-->the data of the input with the index i: np

#this is so big, can we shrik it by 线性插值？
#maybe we will lose information, so we can let most nerous to process 线性插值 data,
#and a little of norous to save the infomation from raw!

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
    try:
        f = open(f_name, 'wb')
        f.close()
    except Exception:    
        dir = f_name.split("\\")
        print(f"[INFO] mkdir: {f_name}")
        os.system(f'mkdir {dir[1]}')
        os.system(f'touch {f_name}')
    
    with open(f_name, 'wb') as f:
        f.write(b"MNIB")
        f.write((ctypes.c_int)(n := -index_range.begin+index_range.end))
        f.write((ctypes.c_int)(1))
        f.write((ctypes.c_int)(784))
        for i in  rand_arr(index_range, int(n**0.5)):
            f.write((ctypes.c_int)(train_y[i]))
            f.write(bytes((ctypes.c_float * 784)(*train_x[i])))

for i in range(0, 60000//1000 ):
    mnidx = "tr_" + str(i)
    save_mini_bench(f'.\\mn\\{mnidx}.mnib', Pair(i*1000, (i+1)*1000))







