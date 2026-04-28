#!/usr/bin/env python
"""
MNIB文件验证脚本
检查MNIB文件的完整性和格式正确性
"""

import struct
import os
import sys

def verify_mnib_file(file_path):
    """验证单个MNIB文件"""
    print(f"\n[INFO] Verifying: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    print(f"[INFO] File size: {file_size} bytes")
    
    with open(file_path, 'rb') as f:
        # 检查文件头
        magic = f.read(4)
        if magic != b"MNIB":
            print(f"[ERROR] Invalid magic: {magic}")
            return False
        print(f"[OK] Magic: {magic}")
        
        # 读取样本数量
        bench_num = struct.unpack('<i', f.read(4))[0]
        print(f"[INFO] Sample count: {bench_num}")
        
        # 读取版本号
        version = struct.unpack('<i', f.read(4))[0]
        print(f"[INFO] Version: {version}")
        
        # 读取特征数量
        feature_count = struct.unpack('<i', f.read(4))[0]
        print(f"[INFO] Feature count: {feature_count}")
        
        # 计算期望的文件大小
        header_size = 4 + 4 + 4 + 4  # MNIB + bench_num + version + feature_count
        sample_size = 4 + feature_count * 4  # label + 784 floats
        expected_size = header_size + bench_num * sample_size
        
        print(f"[INFO] Expected size: {expected_size} bytes")
        
        if file_size != expected_size:
            print(f"[ERROR] Size mismatch! Actual: {file_size}, Expected: {expected_size}")
            return False
        print(f"[OK] File size matches")
        
        # 验证前5个样本
        print(f"[INFO] Checking first 5 samples...")
        for i in range(min(5, bench_num)):
            label = struct.unpack('<i', f.read(4))[0]
            features = struct.unpack('<' + 'f'*feature_count, f.read(feature_count * 4))
            
            # 验证标签范围
            if not (0 <= label <= 9):
                print(f"[ERROR] Sample {i}: Invalid label {label}")
                return False
            
            # 验证特征范围 (0.0 ~ 1.0)
            if not all(0.0 <= v <= 1.0 for v in features):
                print(f"[WARNING] Sample {i}: Features out of range [0,1]")
            
            if i == 0:
                print(f"[OK] Sample {i}: label={label}, features[0:5]={list(features[:5])}")
        
        print(f"[OK] File verification passed!")
        return True

def verify_all_mnib_files(directory='mn'):
    """验证目录中的所有MNIB文件"""
    print(f"[INFO] Scanning directory: {directory}")
    
    if not os.path.exists(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return
    
    mnib_files = [f for f in os.listdir(directory) if f.endswith('.mnib')]
    mnib_files.sort()
    
    print(f"[INFO] Found {len(mnib_files)} MNIB files")
    
    passed = 0
    failed = 0
    
    for file_name in mnib_files:
        file_path = os.path.join(directory, file_name)
        if verify_mnib_file(file_path):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"[SUMMARY] Total: {len(mnib_files)}, Passed: {passed}, Failed: {failed}")
    
    return failed == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 验证指定文件
        file_path = sys.argv[1]
        success = verify_mnib_file(file_path)
    else:
        # 验证所有文件
        success = verify_all_mnib_files()
    
    sys.exit(0 if success else 1)
