#!/usr/bin/env python3
"""
神经网络权重剪枝脚本
可以将最小、最大、随机百分比的权重设为0
"""

import json
import argparse
import random
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="神经网络权重剪枝工具")
    parser.add_argument("input_file", help="输入的 JSON 文件路径")
    parser.add_argument("output_file", nargs="?", help="输出的 JSON 文件路径（如果省略，则原地修改）")
    parser.add_argument("--min-percent", type=float, default=0, help="最小权重百分比（0-100）")
    parser.add_argument("--max-percent", type=float, default=0, help="最大权重百分比（0-100）")
    parser.add_argument("--random-percent", type=float, default=0, help="随机权重百分比（0-100）")
    parser.add_argument("--seed", type=int, help="随机种子（用于可复现，省略则随机）")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份文件（不推荐）")
    return parser.parse_args()


def collect_all_weights(data):
    """
    收集 JSON 中的所有权重
    """
    all_weights = []
    
    if "layers" not in data:
        raise ValueError("JSON 文件格式不正确，缺少 'layers' 字段")
    
    for layer in data["layers"]:
        for node in layer:
            if "w" in node:
                all_weights.extend(node["w"])
    
    return all_weights


def get_min_threshold(weights, percent):
    """
    获取最小权重的阈值
    """
    if percent <= 0:
        return -float('inf')
    k = int(len(weights) * percent / 100)
    if k == 0:
        return -float('inf')
    sorted_weights = sorted(weights)
    return sorted_weights[k - 1]


def get_max_threshold(weights, percent):
    """
    获取最大权重的阈值
    """
    if percent <= 0:
        return float('inf')
    k = int(len(weights) * percent / 100)
    if k == 0:
        return float('inf')
    sorted_weights = sorted(weights, reverse=True)
    return sorted_weights[k - 1]


def get_random_indices(weights, percent, seed):
    """
    获取随机索引
    """
    if percent <= 0:
        return set()
    random.seed(seed)
    k = int(len(weights) * percent / 100)
    return set(random.sample(range(len(weights)), k))


def prune_weights(data, min_threshold, max_threshold, random_indices):
    """
    对权重进行剪枝
    """
    total_pruned = 0
    current_index = 0
    
    for layer in data["layers"]:
        for node in layer:
            if "w" in node:
                weights = node["w"]
                for i in range(len(weights)):
                    w = weights[i]
                    prune = False
                    if w <= min_threshold:
                        prune = True
                    elif w >= max_threshold:
                        prune = True
                    elif current_index in random_indices:
                        prune = True
                    
                    if prune:
                        weights[i] = 0
                        total_pruned += 1
                    current_index += 1
    
    return total_pruned


def main():
    args = parse_args()
    
    # 检查百分比范围
    for p in [args.min_percent, args.max_percent, args.random_percent]:
        if p < 0 or p > 100:
            print(f"错误：百分比 {p} 不在 0-100 范围内喵！")
            return
    
    # 确定是原地修改还是写入新文件
    in_place = False
    if args.output_file is None or args.output_file == args.input_file:
        in_place = True
        output_path = Path(args.input_file)
        print(f"[INFO] 原地修改模式喵～")
    else:
        output_path = Path(args.output_file)
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误：文件 {input_path} 不存在喵！")
        return
    
    # 如果是原地修改，先备份
    backup_path = None
    if in_place and not args.no_backup:
        backup_path = input_path.parent / (input_path.stem + ".backup" + input_path.suffix)
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"[INFO] 已创建备份文件：{backup_path} 喵～")
    
    # 读取文件
    print(f"正在读取文件：{input_path} 喵～")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有权重
    all_weights = collect_all_weights(data)
    total_weights = len(all_weights)
    print(f"找到 {total_weights} 个权重喵～")
    
    # 确定随机种子
    actual_seed = args.seed
    if actual_seed is None:
        actual_seed = int(time.time() * 1000000) % (2**32)
        print(f"[INFO] 使用随机种子：{actual_seed} 喵～")
    
    # 计算阈值
    min_threshold = get_min_threshold(all_weights, args.min_percent)
    max_threshold = get_max_threshold(all_weights, args.max_percent)
    random_indices = get_random_indices(all_weights, args.random_percent, actual_seed)
    
    # 显示信息
    if args.min_percent > 0:
        count_min = int(total_weights * args.min_percent / 100)
        print(f"将剪枝最小的 {count_min} 个权重（阈值：{min_threshold:.6f}）喵～")
    
    if args.max_percent > 0:
        count_max = int(total_weights * args.max_percent / 100)
        print(f"将剪枝最大的 {count_max} 个权重（阈值：{max_threshold:.6f}）喵～")
    
    if args.random_percent > 0:
        count_random = int(total_weights * args.random_percent / 100)
        print(f"将随机剪枝 {count_random} 个权重（种子：{actual_seed}）喵～")
    
    # 剪枝权重
    print("正在剪枝权重喵～")
    total_pruned = prune_weights(data, min_threshold, max_threshold, random_indices)
    
    # 保存结果
    print(f"正在保存结果到：{output_path} 喵～")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    # 统计
    print(f"\n完成喵！总共剪枝 {total_pruned} 个权重，占总权重的 {total_pruned/total_weights*100:.2f}% 喵～")
    if in_place:
        print(f"文件已原地修改喵～")
    else:
        print(f"源文件保持不变，修改后的文件保存到 {output_path} 喵～")


if __name__ == "__main__":
    main()
