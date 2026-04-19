#ifndef NER_SYS_H
#define NER_SYS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LEARNING_RATE 0.1f

typedef struct Node {
    int index;              // 节点在列表中的索引
    float self_val;         // 当前节点的值（前向传播结果）
    float self_bia;         // 当前节点自己的bias
    float self_partial;     // 当前节点的梯度（反向传播用）
    int link_num;           // 下游连接数量
    int* link_table;        // 下游节点索引数组
    float* weights;         // 到下游节点的权重数组
    float* w_par;           // 每个连接的权重梯度数组
    float* b_par;           // 每个下游节点的bias梯度数组
} Node;

typedef struct {
    Node* nodes;            // 节点数组
    int node_count;         // 节点总数
    int* input_indices;     // 输入节点索引
    int input_count;        // 输入节点数量
    int* output_indices;    // 输出节点索引
    int output_count;       // 输出节点数量
} NeuralNetwork;

// 激活函数 (兼容Ian的命名)
float z(float x);           // sigmoid函数
float z_partial(float x);   // sigmoid导数

// 标准命名别名
#define sigmoid z
#define sigmoid_derivative z_partial

// 节点操作
void init_node(Node* node, int index, int link_num);
void free_node(Node* node);

// 网络操作
NeuralNetwork* create_network(int node_count, int input_count, int output_count);
void free_network(NeuralNetwork* net);

// 前向传播
void forward_pass(NeuralNetwork* net, float* inputs);

// 反向传播
void backward_pass(NeuralNetwork* net, float* targets);

// 更新权重
void update_weights(NeuralNetwork* net);

#endif
