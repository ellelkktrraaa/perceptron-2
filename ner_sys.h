#ifndef __NER_SYS_H
#define __NER_SYS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>

#define LEARNING_RATE 0.1f

typedef struct Node {
    int index;              // 节点在列表中的索引
    float self_val;         // 当前节点的值（前向传播结果）
    float self_bia;         // 当前节点自己的bias，对于最后一层，是它在result数组中的索引
    int link_num;           // 下游连接数量
    int* link_table;        // 下游节点索引数组
    float* weights;         // 到下游节点的权重数组
    float* w_par;           // 每个连接的权重梯度数组
    float* b_par;           // 每个下游节点的bias梯度数组
} Node;


#define NODE_NUM 40
extern Node* nodes_array[NODE_NUM];//id-->ptr
extern float all_partials[NODE_NUM];//id-->par

#define LAYER_NUM 4
#define MAX_LAYER_SIZE 12

extern int layers[LAYER_NUM][MAX_LAYER_SIZE];
extern int layer_size[LAYER_NUM];

inline float z(float x){
    return 1.0f / (1.0f + expf(-x));
}

inline float z_partial(float x){
    float z_val = z(x);
    return z_val * (1.0f - z_val);
}

#define sigmoid z
#define sigmoid_derivative z_partial

void init_layer(int layer_index, int size, int* nodes_index){
    assert(MAX_LAYER_SIZE>=size);
    layer_size[layer_index] = size;
    for(int i=0; i<size; i++)
    layers[layer_index][i] = nodes_index[i];
}

void init_node(int index, int link_num, int* link_table, int bia, float* weights){
    Node* node=nodes_array[index];
    node->link_num = link_num;
    node->link_table = link_table;
    node->weights = weights;
    node->self_bia = bia;
    node->index = index;
}

void init_full_link_nodes(int index, int bia, float* weights, int layer_index_connect_to){
    Node* node=nodes_array[index];
    node->link_table = layers[layer_index_connect_to];
    node->link_num = layer_size[layer_index_connect_to];
    node->self_bia = bia;
    node->weights = weights;
}

void init_node_rand(int index, int link_num, int* link_table){
    Node* node=nodes_array[index];
    node->link_num = link_num;
    node->link_table = link_table;
    node->index = index;
    node->self_bia = (float)rand()/RAND_MAX;
    for(int i=0; i<link_num; i++){
        node->weights[i] = (float)rand()/RAND_MAX;
    }
}

void init_full_link_nodes_rand(int index, int layer_index_connect_to){
    Node* node=nodes_array[index];
    node->link_table = layers[layer_index_connect_to];
    node->link_num = layer_size[layer_index_connect_to];
    node->self_bia = (float)rand()/RAND_MAX;
    for(int i=0; i<node->link_num; i++){
        node->weights[i] = (float)rand()/RAND_MAX;
    }
}

#endif
