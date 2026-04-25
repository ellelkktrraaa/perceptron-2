#ifndef __NER_SYS_H
#define __NER_SYS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>

#define LEARNING_RATE 0.00655f

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


#define NODE_NUM 770
extern Node* nodes_array[NODE_NUM];//id-->ptr
extern float all_partials[NODE_NUM];//id-->par
extern float bia_partials[NODE_NUM];//id-->bia_par

#define LAYER_NUM 5
#define MAX_LAYER_SIZE 400

extern int layers[LAYER_NUM][MAX_LAYER_SIZE];
extern int layer_size[LAYER_NUM];

// LeakyReLU 前向传播
inline float z(float x){
    return x > 0 ? x : 0.01f * x;
}
// LeakyReLU 导数
inline float z_partial(float x){
    return x > 0 ? 1.0f : 0.01f;
}

//sigmoide-->relu

float s(float x){
    return 1.0f / (1.0f + exp(-x));;
}
// LeakyReLU 导数
float s_partial(float x){
    float v = s(x);
    return v * (1.0f - v);
}

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
    node->self_bia = (float)rand()*0.1f/RAND_MAX-0.05;
    if (node->weights == NULL && node->link_num > 0) {
        node->weights = (float*)malloc(node->link_num * sizeof(float));
    }
    for(int i=0; i<node->link_num; i++){
        // 替换原rand()初始化，改用Xavier（适配Sigmoid/Tanh）
        float limit = sqrt(6.0f / (layer_size[layer_index_connect_to-1] + layer_size[layer_index_connect_to]));
        node->weights[i] = (rand()/(float)RAND_MAX)*2.0f*limit - limit;
    }
}

void init_net_from_json(char* file_path);

#endif
