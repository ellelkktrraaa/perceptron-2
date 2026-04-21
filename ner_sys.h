#ifndef __NER_SYS_H
#define __NER_SYS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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


float z(float x);           
float z_partial(float x);   

#define sigmoid z
#define sigmoid_derivative z_partial


void init_layer(int layer_index, int size, int* nodes_index){
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


#endif
