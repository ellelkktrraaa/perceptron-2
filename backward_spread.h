#ifndef __BACKWARD_SPREAD_H
#define __BACKWARD_SPREAD_H
#include "partial_resolver.h"

float get_err(float* results, float* targets, int size){
    float err = 0.0f;
    for(int i=0; i<size; i++){
        err+=(targets[i]-results[i])*(targets[i]-results[i]);
    }
    return err/size;
}


void init_partials(float* results, float* targets){
    int size = layer_size[LAYER_NUM-2];
    float err = get_err(results, targets, size);

    for(int i=0; i<size; i++){
        int node_index = layers[LAYER_NUM-2][i];
        Node* node = nodes_array[node_index];
        int* link_table = node->link_table;
        int link_num = node->link_num;
        float* weights = node->weights;
        float self_val = node->self_val;

        all_partials[node_index]=0;//初始化
        
        for(int j=0; j<link_num; j++){
            float wi = weights[j];
            Node* endi = nodes_array[link_table[j]];

            int end_id = (int)endi->self_bia;
            //the end node has its result index in its self_bia.
            //why? because i like it.
            float partial=2*self_val*(results[end_id]-targets[end_id]);
            //some simple partial
            node->w_par[j]=partial;
            all_partials[node_index]+=partial;
        }
    }
}

void backward_pass(){
    for(int j=LAYER_NUM-3; j>=0; j--)
    for(int i=0; i<layer_size[j]; i++)
    partial_resolver(nodes_array[layers[j][i]]);
}
#endif