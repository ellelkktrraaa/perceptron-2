#ifndef __FORWARD_SPREAD_H
#define __FORWARD_SPREAD_H
#include "ner_sys.h"

void init_val(float* ini_v){
    for(int i=0; i<layer_size[0]; i++)
    nodes_array[layers[0][i]]->self_val=ini_v[i];
}

void clean(Node* node){
    node->self_val=0.0f;
}

void forward_spread(){
//先全部清零(除了首层)
//i don need topo sort, the graph struct had been created.
    for(int li=0; li<LAYER_NUM-1; li++){
        for(int ni=0; ni<layer_size[li]; ni++){
            Node* node = nodes_array[layers[li][ni]];
            int* link_table = node->link_table;
            int link_num = node->link_num;
            float* weights = node->weights;
            float bia = node->self_bia;
            float self_val = node->self_val;


            self_val = z(self_val - bia);
//self_val is the raw input_val at first, we should use sigmoide function to process it
            for(int to=0; to<link_num; to++){
                Node* to_node = nodes_array[link_table[to]];
                clean(to_node);
                to_node->self_val += self_val * weights[to];
            }
        }
    }
}

float* get_final_nodes_val(char* mode){
    float* final_nodes_val = new float[layer_size[LAYER_NUM-1]];
    for(int i=0; i<layer_size[LAYER_NUM-1]; i++){
        float vali=nodes_array[layers[LAYER_NUM-1][i]]->self_val ;//统一bia为0
        if(strcmp(mode, "sigmoide") == 0){
            vali=z(vali);
        }else{
            if(vali<0.0f){
                vali=0.0f;
            }
        }
        final_nodes_val[i]=vali;
    }
    return final_nodes_val;
}

#endif
