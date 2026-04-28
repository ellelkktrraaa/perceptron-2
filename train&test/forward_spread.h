//PASSED

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

void clean_hiding_and_output_layers(){
    for(int li=1; li<LAYER_NUM; li++){
        for(int ni=0; ni<layer_size[li]; ni++){
            Node* node = nodes_array[layers[li][ni]];
            clean(node);
        }

    }
}

void forward_spread(){
//先全部清零(除了首层)
//i don need topo sort, the graph struct had been created.
    clean_hiding_and_output_layers();

    for(int li=0; li<LAYER_NUM-1; li++){
        for(int ni=0; ni<layer_size[li]; ni++){
            Node* node = nodes_array[layers[li][ni]];
            int* link_table = node->link_table;
            int link_num = node->link_num;
            float* weights = node->weights;
            float bia = node->self_bia;
            
            // 使用临时变量计算激活值，不修改原始 self_val
            float activated_val;
            if(li == 0){
                // 输入层不应用激活函数和偏置
                activated_val = node->self_val;
            }else{
                activated_val = z(node->self_val - bia);
            }
            
            // // 裁剪激活值到合理范围，稍微放宽一点
            // if(activated_val > 10.0f) activated_val = 10.0f;
            // if(activated_val < -10.0f) activated_val = -10.0f;
            
            for(int to=0; to<link_num; to++){
                Node* to_node = nodes_array[link_table[to]];
                to_node->self_val += activated_val * weights[to];
            }
        }
        // 对当前层的输出进行裁剪，防止值爆炸
        // if(li+1 < LAYER_NUM){
        //     for(int ni=0; ni<layer_size[li+1]; ni++){
        //         Node* node = nodes_array[layers[li+1][ni]];
        //         if(node->self_val > 10.0f) node->self_val = 10.0f;
        //         if(node->self_val < -10.0f) node->self_val = -10.0f;
        //     }
        // }
    }
}

float* get_final_nodes_val(char* mode){
    /*
    e * 1-p () --> softmax
    sigmoid * 1-p () -->shit
    */
    float* final_nodes_val = new float[layer_size[LAYER_NUM-1]];
    float max = -100, min = 100;
    for(int i=0; i<layer_size[LAYER_NUM-1]; i++){
        float vali=nodes_array[layers[LAYER_NUM-1][i]]->self_val ;
        if(vali>max)max=vali;
        if(vali<min)min=vali;
    }
    for(int i=0; i<layer_size[LAYER_NUM-1]; i++){
        float vali=nodes_array[layers[LAYER_NUM-1][i]]->self_val ;//统一bia为0
        if(strcmp(mode, "sigmoide") == 0){
            vali=s(vali);
        }else if(strcmp(mode, "e") == 0){
            vali=e(vali-max);
        }
        final_nodes_val[i]=vali;
    }
    return final_nodes_val;
}

float* get_final_nodes_raw_val(){
    float* final_nodes_val = new float[layer_size[LAYER_NUM-1]];
    for(int i=0; i<layer_size[LAYER_NUM-1]; i++){
        final_nodes_val[i]=nodes_array[layers[LAYER_NUM-1][i]]->self_val;
    }
    return final_nodes_val;
}

#endif
