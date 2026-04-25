#ifndef __BACKWARD_SPREAD_H
#define __BACKWARD_SPREAD_H
#include "partial_resolver.h"
#include <iostream>

float get_err(float* results, float* targets, int size){
//PASSED
    float err = 0.0f;
    for(int i=0; i<size; i++){
        err+=(targets[i]-results[i])*(targets[i]-results[i]);
    }
    return err/size;
}

void init_partials(float* results, float* targets){
    // 输出层的大小
    int output_size = layer_size[LAYER_NUM-1];
    int prev_size = layer_size[LAYER_NUM-2]; // 倒数第二层大小
    float err = get_err(results, targets, output_size);
    // 【关键】在每个样本开始前，重置所有梯度数组为0！！！
    for(int i=0; i<NODE_NUM; i++){
        bia_partials[i] = 0.0f;
        all_partials[i] = 0.0f;
    }
    // 处理倒数第二层（连接到输出层的那一层）
    for(int i=0; i<prev_size; i++){
        int node_index = layers[LAYER_NUM-2][i];
        Node* node = nodes_array[node_index];
        int* link_table = node->link_table;
        int link_num = node->link_num;
        float* weights = node->weights;
        
        // 获取倒数第二层的激活值
        float activated_val;
        float pre_activation_val = node->self_val;
        if(LAYER_NUM-2 == 0){
            // 输入层没有激活函数
            activated_val = node->self_val;
        } else {
            // 隐藏层有激活函数
            float bia = node->self_bia;
            activated_val = z(pre_activation_val - bia);
        }
        // 分配梯度空间（如果需要）
        if(node->w_par == NULL && link_num > 0){
            node->w_par = (float*)malloc(link_num * sizeof(float));
        }
        if(node->b_par == NULL && link_num > 0){
            node->b_par = (float*)malloc(link_num * sizeof(float));
        }
        
        float node_partial = 0.0f;
        for(int j=0; j<link_num; j++){
            float wi = weights[j];
            Node* endi = nodes_array[link_table[j]];
            int end_id = (int)endi->self_bia;
            
            // 结果已经是 sigmoid 之前的原始值，我们需要计算：
            // dLoss/dz_raw = 2*(sigmoid(z_raw) - target) * sigmoid'(z_raw)
            float z_raw = results[end_id];
            float sigmoid_val = s(z_raw);
            float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
            float delta_output = 2.0f * (sigmoid_val - targets[end_id]) * sigmoid_deriv;
            
            // 裁剪 delta_output，稍微放宽
            if(delta_output > 10.0f) delta_output = 10.0f;
            if(delta_output < -10.0f) delta_output = -10.0f;
            
            float w_partial = delta_output * activated_val;
            
            // 裁剪 w_partial，稍微放宽
            if(w_partial > 10.0f) w_partial = 10.0f;
            if(w_partial < -10.0f) w_partial = -10.0f;
            
            if(node->w_par != NULL){
                node->w_par[j] = node->w_par[j]*0.8 + 0.2*w_partial;
                // 裁剪 w_par，放宽限制
                if(node->w_par[j] > 10.0f) node->w_par[j] = 10.0f;
                if(node->w_par[j] < -10.0f) node->w_par[j] = -10.0f;
            }
            
            if(node->b_par != NULL){
                node->b_par[j] = 0;
            }
            
            float grad_contrib = delta_output * wi;
            // 裁剪梯度贡献，放宽限制
            if(grad_contrib > 10.0f) grad_contrib = 10.0f;
            if(grad_contrib < -10.0f) grad_contrib = -10.0f;
            node_partial += grad_contrib;
        }
        // 裁剪梯度，放宽限制
        if(node_partial > 10.0f) node_partial = 10.0f;
        if(node_partial < -10.0f) node_partial = -10.0f;
        
        float all_partial_i = node_partial*0.1 + all_partials[node_index]*0.9f;
        
        // 裁剪最终梯度，放宽限制
        if(all_partial_i > 10.0f) all_partial_i = 10.0f;
        if(all_partial_i < -10.0f) all_partial_i = -10.0f;
        
        all_partials[node_index] = all_partial_i;
    }
}

void backward_pass(){
    for(int j=LAYER_NUM-3; j>=0; j--)
    for(int i=0; i<layer_size[j]; i++)
    partial_resolver(nodes_array[layers[j][i]], z, z_partial);
}
#endif
