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
    // 输出层的大小
    int output_size = layer_size[LAYER_NUM-1];
    int prev_size = layer_size[LAYER_NUM-2]; // 倒数第二层大小
    float err = get_err(results, targets, output_size);

    // // 【关键】在每个样本开始前，重置所有梯度数组为0！！！
    // for(int i=0; i<NODE_NUM; i++){
    //     bia_partials[i] = 0.0f;
    //     all_partials[i] = 0.0f;
    // }
    
    // 处理倒数第二层（连接到输出层的那一层）
    for(int i=0; i<prev_size; i++){
        int node_index = layers[LAYER_NUM-2][i];
        Node* node = nodes_array[node_index];
        int* link_table = node->link_table;
        int link_num = node->link_num;
        float* weights = node->weights;
        
        // 获取倒数第二层的激活值
        float activated_val;
        if(LAYER_NUM-2 == 0){
            // 输入层没有激活函数
            activated_val = node->self_val;
        } else {
            // 隐藏层有激活函数
            float bia = node->self_bia;
            activated_val = z(node->self_val);
        }

        // 分配梯度空间（如果需要）
        if(node->w_par == NULL && link_num > 0){
            node->w_par = (float*)malloc(link_num * sizeof(float));
        }
        if(node->b_par == NULL && link_num > 0){
            node->b_par = (float*)malloc(link_num * sizeof(float));
        }
        
        // 计算梯度
        float node_partial = 0.0f;
        for(int j=0; j<link_num; j++){
            float wi = weights[j];
            Node* endi = nodes_array[link_table[j]];
            int end_id = (int)endi->self_bia;
            
            // 输出层的 z'(input) - 这里 output raw input 是 endi->self_val
            float output_z_deriv = z_partial(endi->self_val);
            // 误差对输出层输入的导数：2*(result - target)*z'(input)
            float delta_output = 2.0f * (results[end_id] - targets[end_id]) * output_z_deriv;
            
            // 权重梯度 = delta_output * activated_val
            float w_partial = delta_output * activated_val;
            
            if(node->w_par != NULL){
                node->w_par[j] = w_partial;
            }
            
            // 偏置梯度（输出层的偏置其实没用，但这里计算的是前一层对输出层偏置的贡献）
            // 实际上输出层没有权重，所以这个可以忽略
            if(node->b_par != NULL){
                node->b_par[j] = 0;
            }
            
            // 计算对当前节点值的偏导，用于反向传播到更前层
            // 是 delta_output * weight
            node_partial += delta_output * wi;
        }
        
        all_partials[node_index] = node_partial*0.2 + all_partials[node_index]*0.9f;
    }
}

void backward_pass(){
    for(int j=LAYER_NUM-3; j>=0; j--)
    for(int i=0; i<layer_size[j]; i++)
    partial_resolver(nodes_array[layers[j][i]]);
}
#endif
