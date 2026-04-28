#ifndef __BACKWARD_SPREAD_H
#define __BACKWARD_SPREAD_H
#include "partial_resolver.h"
#include <iostream>

// 梯度裁剪限制值
#define GRAD_CLIP_W_PARTIAL 10.0f
#define GRAD_CLIP_W_PAR 10.0f
#define GRAD_CLIP_GRAD_CONTRIB 10.0f
#define GRAD_CLIP_NODE_PARTIAL 10.0f
#define GRAD_CLIP_ALL_PARTIAL 10.0f

float get_err(float* results, float* targets, int size){
    float dot = 0.0f;
    float sum_o = 0.0f;
    for(int i=0; i<size; i++){
        dot += results[i] * targets[i];
        sum_o += results[i];
    }
    if(sum_o < 1e-6f) sum_o = 1e-6f; // 防止除零
    float p = dot / sum_o;
    return 1.0f - p;
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
    // 预计算sum_o和找到正确类别
    float sum_o = 0.0f;
    int correct_class = -1;
    for(int i=0; i<output_size; i++){
        float o = results[i];
        sum_o += o;//sigma(exp(o))
        if(targets[i] > 0.5f) correct_class = i;
    }
    if(sum_o < 1e-6f) sum_o = 1e-6f;
    
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
            
            // loss = 1 - dot(o,t)/sum_o
            // dLoss/do_i = -d(p)/do_i

            // p = o_c / sum_o (c是正确类别)
            // results[i] = exp(x_i - max)，已经是exp后的值
            // dp/dx_i = (δ_ic * sum_o - exp_i) / sum_o² * exp_i
            float exp_i = results[end_id];  // 直接用，不再exp
            
            // Softmax loss梯度 (α = 1)
            float softmax_grad;
            if(end_id == correct_class){
                softmax_grad = -((sum_o - exp_i) / (sum_o * sum_o)) * exp_i;
            } else {
                softmax_grad = (results[correct_class] / (sum_o * sum_o)) * exp_i;
            }
            
            // MSE on sigmoid梯度 (β = 4)
            // 获取raw值
            float raw_i = nodes_array[layers[LAYER_NUM-1][end_id]]->self_val;
            float sig_i = s(raw_i);  // sigmoid(raw_i)
            float t_i = targets[end_id];
            // dMSE/dx_i = 2 * (sigmoid(x_i) - t_i) * sigmoid(x_i) * (1 - sigmoid(x_i))
            float mse_grad = 2.0f * (sig_i - t_i) * sig_i * (1.0f - sig_i);
            
            // 组合梯度：α * softmax_grad + β * mse_grad，比例1:2
            float delta_output = 1.0f * softmax_grad + 2.0f * mse_grad;
            
            // 裁剪 delta_output
            if(delta_output > GRAD_CLIP_W_PARTIAL) delta_output = GRAD_CLIP_W_PARTIAL;
            if(delta_output < -GRAD_CLIP_W_PARTIAL) delta_output = -GRAD_CLIP_W_PARTIAL;
            
            float w_partial = delta_output * activated_val;


            
            // 裁剪 w_partial
            if(w_partial > GRAD_CLIP_W_PARTIAL) w_partial = GRAD_CLIP_W_PARTIAL;
            if(w_partial < -GRAD_CLIP_W_PARTIAL) w_partial = -GRAD_CLIP_W_PARTIAL;
            
            if(node->w_par != NULL){
                node->w_par[j] = node->w_par[j]*0.5 + 0.5*w_partial;
                // 裁剪 w_par
                if(node->w_par[j] > GRAD_CLIP_W_PAR) node->w_par[j] = GRAD_CLIP_W_PAR;
                if(node->w_par[j] < -GRAD_CLIP_W_PAR) node->w_par[j] = -GRAD_CLIP_W_PAR;
            }
            
            if(node->b_par != NULL){
                node->b_par[j] = 0;
            }
            
            float grad_contrib = delta_output * wi;
            // 裁剪梯度贡献
            if(grad_contrib > GRAD_CLIP_GRAD_CONTRIB) grad_contrib = GRAD_CLIP_GRAD_CONTRIB;
            if(grad_contrib < -GRAD_CLIP_GRAD_CONTRIB) grad_contrib = -GRAD_CLIP_GRAD_CONTRIB;
            node_partial += grad_contrib;
        }
        // 裁剪梯度
        if(node_partial > GRAD_CLIP_NODE_PARTIAL) node_partial = GRAD_CLIP_NODE_PARTIAL;
        if(node_partial < -GRAD_CLIP_NODE_PARTIAL) node_partial = -GRAD_CLIP_NODE_PARTIAL;
        
        float all_partial_i = node_partial*0.2 + all_partials[node_index]*0.8f;
        
        // 裁剪最终梯度
        if(all_partial_i > GRAD_CLIP_ALL_PARTIAL) all_partial_i = GRAD_CLIP_ALL_PARTIAL;
        if(all_partial_i < -GRAD_CLIP_ALL_PARTIAL) all_partial_i = -GRAD_CLIP_ALL_PARTIAL;
        
        all_partials[node_index] = all_partial_i;
    }
}

void backward_pass(){
    for(int j=LAYER_NUM-3; j>=0; j--)
    for(int i=0; i<layer_size[j]; i++)
    partial_resolver(nodes_array[layers[j][i]], z, z_partial);
}
#endif
