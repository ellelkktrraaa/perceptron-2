#ifndef PARTIAL_RESOLVER_H
#define PARTIAL_RESOLVER_H

#include "ner_sys.h"

// 全局变量，存储整个网络的nodes数组
extern Node* nodes_array;

// 使用Node结构体的梯度计算函数
// node: 当前节点 (包含self_val, weights, b_par, w_par等)
// downstream_partials: 下游节点传回来的梯度数组
void partial_resolver(Node* node, float* downstream_partials){
    // 从Node结构体中提取参数
    float self_val = node->self_val;
    float* weights = node->weights;
    size_t weight_size = node->link_num;

    // 初始化存储（如果需要）
    if(node->w_par == NULL){
        node->w_par = (float*)malloc(weight_size * sizeof(float));
    }
    if(node->b_par == NULL){
        node->b_par = (float*)malloc(weight_size * sizeof(float));
    }

    //caculate the partials of weights
    float* deltas = (float*)malloc(sizeof(float)*weight_size);
    for(int i=0; i<weight_size; i++){
        deltas[i] = downstream_partials[i]*LEARNING_RATE;
    }

    // 使用Node结构体的存储
    float* w_par = node->w_par;
    float* b_par = node->b_par;
    float self_partial = 0.0f;

    for(int i=0; i<weight_size; i++){
        // 获取下游节点的bias：nodes_array[node->link_table[i]].self_bia
        float downstream_bias = nodes_array[node->link_table[i]].self_bia;
        float freshed_contribution_to_node = z_partial(self_val*weights[i] - downstream_bias);
        w_par[i] = freshed_contribution_to_node*self_val*downstream_partials[i]*1.0f;
        b_par[i] = freshed_contribution_to_node*downstream_partials[i]*(-1.0f);
        self_partial += downstream_partials[i]*weights[i]*z_partial(self_val*weights[i] - downstream_bias);
    }

    // 将结果存回Node结构体
    node->self_partial = self_partial;
    /**
     * 所以我们要聚焦在最后两个部分呢喵, ian是这样想的:
     (∂downstream_node_i/∂net_input_i), 
     也就是(p(node_i)/p(new_node)), 应该是前一个点的输出值对后一个点的值的影响呢, 
     也就是 输出值:self_val*weight, 对下一个点的值的贡献: z(self_val*weight_i-bia_i)的影响呢, 
     所以是z'(x-bia)|x=z_partial(x-bia)*1. 
     最后是(∂net_input_i/∂weights[i]),
     也就是 (new_node/p(w_i)) , w_i对输出值的影响呢, (wi*self_val)'=self_val, 是这样喵!

     好呢喵, 我们来推b_partial喵, 
     首先(pE/p(b_i))=(pE/p(node_i))*(p(node_i)/p(new_node))*(p(new_node)/p(b_i)),
     2). pz(x-b_i)/p(x-bi)=z_partial(x-b_i)*1
     3).p(x-b_i)/pb_i=-1
    
    最后呢喵, 我们可以得到self_partial喵, 
    也就是(pE/p(self_val)
    =Σ{i=0,weight_size}[pE/p(node_i)*p(node_i)/p(self_val)]
    =Σ{i=0,weight_size}[node_partials[i]*weights[i]*z_partial(self_val*weights[i]-bias[i])]
     */


/*we need to caculate the partial-graphs on then bench first and then averange them
**=================================================================================
**then we can update the weights and the bias
*/
    free(deltas);
}

#endif
