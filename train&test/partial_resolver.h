#ifndef __PARTIAL_RESOLVER_H
#define __PARTIAL_RESOLVER_H

#include "ner_sys.h"
//国内提供的所有ai连梯度的正负都分不清喵
//千万不要在核心算法上用ai喵
//甚至认为要吧bias加上去, 气笑了喵
void partial_resolver(Node* node, float simu(float),  float simup(float)){

    float self_val = node->self_val;
    float* weights = node->weights;
    int* link_table = node->link_table;
    size_t weight_size = node->link_num;

    if(node->w_par == NULL){
        node->w_par = (float*)malloc(weight_size * sizeof(float));
        for(int i=0; i<weight_size; i++){
            node->w_par[i] = 0;
        }
    }
    if(node->b_par == NULL){
        node->b_par = (float*)malloc(weight_size * sizeof(float));
        for(int i=0; i<weight_size; i++){
            node->b_par[i] = 0;
        }
    }
//是否zerofy w_par, b_par: 不需要, 全是赋值操作
    float* w_par = node->w_par;
    float* b_par = node->b_par;
    float self_partial = 0.0f;
    
    float bia = node->self_bia;
    float activated_val = z(self_val - bia);

    for(int i=0; i<weight_size; i++){
        float downstream_bias = nodes_array[node->link_table[i]]->self_bia;
        float downstream_self_val = nodes_array[link_table[i]]->self_val;
        float freshed_contribution_to_node = z_partial(downstream_self_val - downstream_bias);
        float w_grad = freshed_contribution_to_node*activated_val*all_partials[link_table[i]]*1.0f;
        float b_grad = freshed_contribution_to_node*all_partials[link_table[i]]*(-1.0f);
        
        // 裁剪梯度，放宽限制
        if(w_grad > 3.0f) w_grad = 3.0f;
        if(w_grad < -3.0f) w_grad = -3.0f;
        if(b_grad > 3.0f) b_grad = 3.0f;
        if(b_grad < -3.0f) b_grad = -3.0f;
        
        w_par[i] = w_par[i]*0.8+0.2*w_grad;
        b_par[i] = b_par[i]*0.8+0.2*b_grad;
        
        // 裁剪 w_par 和 b_par，放宽限制
        if(w_par[i] > 10.0f) w_par[i] = 10.0f;
        if(w_par[i] < -10.0f) w_par[i] = -10.0f;
        if(b_par[i] > 10.0f) b_par[i] = 10.0f;
        if(b_par[i] < -10.0f) b_par[i] = -10.0f;
        
        bia_partials[node->link_table[i]] += b_par[i];
        float grad_contrib = all_partials[link_table[i]]*weights[i]*freshed_contribution_to_node;
        // 裁剪梯度贡献，放宽限制
        if(grad_contrib > 10.0f) grad_contrib = 10.0f;
        if(grad_contrib < -10.0f) grad_contrib = -10.0f;
        self_partial += grad_contrib;
    }

    // 裁剪 self_partial，放宽限制
    if(self_partial > 10.0f) self_partial = 10.0f;
    if(self_partial < -10.0f) self_partial = -10.0f;

    all_partials[node->index] = 0.2*self_partial + 0.8*all_partials[node->index];
    
    // 裁剪最终梯度，放宽限制
    if(all_partials[node->index] > 10.0f) all_partials[node->index] = 10.0f;
    if(all_partials[node->index] < -10.0f) all_partials[node->index] = -10.0f;
    /**
     * 所以我们要聚焦在最后两个部分呢喵, ian是这样想的:
     (∂downstream_node_i/∂net_input_i), 
     也就是(p(node_i)/p(new_node)), 应该是前一个点的输出值对后一个点的值的影响呢, 
     也就是 输出值:self_val*weight, 对下一个点的值的贡献: z(sigma(self_val*weight_i)-bia_i)的影响呢, 
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
}

#endif
