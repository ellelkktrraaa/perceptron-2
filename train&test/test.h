#ifndef __TEST_H
#define __TEST_H
#include "backward_spread.h"
#include "forward_spread.h"
#include "store_net.h"
#include "load_net.h"
#include <thread>
#include <fstream>
#include <windows.h>
#include <stdio.h>
#include <string.h>

int layers[LAYER_NUM][MAX_LAYER_SIZE];
int layer_size[LAYER_NUM];
Node* nodes_array[NODE_NUM];//id-->ptr
float all_partials[NODE_NUM];//id-->par
float bia_partials[NODE_NUM];//id-->bia_par

// 网络层大小配置：400+200+60+10
#define LAYER0_SIZE 400
#define LAYER1_SIZE 200
#define LAYER2_SIZE 60
#define LAYER3_SIZE 100
#define LAYER4_SIZE 10

#define RETRAIN_LEARNIN_RATE 0.000225f

#define DATA_BEGIN_INDEX 17
#define DATA_END_INDEX 60

#define __FILE "network-2.json"
#define DATA_PATH "mn\\tr_12.mnib"

void init_network(){
    for(int i=0; i<NODE_NUM; i++){
        nodes_array[i] = (Node*)malloc(sizeof(Node));
        nodes_array[i]->index = i;
        nodes_array[i]->self_val = 0.0f;
        nodes_array[i]->w_par = NULL;
        nodes_array[i]->b_par = NULL;
        nodes_array[i]->weights = NULL;
        nodes_array[i]->link_table = NULL;
        nodes_array[i]->link_num = 0;
    }
    
    // 0-399
    int layer0_nodes[LAYER0_SIZE];
    for(int i=0; i<LAYER0_SIZE; i++) layer0_nodes[i] = i;
    init_layer(0, LAYER0_SIZE, layer0_nodes);
    
    // 400-489
    int layer1_nodes[LAYER1_SIZE];
    for(int i=0; i<LAYER1_SIZE; i++) layer1_nodes[i] = LAYER0_SIZE + i;
    init_layer(1, LAYER1_SIZE, layer1_nodes);
    
    // 490-549
    int layer2_nodes[LAYER2_SIZE];
    for(int i=0; i<LAYER2_SIZE; i++) layer2_nodes[i] = LAYER0_SIZE + LAYER1_SIZE + i;
    init_layer(2, LAYER2_SIZE, layer2_nodes);
    
    // 550-559
    int layer3_nodes[LAYER3_SIZE];
    for(int i=0; i<LAYER3_SIZE; i++) layer3_nodes[i] = LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE + i;
    init_layer(3, LAYER3_SIZE, layer3_nodes);
    

    int layer4_nodes[LAYER4_SIZE];
    for(int i=0; i<LAYER3_SIZE; i++) layer4_nodes[i] = LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE + LAYER3_SIZE + i;
    init_layer(4, LAYER4_SIZE, layer4_nodes);

    // 使用 Xavier 初始化权重
    float limit1 = sqrt(6.0f / (LAYER0_SIZE + LAYER1_SIZE));
    for(int i=0; i<LAYER0_SIZE; i++){
        nodes_array[i]->link_num = LAYER1_SIZE;
        nodes_array[i]->link_table = layers[1];
        nodes_array[i]->weights = (float*)malloc(LAYER1_SIZE * sizeof(float));
        nodes_array[i]->self_bia = 0.0f;  // 初始偏置为0
        for(int j=0; j<LAYER1_SIZE; j++){
            nodes_array[i]->weights[j] = ((float)rand()/RAND_MAX)*2.0f*limit1 - limit1;
        }
    }
    
    float limit2 = sqrt(6.0f / (LAYER1_SIZE + LAYER2_SIZE));
    for(int i=0; i<LAYER1_SIZE; i++){
        int idx = LAYER0_SIZE + i;
        nodes_array[idx]->link_num = LAYER2_SIZE;
        nodes_array[idx]->link_table = layers[2];
        nodes_array[idx]->weights = (float*)malloc(LAYER2_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = 0.0f;  // 初始偏置为0
        for(int j=0; j<LAYER2_SIZE; j++){
            nodes_array[idx]->weights[j] = ((float)rand()/RAND_MAX)*2.0f*limit2 - limit2;
        }
    }
    
    float limit3 = sqrt(6.0f / (LAYER2_SIZE + LAYER3_SIZE));
    for(int i=0; i<LAYER2_SIZE; i++){
        int idx = LAYER0_SIZE + LAYER1_SIZE + i;
        nodes_array[idx]->link_num = LAYER3_SIZE;
        nodes_array[idx]->link_table = layers[3];
        nodes_array[idx]->weights = (float*)malloc(LAYER3_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = 0.0f;  // 初始偏置为0
        for(int j=0; j<LAYER3_SIZE; j++){
            nodes_array[idx]->weights[j] = ((float)rand()/RAND_MAX)*2.0f*limit3 - limit3;
        }
    }

    float limit4 = sqrt(6.0f / (LAYER3_SIZE + LAYER4_SIZE));
    for(int i=0; i<LAYER3_SIZE; i++){
        int idx = LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE + i;
        nodes_array[idx]->link_num = LAYER4_SIZE;
        nodes_array[idx]->link_table = layers[4];
        nodes_array[idx]->weights = (float*)malloc(LAYER4_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = 0.0f;  // 初始偏置为0
        for(int j=0; j<LAYER4_SIZE; j++){
            nodes_array[idx]->weights[j] = ((float)rand()/RAND_MAX)*2.0f*limit4 - limit4;
        }
    }
    
    // 第3层（输出层）不需要权重，只需要设置bia作为标识
    for(int i=0; i<LAYER4_SIZE; i++){
        int idx = LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE + LAYER3_SIZE + i;
        nodes_array[idx]->self_bia = (float)i;  // 用bia存储节点在输出层的索引
        nodes_array[idx]->link_num = 0;
        nodes_array[idx]->weights = NULL;
        nodes_array[idx]->link_table = NULL;
    }
    
    printf("[INFO] Network initialized: %d->%d->%d->%d->%d\n", LAYER0_SIZE, LAYER1_SIZE, LAYER2_SIZE, LAYER3_SIZE, LAYER4_SIZE);
}

void maxav_poolify(float* input, float* output, int in_f, int in_s, int out_f, int out_s){
    float df = (float)in_f/out_f,
          ds = (float)in_s/out_s;

    for(int f=0; f<out_f; f++){
        for(int s=0; s<out_s; s++){
            float begin_f = df*f,
                  end_f = df*(f+1),
                  begin_s = ds*s,
                  end_s = ds*(s+1); 
            float max_weighted_v = 0.0f;
            for(int f_i = (int)begin_f; f_i<end_f; f_i++){
                for(int s_i = (int)begin_s; s_i<end_s; s_i++){
                    // 计算该像素在池化区域内的覆盖面积
                    float pixel_begin_f = (float)f_i,
                          pixel_end_f = (float)(f_i + 1),
                          pixel_begin_s = (float)s_i,
                          pixel_end_s = (float)(s_i + 1);
                    
                    float overlap_f = fminf(end_f, pixel_end_f) - fmaxf(begin_f, pixel_begin_f),
                          overlap_s = fminf(end_s, pixel_end_s) - fmaxf(begin_s, pixel_begin_s);
                    float overlap_area = overlap_f * overlap_s;
                    
                    float weighted_val = input[f_i*in_f+s_i] * overlap_area;
                    max_weighted_v = max_weighted_v < weighted_val ? weighted_val : max_weighted_v;
                }
            }
            output[f*out_s+s] = max_weighted_v;
        }
    }
}

// 计算当前网络的平均loss
float calculate_loss(float lable){
    int output_size = layer_size[LAYER_NUM-1];
    float* targets = new float[output_size](); // 初始化为0
    if(lable >= 0 && lable < output_size){
        targets[(int)lable] = 1.0f;
    }
    float* results = get_final_nodes_val((char*)"e");
    float err = get_err(results, targets, output_size);
    delete[] targets;
    delete[] results;
    return err;
}
// 判断网络输出结果，返回可能性最大的类别
int get_predicted_class(){
    float* results = get_final_nodes_val((char*)"e");
    int output_size = layer_size[LAYER_NUM-1];
    int predicted_class = 0;
    float max_value = results[0];
    for(int j=1; j<output_size; j++){
        if(results[j] > max_value){
            max_value = results[j];
            predicted_class = j;
        }
    }
    delete[] results;
    return predicted_class;
}

// 计算预测是否正确
bool is_prediction_correct(int lable){
    int predicted_class = get_predicted_class();
    return predicted_class == lable;
}

#endif
