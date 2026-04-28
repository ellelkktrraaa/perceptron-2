
#include "test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

float cur_loss = 1.0f;

void test_bench(char* file_path){
    printf("[INFO] Opening file: %s\n", file_path);
    FILE* f = fopen(file_path, "rb");
    if(f == NULL){
        printf("[ERROR] Failed to open file: %s, errno: %d\n", file_path, errno);
        return;
    }

    char __magic[5] = {0};
    if(fread(__magic, 1, 4, f) != 4 || strncmp(__magic, "MNIB", 4) != 0){
        printf("[ERROR] Invalid MNIB file format: %s\n", file_path);
        fclose(f);
        return;
    }

    char bench_num_s[5] = {0};
    char dummy_s[5] = {0};
    char lable_s[5] = {0};

    fread(bench_num_s, 1, 4, f);
    int bench_num = *(int*)bench_num_s;

    fread(dummy_s, 1, 4, f);
    int version = *(int*)dummy_s;

    fread(dummy_s, 1, 4, f);
    int feature_count = *(int*)dummy_s;

    int data_size = feature_count * sizeof(float);

    int correct_count = 0;
    int total_count = 0;

    for(int i=0; i<bench_num; i++){
        fread(lable_s, 1, 4, f);
        int lable = *(int*)lable_s;

        char* input_s = new char[data_size];
        int read_bytes = fread(input_s, 1, data_size, f);

        if(read_bytes != data_size){
            delete[] input_s;
            break;
        }
        float* input = (float*)input_s;
        
        #define INI_SIZE_F 28
        #define INI_SIZE_S 28
        #define SHRINKED_SIZE_F 20
        #define SHRINKED_SIZE_S 20
        float shrinked_input[SHRINKED_SIZE_F * SHRINKED_SIZE_S];
        maxav_poolify(input, shrinked_input, INI_SIZE_F, INI_SIZE_S, SHRINKED_SIZE_F, SHRINKED_SIZE_S);

        init_val(shrinked_input);
        forward_spread();
        
        float* results_raw = get_final_nodes_raw_val();
        float* results_softmax = get_final_nodes_val((char*)"e");

        int output_size = layer_size[LAYER_NUM-1];
        float* targets = new float[output_size]();
        
        if(lable >= 0 && lable < output_size){
            targets[lable] = 1.0f;
        }

        int predicted_class = get_predicted_class();
        if(predicted_class == lable){
            correct_count++;
        }
        total_count++;

        // if(rand() % 200 == 0 ){
        //     printf("[DEBUG] Output layer raw inputs: \n");
        //     for(int j=0; j<output_size; j++){
        //         float raw_input = results_raw[j];
        //         float sigmoid_input = results_softmax[j];
        //         float target = targets[j];
        //         printf("t: %.4f ", target);
        //         printf("r_raw: %.4f ", raw_input);
        //         printf("r_sig: %.4f ", sigmoid_input);
        //         printf("d: %.4f ", target-sigmoid_input);
        //         if(target>0.1){
        //             printf(" t");
        //         }
        //         if(predicted_class==j){
        //             printf(" p");
        //         }
        //         printf("\n");
        //     }
        // }

        // 释放内存，防止泄漏
        delete[] input_s;
        delete[] targets;
    }

    // 关闭文件
    fclose(f);

    printf("TEST FINISHED\n");
    // 修复精度计算错误
    printf("    ACC: %.2f\n", (float)correct_count / total_count);
}

int main() { 
    init_network();

    float loaded_loss = 1.0f;
    float loaded_lr = LEARNING_RATE;
    
    // 修复：__FILE 是宏，替换为合法字符串
    if(load_network("network-2.json", &loaded_loss, &loaded_lr)){
        printf("[INFO] Resumed from previous training, loss: %.4f\n", cur_loss);
    } else {
        printf("[INFO] Starting new training\n");
    }
    
    printf("[INFO] Network loading completed\n");
    
    char* file_path = ".\\mn\\test10000.mnib";
    test_bench(file_path);

    return 0; 
}
