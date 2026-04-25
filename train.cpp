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

#define DATA_BEGIN_INDEX 0
#define DATA_END_INDEX 60

#define __FILE "network-2.json"
#define DATA_PATH "mn\\tr_12.mnib"

#define RUNS 120

float loss=1.0f;


// 调用 Python 脚本的函数
int call_python_script(const char* script_path, const char* args) {
    char cmd[512];
    if (args != NULL && strlen(args) > 0) {
        sprintf(cmd, "python \"%s\" %s", script_path, args);
    } else {
        sprintf(cmd, "python \"%s\"", script_path);
    }
    
    printf("[INFO] Calling Python script: %s mio~\n", cmd);
    int result = system(cmd);
    
    if (result == 0) {
        printf("[INFO] Python script completed successfully nia~\n");
    } else {
        printf("[WARNING] Python script returned with code: %d nia~\n", result);
    }
    
    return result;
}


float min_lr = 0.000195f; 
float decay_rate = 0.985f; 
float current_lr = LEARNING_RATE;


// 初始化400+90+60+10全连接神经网络
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
    float* results = get_final_nodes_val("sigmoide");
    float err = get_err(results, targets, output_size);
    delete[] targets;
    delete[] results;
    return err;
}
// 判断网络输出结果，返回可能性最大的类别
int get_predicted_class(){
    float* results = get_final_nodes_val((char*)"sigmoide");
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

void train_sigal_bench(char* file_path){
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
    char bench_num_s[5] = {0},
        dummy_s[5]={0},
        lable_s[5]={0};

    fread(bench_num_s, 1, 4, f);

    int bench_num = *(int*)(bench_num_s);

    // 读取版本号（用于未来扩展）
    fread(dummy_s, 1, 4, f);
    int version = *(int*)(dummy_s);

    // 读取特征数量
    fread(dummy_s, 1, 4, f);
    int feature_count = *(int*)(dummy_s);

    // 数据大小：784个float = 3136字节
    int data_size = feature_count * sizeof(float);


    for(int runs=0; runs<RUNS; runs++){
        int correct_count = 0;
        int total_count = 0;
        
        // 计算当前轮的学习率（指数衰减）
        current_lr = current_lr*decay_rate;
        current_lr = current_lr > min_lr ? current_lr : min_lr;
        
        // printf("[INFO] Epoch %d, Learning Rate: %.6f\n", runs+1, current_lr);
        
        fseek(f, 16, SEEK_SET);
        
        for(int i=0; i<bench_num; i++){
            // 读取标签
            fread(lable_s, 1, 4, f);
            int lable = *(int*)(lable_s);

            char* input_s = new char[data_size];
            int read_bytes = fread(input_s, 1, data_size, f);

            if(read_bytes != data_size){
                delete[] input_s;
                break;
            }
            float* input = (float*)input_s;
            
    //train begin
    //卷积28*28-->20*20, 边缘特征很重要, 最大池化?
    #define INI_SIZE_F 28
    #define INI_SIZE_S 28
    #define SHRINKED_SIZE_F 20
    #define SHRINKED_SIZE_S 20
            float shrinked_input[SHRINKED_SIZE_F * SHRINKED_SIZE_S];
            maxav_poolify(input, shrinked_input, INI_SIZE_F, INI_SIZE_S, SHRINKED_SIZE_F, SHRINKED_SIZE_S);
    //池化完成, 会不会很糊? --> 从值的最大池化改成了池的值*覆盖面积的最大池化, 应该会好一些
    // 将池化结果输入神经网络第0层
            init_val(shrinked_input);
    //get partial
            forward_spread();
            float* results_sigmoid = get_final_nodes_val((char*)"sigmoide");
            float* results_raw = get_final_nodes_raw_val();

            int output_size = layer_size[LAYER_NUM-1];
            float* targets = new float[output_size]();
            if(lable >= 0 && lable < output_size){
                targets[lable] = 1.0f;
            }

            // // 打印输出层的原始输入值（应用 sigmoide 之前）
            // if(rand() % 2000 == 0 ){//&& lable == 1 && res == 2){
            //     printf("[DEBUG] Output layer raw inputs: \n");
            //     for(int j=0; j<output_size; j++){
            //         float raw_input = results_raw[j];
            //         float sigmoid_input = results_sigmoid[j];
            //         float target = targets[j];
            //         printf("t: %.4f ", target);
            //         printf("r_raw: %.4f ", raw_input);
            //         printf("r_sig: %.4f ", sigmoid_input);
            //         printf("d: %.4f \n", target-sigmoid_input);
            //     }
            // }

            int predicted_class = get_predicted_class();
            
            
            if(predicted_class == lable){
                correct_count++;
            }
            total_count++;
            
            init_partials(results_raw, targets);
            delete[] results_sigmoid;
            delete[] results_raw;
            delete[] targets;
            backward_pass();
            // for(int i=0; i<NODE_NUM; i++){
            //     std::cout<<all_partials[i]<<" ";
            // }
            delete[] input_s;
    //update
            for(int li=0; li<LAYER_NUM-1; li++){
                for(int ni=0; ni<layer_size[li]; ni++){
                    Node* node = nodes_array[layers[li][ni]];
                    
                    // 梯度裁剪并更新偏置（使用动态学习率）
                    float b_grad = current_lr * bia_partials[node->index];
                    if(b_grad > 1.0f) b_grad = 1.0f;
                    if(b_grad < -1.0f) b_grad = -1.0f;
                    node->self_bia -= b_grad;
                    
                    // 裁剪偏置值，放宽限制
                    if(node->self_bia > 10.0f) node->self_bia = 10.0f;
                    if(node->self_bia < -10.0f) node->self_bia = -10.0f;
                    
                    for(int wi=0; wi<node->link_num; wi++){

                        float w_grad = current_lr * node->w_par[wi];
                        if(w_grad > 1.0f) w_grad = 1.0f;
                        if(w_grad < -1.0f) w_grad = -1.0f;
                        node->weights[wi] -= w_grad;
                        

                        if(node->weights[wi] > 10.0f) node->weights[wi] = 10.0f;
                        if(node->weights[wi] < -10.0f) node->weights[wi] = -10.0f;
                    }
                }
            }
            
            // 计算这次传播的loss（使用get_err函数）
            float batch_loss = calculate_loss(lable);
            loss = 0.8f * loss + 0.2f * batch_loss;
        }
        
        // 打印本轮训练结果
        if((total_count > 0 && runs%20 == 0 )|| (runs <= 10 && runs%2==0)){
            float accuracy = (float)correct_count / total_count * 100;
            printf("[INFO] Run %d/%d completed: %d/%d correct (%.2f%% accuracy), loss: %.8f %%, learning_rate: %.8f\n", 
                   runs+1, RUNS, correct_count, total_count, accuracy, loss*100, current_lr);
        }
        if(runs%50==0){
            save_network(__FILE, loss, LEARNING_RATE);
            printf("saved to %s", __FILE);
        }
    }

    fclose(f);
}

int main(){
    printf("[INFO] training\n");
    srand(10086);
    
    // 初始化神经网络
    init_network();
    
    
    // // 尝试加载已有网络
    // float loaded_loss = 1.0f;
    // float loaded_lr = LEARNING_RATE;
    // if(load_network(__FILE, &loaded_loss, &loaded_lr)){
    //     loss = loaded_loss;
    //     printf("[INFO] Resumed from previous training, loss: %.4f\n", loss);
    // } else {
    //     printf("[INFO] Starting new training\n");
    // }
    
    // printf("[INFO] Network loading completed\n");
    
    // 使用可修改的字符数组作为文件路径

    // 执行训练
    for(int j = 0; j<5; j++){
        for(int i = DATA_BEGIN_INDEX; i <DATA_END_INDEX; i++){    
            
            // 尝试加载已有网络
            float loaded_loss = 1.0f;
            float loaded_lr = LEARNING_RATE;
            if(load_network(__FILE, &loaded_loss, &loaded_lr)){
                loss = loaded_loss;
                printf("[INFO] Resumed from previous training, loss: %.4f\n", loss);
            } else {
                printf("[INFO] Starting new training\n");
            }
            
            printf("[INFO] Network loading completed\n");
            
            char file_path[50] = {0};
            char st1[] = "mn\\tr_",
                st2[] = ".mnib";
            sprintf(file_path, "%s%d%s", st1, i, st2);

            train_sigal_bench(file_path);
            save_network(__FILE, loss, LEARNING_RATE);

            current_lr = RETRAIN_LEARNIN_RATE;
            printf("[INFO] Epoch %d completed, calling Python pruning script mio~\n", j+1);
            char python_args[256];
            sprintf(python_args, "\"%s\" --random-percent 0.5", __FILE);
            call_python_script("e:\\c\\perceptron-2\\prune_weights.py", python_args);
        }
    }
    printf("[INFO] Training completed! Final loss: %.4f\n", loss);
}
