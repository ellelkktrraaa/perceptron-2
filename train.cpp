#include "backward_spread.h"
#include "forward_spread.h"
#include "store_net.h"
#include "load_net.h"
#include <thread>
#include <fstream>

int layers[LAYER_NUM][MAX_LAYER_SIZE];
int layer_size[LAYER_NUM];
Node* nodes_array[NODE_NUM];//id-->ptr
float all_partials[NODE_NUM];//id-->par
float bia_partials[NODE_NUM];//id-->bia_par

// 网络层大小配置：400+200+60+10
#define LAYER0_SIZE 400
#define LAYER1_SIZE 200
#define LAYER2_SIZE 60
#define LAYER3_SIZE 10

#define RUNS 300

float loss=1.0f;


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
    
    // 初始化第0层到第1层的连接（需要预先分配weights内存）
    for(int i=0; i<LAYER0_SIZE; i++){
        nodes_array[i]->link_num = LAYER1_SIZE;
        nodes_array[i]->link_table = layers[1];
        nodes_array[i]->weights = (float*)malloc(LAYER1_SIZE * sizeof(float));
        nodes_array[i]->self_bia = (float)rand()/RAND_MAX;
        for(int j=0; j<LAYER1_SIZE; j++){
            nodes_array[i]->weights[j] = (float)rand()/RAND_MAX;
        }
    }
    
    // 初始化第1层到第2层的连接
    for(int i=0; i<LAYER1_SIZE; i++){
        int idx = LAYER0_SIZE + i;
        nodes_array[idx]->link_num = LAYER2_SIZE;
        nodes_array[idx]->link_table = layers[2];
        nodes_array[idx]->weights = (float*)malloc(LAYER2_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = (float)rand()/RAND_MAX;
        for(int j=0; j<LAYER2_SIZE; j++){
            nodes_array[idx]->weights[j] = (float)rand()/RAND_MAX;
        }
    }
    
    // 初始化第2层到第3层的连接
    for(int i=0; i<LAYER2_SIZE; i++){
        int idx = LAYER0_SIZE + LAYER1_SIZE + i;
        nodes_array[idx]->link_num = LAYER3_SIZE;
        nodes_array[idx]->link_table = layers[3];
        nodes_array[idx]->weights = (float*)malloc(LAYER3_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = (float)rand()/RAND_MAX;
        for(int j=0; j<LAYER3_SIZE; j++){
            nodes_array[idx]->weights[j] = (float)rand()/RAND_MAX;
        }
    }
    
    // 第3层（输出层）不需要权重，只需要设置bia作为标识
    for(int i=0; i<LAYER3_SIZE; i++){
        int idx = LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE + i;
        nodes_array[idx]->self_bia = (float)i;  // 用bia存储节点在输出层的索引
        nodes_array[idx]->link_num = 0;
        nodes_array[idx]->weights = NULL;
        nodes_array[idx]->link_table = NULL;
    }
    
    printf("[INFO] Network initialized: %d+%d+%d+%d\n", LAYER0_SIZE, LAYER1_SIZE, LAYER2_SIZE, LAYER3_SIZE);
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
    float targets[10] = {0};
    targets[(int)lable] = 1;
    float* results = get_final_nodes_val("sigmoide");
    return get_err(results, targets, 10);

}
// 判断网络输出结果，返回可能性最大的类别
int get_predicted_class(){
    float* results = get_final_nodes_val((char*)"sigmoide");
    int predicted_class = 0;
    float max_value = results[0];
    for(int j=1; j<LAYER3_SIZE; j++){
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
            float* results = get_final_nodes_val((char*)"signoide");

            
            float targets[LAYER3_SIZE] = {0.0f};
            if(lable >= 0 && lable < LAYER3_SIZE){
                targets[lable] = 1.0f;
            }

            // 打印输出层的原始输入值（应用 sigmoid 之前）
            int res =  get_predicted_class();
            if(i == 0 ){//&& lable == 1 && res == 2){
                printf("[DEBUG] Output layer raw inputs: \n");
                for(int j=0; j<LAYER3_SIZE; j++){
                    float raw_input = results[j];
                    float target = targets[j];
                    printf("t: %.4f ", target);
                    printf("r: %.4f ", raw_input);
                    printf("d: %.4f \n", target-raw_input);
                }
            }
            
            // 计算预测是否正确
            int predicted_class = get_predicted_class();
            
            
            if(predicted_class == lable){
                correct_count++;
            }
            total_count++;
            
            init_partials(results, targets);
            backward_pass();
            
            delete[] input_s;
    //update
            for(int li=0; li<LAYER_NUM-1; li++){
                for(int ni=0; ni<layer_size[li]; ni++){
                    Node* node = nodes_array[layers[li][ni]];
                    node->self_bia -= LEARNING_RATE * bia_partials[node->index];
                    for(int wi=0; wi<node->link_num; wi++){
                        node->weights[wi] -= LEARNING_RATE * node->w_par[wi];
                    }
                }
            }
            
            // 计算这次传播的loss（使用get_err函数）
            float batch_loss = calculate_loss(lable);
            loss = 0.8f * loss + 0.2f * batch_loss;
            
            delete[] results;
        }
        
        // 打印本轮训练结果
        if(total_count > 0){
            float accuracy = (float)correct_count / total_count * 100;
            printf("[INFO] Run %d/%d completed: %d/%d correct (%.2f%% accuracy), loss: %.4f\n", 
                   runs+1, RUNS, correct_count, total_count, accuracy, loss);
        }
    }

    fclose(f);
}

int main(){
    printf("[INFO] training\n");
    
    // 初始化神经网络
    init_network();
    
    printf("[INFO] About to load network from network.json\n");
    
    // 尝试加载已有网络
    float loaded_loss = 1.0f;
    float loaded_lr = LEARNING_RATE;
    if(load_network("network.json", &loaded_loss, &loaded_lr)){
        loss = loaded_loss;
        printf("[INFO] Resumed from previous training, loss: %.4f\n", loss);
    } else {
        printf("[INFO] Starting new training\n");
    }
    
    printf("[INFO] Network loading completed\n");
    
    // 使用可修改的字符数组作为文件路径
    char data_path[] = "mn\\tr_11.mnib";
    
    // 执行训练
    train_sigal_bench(data_path);
    
    save_network("network.json", loss, LEARNING_RATE);
    printf("[INFO] Training completed! Final loss: %.4f\n", loss);
}
