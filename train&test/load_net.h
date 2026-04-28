#ifndef __LOAD_NET_H
#define __LOAD_NET_H

#include "ner_sys.h"
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

// 从JSON文件加载网络
// 结构: {loss: float, learning_rate: float, layers: [[{w: [float...], b: float}...]...]}
bool load_network(const char* file_path, float* loss, float* learning_rate){
    std::ifstream file(file_path);
    if(!file.is_open()){
        printf("[ERROR] Cannot open file: %s\n", file_path);
        return false;
    }
    
    json j;
    try{
        file >> j;
    }catch(...){
        printf("[ERROR] Invalid JSON format in: %s\n", file_path);
        file.close();
        return false;
    }
    file.close();
    
    // 读取loss和learning_rate
    if(loss) *loss = j.value("loss", 0.0f);
    if(learning_rate) *learning_rate = j.value("learning_rate", 0.1f);
    
    // 读取各层权重和偏置
    if(!j.contains("layers") || !j["layers"].is_array()){
        printf("[ERROR] Missing or invalid 'layers' in JSON\n");
        return false;
    }
    
    json layers_json = j["layers"];
    if(layers_json.size() != LAYER_NUM){
        printf("[ERROR] Layer count mismatch: JSON has %zu, expected %d\n", 
               layers_json.size(), LAYER_NUM);
        return false;
    }
    
    // 遍历每一层
    for(int li=0; li<LAYER_NUM; li++){
        json layer_json = layers_json[li];
        if(!layer_json.is_array() || layer_json.size() != layer_size[li]){
            printf("[ERROR] Layer %d node count mismatch\n", li);
            return false;
        }
        
        // 遍历每个节点
        for(int ni=0; ni<layer_size[li]; ni++){
            Node* node = nodes_array[layers[li][ni]];
            json node_json = layer_json[ni];
            
            // 读取偏置
            if(node_json.contains("b")){
                node->self_bia = node_json["b"].get<float>();
            }
            
            // 读取权重（除了最后一层）
            if(li < LAYER_NUM-1 && node_json.contains("w") && node_json["w"].is_array()){
                json weights_json = node_json["w"];
                if(weights_json.size() != node->link_num){
                    printf("[ERROR] Layer %d Node %d weight count mismatch\n", li, ni);
                    return false;
                }
                for(int wi=0; wi<node->link_num; wi++){
                    node->weights[wi] = weights_json[wi].get<float>();
                }
            }
        }
    }
    
    printf("[INFO] Network loaded from %s\n", file_path);
    return true;
}

#endif
