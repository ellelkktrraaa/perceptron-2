#ifndef __STORE_NET_H
#define __STORE_NET_H

#include "ner_sys.h"
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

// 保存网络到JSON文件
// 结构: {loss: float, learning_rate: float, layers: [[{w: [float...], b: float}...]...]}
void save_network(const char* file_path, float loss, float learning_rate){
    json j;
    j["loss"] = loss;
    j["learning_rate"] = learning_rate;
    j["layers"] = json::array();
    
    // 遍历每一层（除了最后一层，因为它是输出层，没有权重）
    for(int li=0; li<LAYER_NUM-1; li++){
        json layer = json::array();
        for(int ni=0; ni<layer_size[li]; ni++){
            Node* node = nodes_array[layers[li][ni]];
            json node_json;
            
            // 保存权重
            node_json["w"] = json::array();
            for(int wi=0; wi<node->link_num; wi++){
                node_json["w"].push_back(node->weights[wi]);
            }
            
            // 保存偏置
            node_json["b"] = node->self_bia;
            
            layer.push_back(node_json);
        }
        j["layers"].push_back(layer);
    }
    
    // 最后一层只保存偏置（作为输出节点的标识）
    json last_layer = json::array();
    for(int ni=0; ni<layer_size[LAYER_NUM-1]; ni++){
        Node* node = nodes_array[layers[LAYER_NUM-1][ni]];
        json node_json;
        node_json["w"] = json::array();
        node_json["b"] = node->self_bia;
        last_layer.push_back(node_json);
    }
    j["layers"].push_back(last_layer);
    
    // 写入文件
    std::ofstream file(file_path);
    file << j.dump(4);  // 4空格缩进，便于阅读
    file.close();
    
    printf("[INFO] Network saved to %s\n", file_path);
}

#endif
