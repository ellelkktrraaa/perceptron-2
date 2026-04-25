#include "forward_spread.h"
#include <cstdio>
#include <cstdlib>

int layers[LAYER_NUM][MAX_LAYER_SIZE];
int layer_size[LAYER_NUM];
Node* nodes_array[NODE_NUM];
float all_partials[NODE_NUM];
float bia_partials[NODE_NUM];

#define TEST_LAYER0_SIZE 2
#define TEST_LAYER1_SIZE 2
#define TEST_LAYER2_SIZE 2
#define TEST_LAYER3_SIZE 2

void init_test_network() {
    for (int i = 0; i < NODE_NUM; i++) {
        nodes_array[i] = (Node*)malloc(sizeof(Node));
        nodes_array[i]->index = i;
        nodes_array[i]->self_val = 0.0f;
        nodes_array[i]->w_par = NULL;
        nodes_array[i]->b_par = NULL;
        nodes_array[i]->weights = NULL;
        nodes_array[i]->link_table = NULL;
        nodes_array[i]->link_num = 0;
        nodes_array[i]->self_bia = 0.0f;
    }

    int layer0_nodes[TEST_LAYER0_SIZE] = {0, 1};
    init_layer(0, TEST_LAYER0_SIZE, layer0_nodes);

    int layer1_nodes[TEST_LAYER1_SIZE] = {2, 3};
    init_layer(1, TEST_LAYER1_SIZE, layer1_nodes);

    int layer2_nodes[TEST_LAYER2_SIZE] = {4, 5};
    init_layer(2, TEST_LAYER2_SIZE, layer2_nodes);

    int layer3_nodes[TEST_LAYER3_SIZE] = {6, 7};
    init_layer(3, TEST_LAYER3_SIZE, layer3_nodes);

    for (int i = 0; i < TEST_LAYER0_SIZE; i++) {
        nodes_array[i]->link_num = TEST_LAYER1_SIZE;
        nodes_array[i]->link_table = layers[1];
        nodes_array[i]->weights = (float*)malloc(TEST_LAYER1_SIZE * sizeof(float));
        nodes_array[i]->self_bia = 0.0f;
        for (int j = 0; j < TEST_LAYER1_SIZE; j++) {
            nodes_array[i]->weights[j] = 1.0f;
        }
    }

    for (int i = 0; i < TEST_LAYER1_SIZE; i++) {
        int idx = TEST_LAYER0_SIZE + i;
        nodes_array[idx]->link_num = TEST_LAYER2_SIZE;
        nodes_array[idx]->link_table = layers[2];
        nodes_array[idx]->weights = (float*)malloc(TEST_LAYER2_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = 0.0f;
        for (int j = 0; j < TEST_LAYER2_SIZE; j++) {
            nodes_array[idx]->weights[j] = 1.0f;
        }
    }

    for (int i = 0; i < TEST_LAYER2_SIZE; i++) {
        int idx = TEST_LAYER0_SIZE + TEST_LAYER1_SIZE + i;
        nodes_array[idx]->link_num = TEST_LAYER3_SIZE;
        nodes_array[idx]->link_table = layers[3];
        nodes_array[idx]->weights = (float*)malloc(TEST_LAYER3_SIZE * sizeof(float));
        nodes_array[idx]->self_bia = 0.0f;
        for (int j = 0; j < TEST_LAYER3_SIZE; j++) {
            nodes_array[idx]->weights[j] = 1.0f;
        }
    }

    for (int i = 0; i < TEST_LAYER3_SIZE; i++) {
        int idx = TEST_LAYER0_SIZE + TEST_LAYER1_SIZE + TEST_LAYER2_SIZE + i;
        nodes_array[idx]->self_bia = (float)i;
        nodes_array[idx]->link_num = 0;
        nodes_array[idx]->weights = NULL;
        nodes_array[idx]->link_table = NULL;
    }
}

void print_network_state(const char* label) {
    printf("\n=== %s ===\n", label);
    printf("Layer 0: ");
    for (int i = 0; i < TEST_LAYER0_SIZE; i++) {
        printf("%.4f ", nodes_array[layers[0][i]]->self_val);
    }
    printf("\nLayer 1: ");
    for (int i = 0; i < TEST_LAYER1_SIZE; i++) {
        printf("%.4f ", nodes_array[layers[1][i]]->self_val);
    }
    printf("\nLayer 2: ");
    for (int i = 0; i < TEST_LAYER2_SIZE; i++) {
        printf("%.4f ", nodes_array[layers[2][i]]->self_val);
    }
    printf("\nLayer 3: ");
    for (int i = 0; i < TEST_LAYER3_SIZE; i++) {
        printf("%.4f ", nodes_array[layers[3][i]]->self_val);
    }
    printf("\n");
}

int main() {
    printf("=== Debug Forward Spread ===\n");
    
    init_test_network();
    
    float test_input[TEST_LAYER0_SIZE] = {1.0f, 1.0f};
    init_val(test_input);
    print_network_state("After init_val (before forward_spread)");
    
    forward_spread();
    print_network_state("After forward_spread");
    
    float* results_sig = get_final_nodes_val((char*)"sigmoide");
    float* results_z = get_final_nodes_val((char*)"other");
    printf("\nFinal outputs:\n");
    printf("  Sigmoid: [%.4f, %.4f]\n", results_sig[0], results_sig[1]);
    printf("  ReLU:    [%.4f, %.4f]\n", results_z[0], results_z[1]);
    
    delete[] results_sig;
    delete[] results_z;
    
    return 0;
}
