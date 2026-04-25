#include "backward_spread.h"
#include "forward_spread.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
        all_partials[i] = 0.0f;
        bia_partials[i] = 0.0f;
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

    printf("[TEST] Test network initialized: %d+%d+%d+%d\n", 
           TEST_LAYER0_SIZE, TEST_LAYER1_SIZE, TEST_LAYER2_SIZE, TEST_LAYER3_SIZE);
}

void reset_all_partials() {
    for (int i = 0; i < NODE_NUM; i++) {
        all_partials[i] = 0.0f;
        bia_partials[i] = 0.0f;
    }
}

void test_get_err() {
    printf("\n=== Testing get_err ===\n");
    
    float results[3] = {0.5f, 0.5f, 0.5f};
    float targets[3] = {1.0f, 0.0f, 1.0f};
    
    float err = get_err(results, targets, 3);
    
    printf("Results: [%.4f, %.4f, %.4f]\n", results[0], results[1], results[2]);
    printf("Targets: [%.4f, %.4f, %.4f]\n", targets[0], targets[1], targets[2]);
    printf("Error: %.4f\n", err);
    
    float expected_err = (0.25f + 0.25f + 0.25f) / 3.0f;
    bool passed = fabs(err - expected_err) < 0.0001f;
    
    printf("get_err test %s\n", passed ? "PASSED" : "FAILED");
}

void test_init_partials_simple() {
    printf("\n=== Testing init_partials (simple case) ===\n");
    
    float test_input[TEST_LAYER0_SIZE] = {0.5f, 0.5f};
    init_val(test_input);
    forward_spread();
    
    float results[TEST_LAYER3_SIZE] = {0.0f, 0.0f};
    float targets[TEST_LAYER3_SIZE] = {1.0f, 0.0f};
    
    printf("Input: [%.4f, %.4f]\n", test_input[0], test_input[1]);
    printf("Results: [%.4f, %.4f]\n", results[0], results[1]);
    printf("Targets: [%.4f, %.4f]\n", targets[0], targets[1]);
    
    reset_all_partials();
    init_partials(results, targets);
    
    printf("After init_partials:\n");
    for (int i = 0; i < TEST_LAYER2_SIZE; i++) {
        int idx = TEST_LAYER0_SIZE + TEST_LAYER1_SIZE + i;
        printf("  Layer 2 node %d (index %d): all_partial = %.6f\n", i, idx, all_partials[idx]);
        Node* node = nodes_array[idx];
        if (node->w_par != NULL) {
            for (int j = 0; j < node->link_num; j++) {
                printf("    w_par[%d] = %.6f\n", j, node->w_par[j]);
            }
        }
    }
    
    printf("init_partials simple test completed\n");
}

void test_init_partials_different_targets() {
    printf("\n=== Testing init_partials (different targets) ===\n");
    
    float test_input[TEST_LAYER0_SIZE] = {1.0f, 1.0f};
    init_val(test_input);
    forward_spread();
    
    float results[TEST_LAYER3_SIZE] = {0.3f, 0.7f};
    float targets[TEST_LAYER3_SIZE] = {0.0f, 1.0f};
    
    printf("Input: [%.4f, %.4f]\n", test_input[0], test_input[1]);
    printf("Results: [%.4f, %.4f]\n", results[0], results[1]);
    printf("Targets: [%.4f, %.4f]\n", targets[0], targets[1]);
    
    reset_all_partials();
    init_partials(results, targets);
    
    printf("After init_partials:\n");
    for (int i = 0; i < TEST_LAYER2_SIZE; i++) {
        int idx = TEST_LAYER0_SIZE + TEST_LAYER1_SIZE + i;
        printf("  Layer 2 node %d (index %d): all_partial = %.6f\n", i, idx, all_partials[idx]);
        Node* node = nodes_array[idx];
        if (node->w_par != NULL) {
            for (int j = 0; j < node->link_num; j++) {
                printf("    w_par[%d] = %.6f\n", j, node->w_par[j]);
            }
        }
    }
    
    printf("init_partials different targets test completed\n");
}

void free_test_network() {
    for (int i = 0; i < NODE_NUM; i++) {
        if (nodes_array[i]->weights != NULL) {
            free(nodes_array[i]->weights);
        }
        if (nodes_array[i]->w_par != NULL) {
            free(nodes_array[i]->w_par);
        }
        if (nodes_array[i]->b_par != NULL) {
            free(nodes_array[i]->b_par);
        }
        free(nodes_array[i]);
    }
}

int main() {
    printf("========================================\n");
    printf("  Backward Spread Module Tests\n");
    printf("========================================\n");
    
    init_test_network();
    
    test_get_err();
    test_init_partials_simple();
    test_init_partials_different_targets();
    
    free_test_network();
    
    printf("\n========================================\n");
    printf("  All tests completed!\n");
    printf("========================================\n");
    
    return 0;
}
