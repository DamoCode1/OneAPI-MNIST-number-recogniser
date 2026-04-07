// MNIST dataset = https://www.kaggle.com/datasets/amineipad/mnist-dataset/data
// Image counts per number 0->9 =  4132, 4684, 4177, 4351, 4072, 3795, 4137, 4401, 4063, 4188 (Minimum = 3795, so 3795 batches with each number) 
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

//Initialisation
#include <sycl/sycl.hpp>
#include <iostream>
#include <filesystem>
using namespace sycl;
using namespace std;

/*
* The neural net is fixed at 4 layers(including input and output).
* The input neuron count should be fixed for 28x28 images, and output neuron count should be fixed for numbers 0->9.
* However, the neuron count of the intermediate layers can be chosen freely.
* Furthermore, the batch size is fixed for 10 numbers, while the amount of batches can be lowered.
* The amount of epochs can be chosen freely.
*/
const int lInSize = 784, l1Size = 16, l2Size = 16, lOutSize = 10;
const int batchSize = 10, batchCount = 3795, epochcount = 10;

queue q;
float* bias1;
float* bias2;
float* biasOut;
float* weightIn;
float* weight1;
float* weight2;
bool training;

// ChatGPT made a device-based random value generator, which is applied over a list of values. A unique randID is also inputed, for variance over lists.
inline void randomParamaterInit(float* paramaters, int size, int randID, queue& q) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            float randVal = float(((i + randID * 10000000) ^ 0xA3B1C2D3) * 1664525 + 1013904223 & 0xFFFFFF) / float(0xFFFFFF); // Random value between 0 and 1
            paramaters[i] = randVal * 2 - 1;
        });
    });
}

inline void filedParamaterInit(float* paramaters, float* paramatersRead, int size, queue& q) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            paramaters[i] = paramatersRead[i];
        });
    });
}

inline void train() {
    randomParamaterInit(bias1, l1Size, 1, q);
    randomParamaterInit(bias2, l2Size, 2, q);
    randomParamaterInit(biasOut, lOutSize, 3, q);
    randomParamaterInit(weightIn, lInSize * l1Size, 4, q);
    randomParamaterInit(weight1, l1Size * l2Size, 5, q);
    randomParamaterInit(weight2, l2Size * lOutSize, 6, q);
    q.wait();
}

inline void test() {
    ifstream paramIn("paramaters.txt");
    if (!paramIn.is_open()) cerr << "Failed to link\n";
    float bias1Read[l1Size], bias2Read[l2Size], biasOutRead[lOutSize], weightInRead[lInSize * l1Size], weight1Read[l1Size * l2Size], weight2Read[l2Size * lOutSize];
    for (int i = 0; i < l1Size; i++) paramIn >> bias1Read[i];
    for (int i = 0; i < l2Size; i++) paramIn >> bias2Read[i];
    for (int i = 0; i < lOutSize; i++) paramIn >> biasOutRead[i];
    for (int i = 0; i < lInSize * l1Size; i++) paramIn >> weightInRead[i];
    for (int i = 0; i < l1Size * l2Size; i++) paramIn >> weight1Read[i];
    for (int i = 0; i < l2Size * lOutSize; i++) paramIn >> weight2Read[i];
    q.memcpy(bias1, bias1Read, l1Size * sizeof(float));
    q.memcpy(bias2, bias2Read, l2Size * sizeof(float));
    q.memcpy(biasOut, biasOutRead, lOutSize * sizeof(float));
    q.memcpy(weightIn, weightInRead, lInSize * l1Size * sizeof(float));
    q.memcpy(weight1, weight1Read, l1Size * l2Size * sizeof(float));
    q.memcpy(weight2, weight2Read, l2Size * lOutSize * sizeof(float));
    q.wait();
}

int main() {
    q = queue(default_selector_v);
    cout << q.get_device().get_info<info::device::name>() << "\n";
    cout << "(1 = Training, 0 = Testing): ";
    cin >> training;
    
    // Allocates memory to store paramaters
    bias1 = malloc_device<float>(l1Size, q);
    bias2 = malloc_device<float>(l2Size, q);
    biasOut = malloc_device<float>(lOutSize, q);
    weightIn = malloc_device<float>(lInSize * l1Size, q);
    weight1 = malloc_device<float>(l1Size * l2Size, q);
    weight2 = malloc_device<float>(l2Size * lOutSize, q);

    if (training) train();
    else test();
    
    float bias1Host[l1Size], bias2Host[l2Size], biasOutHost[lOutSize], weightInHost[lInSize * l1Size], weight1Host[l1Size * l2Size], weight2Host[l2Size * lOutSize];
    q.memcpy(bias1Host, bias1, l1Size * sizeof(float));
    q.memcpy(bias2Host, bias2, l2Size * sizeof(float));
    q.memcpy(biasOutHost, biasOut, lOutSize * sizeof(float));
    q.memcpy(weightInHost, weightIn, lInSize * l1Size * sizeof(float));
    q.memcpy(weight1Host, weight1, l1Size * l2Size * sizeof(float));
    q.memcpy(weight2Host, weight2, l2Size * lOutSize * sizeof(float));
    q.wait();

    for (int i = 0; i < l1Size; i++) cout << bias1Host[i] << " ";
    cout << "\n";
    for (int i = 0; i < l2Size; i++) cout << bias2Host[i] << " ";
    cout << "\n";
    for (int i = 0; i < lOutSize; i++) cout << biasOutHost[i] << " ";
    cout << "\n";
    for (int i = 0; i < lInSize * l1Size; i++) cout << weightInHost[i] << " ";
    cout << "\n";
    for (int i = 0; i < l1Size * l2Size; i++) cout << weight1Host[i] << " ";
    cout << "\n";
    for (int i = 0; i < l2Size * lOutSize; i++) cout << weight2Host[i] << " ";
    cout << "\n";

    if (training) {
        ofstream paramOut("paramaters.txt");
        for (float i : bias1Host) paramOut << i;
        for (float i : bias2Host) paramOut << i;
        for (float i : biasOutHost) paramOut << i;
        for (float i : weightInHost) paramOut << i;
        for (float i : weight1Host) paramOut << i;
        for (float i : weight2Host) paramOut << i;
    }
}
