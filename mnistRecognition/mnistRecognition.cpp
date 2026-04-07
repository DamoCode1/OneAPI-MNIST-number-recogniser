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
#define STB_IMAGE_IMPLEMENTATION
#include "VC_IncludePath\stb_image.h"
using namespace sycl;
using namespace std;
using namespace filesystem;

/*
* The neural net is fixed at 4 layers(including input and output).
* The input neuron count should be fixed for 28x28 images, and output neuron count should be fixed for numbers 0->9.
* However, the neuron count of the intermediate layers can be chosen freely.
* Furthermore, each batch contains each of the 10 numbers, and the amount of batches can be lowered.
* The amount of epochs can be chosen freely.
*/
const int lInSize = 784, l1Size = 16, l2Size = 16, lOutSize = 10;
const int batchCount = 3795, epochcount = 10;

queue q;
float* bias1;
float* bias2;
float* biasOut;
float* weightIn;
float* weight1;
float* weight2;
float* activationIn;
float* activation1;
float* activation2;
float* activationOut;
float* sigmoidedIn;
float* sigmoided1;
float* sigmoided2;
float* sigmoidedOut;
bool training;

// ChatGPT made a device-based random value generator, which is applied over a list of values. A unique randID is also inputed, for variance over lists.
inline void randomParamaterInit(float* paramaters, int size, int randID, queue& q, bool isBias) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            float randVal = float(((i + randID * 10000000) ^ 0xA3B1C2D3) * 1664525 + 1013904223 & 0xFFFFFF) / float(0xFFFFFF); // Random value between 0 and 1
            paramaters[i] = randVal * 2 - 1;
        });
    });
}

inline void imageActivationInit(float* activations, int size, queue& q) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            activations[i] = 10000;
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

inline void sigmoidLayer(float* activationLayer, float* sigmoidLayer, int size) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            sigmoidLayer[i] = 1 / (1 + std::exp(-activationLayer[i]));
        });
    });
    q.wait();
}

inline void forwardPropogateLayer(float* prevSigmoidedLayer, int prevSize, float* weights, float* biases, float* curLayer, int curSize) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(curSize), [=](id<1> i) {
            curLayer[i] = biases[i];
            for (int j = 0; j < prevSize; j++) {
                float lastSigmoidedActivation = prevSigmoidedLayer[j];
                float connectionWeight = weights[i * prevSize + j];
                curLayer[i] += lastSigmoidedActivation * connectionWeight;
            }
        });
    });
    q.wait();
}

inline void forwardPropogate() {
    sigmoidLayer(activationIn, sigmoidedIn, lInSize);
    forwardPropogateLayer(sigmoidedIn, lInSize, weightIn, bias1, activation1, l1Size);
    sigmoidLayer(activation1, sigmoided1, l1Size);
    forwardPropogateLayer(sigmoided1, l1Size, weight1, bias2, activation2, l2Size);
    sigmoidLayer(activation2, sigmoided2, l2Size);
    forwardPropogateLayer(sigmoided2, l2Size, weight2, biasOut, activationOut, lOutSize);
    sigmoidLayer(activationOut, sigmoidedOut, lOutSize);
}

inline void train() {
    vector<vector<vector<float>>> images(10, vector<vector<float>>(batchCount, vector<float>(784)));
    for (int i = 0; i < 10; i++) {
        int curCount = 0;
        for (const auto& entry : directory_iterator("trainingSet\\" + to_string(i))) {
            int width, height, channels;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 1);
            if (entry.path().string() == "trainingSet\\0\\img_1.jpg") {
                for (int x = 0; x < width; x++) {
                    for (int y = 0; y < height; y++) {
                        float value = data[x * width + y];
                        value /= 255;
                        images[i][curCount][x * width + y] = value;
                    }
                }
            }
            stbi_image_free(data);
            curCount++;
            if (curCount == batchCount) break;
        }
    }

    randomParamaterInit(bias1, l1Size, 1, q, true);
    randomParamaterInit(bias2, l2Size, 2, q, true);
    randomParamaterInit(biasOut, lOutSize, 3, q, true);
    randomParamaterInit(weightIn, lInSize * l1Size, 4, q, false);
    randomParamaterInit(weight1, l1Size * l2Size, 5, q, false);
    randomParamaterInit(weight2, l2Size * lOutSize, 6, q, false);
    q.wait();
    
    forwardPropogate();
    float sigmoidedOutHost[lOutSize];
    q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float));
    q.wait();
    for (int i = 0; i < lOutSize; i++) cout << sigmoidedOutHost[i] << " ";
    cout << "\n-----------\n";
    for (int batchID = 0; batchID < 1; batchID++) {
        for (int i = 0; i < 10; i++) {
            imageActivationInit(activationIn, lInSize, q);
            
            forwardPropogate();
            float sigmoidedOutHost[lOutSize];
            q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float));
            q.wait();
            for (int i = 0; i < lOutSize; i++) cout << sigmoidedOutHost[i] << " ";
            cout << "\n-----------\n";
        }
    }
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
    activationIn = malloc_device<float>(lInSize, q);
    activation1 = malloc_device<float>(l1Size, q);
    activation2 = malloc_device<float>(l2Size, q);
    activationOut = malloc_device<float>(lOutSize, q);
    sigmoidedIn = malloc_device<float>(lInSize, q);
    sigmoided1 = malloc_device<float>(l1Size, q);
    sigmoided2 = malloc_device<float>(l2Size, q);
    sigmoidedOut = malloc_shared<float>(lOutSize, q); // This is shared, so the data can be retrieved from the device to be read when testing
    
    if (training) train();
    else test();
    
    float bias1Host[l1Size], bias2Host[l2Size], biasOutHost[lOutSize], 
        weightInHost[lInSize * l1Size], weight1Host[l1Size * l2Size], weight2Host[l2Size * lOutSize], 
        activationInHost[lInSize], activation1Host[l1Size], activation2Host[l2Size], activationOutHost[lOutSize],
        sigmoidedInHost[lInSize], sigmoided1Host[l1Size], sigmoided2Host[l2Size], sigmoidedOutHost[lOutSize];
    q.memcpy(bias1Host, bias1, l1Size * sizeof(float));
    q.memcpy(bias2Host, bias2, l2Size * sizeof(float));
    q.memcpy(biasOutHost, biasOut, lOutSize * sizeof(float));
    q.memcpy(weightInHost, weightIn, lInSize * l1Size * sizeof(float));
    q.memcpy(weight1Host, weight1, l1Size * l2Size * sizeof(float));
    q.memcpy(weight2Host, weight2, l2Size * lOutSize * sizeof(float));
    q.memcpy(activationInHost, activationIn, lInSize * sizeof(float));
    q.memcpy(activation1Host, activation1, l1Size * sizeof(float));
    q.memcpy(activation2Host, activation2, l2Size * sizeof(float));
    q.memcpy(activationOutHost, activationOut, lOutSize * sizeof(float));
    q.memcpy(sigmoidedInHost, sigmoidedIn, lInSize * sizeof(float));
    q.memcpy(sigmoided1Host, sigmoided1, l1Size * sizeof(float));
    q.memcpy(sigmoided2Host, sigmoided2, l2Size * sizeof(float));
    q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float));
    q.wait();

    /*
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
    */
    for (int i = 0; i < lInSize; i++) cout << activationInHost[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < l1Size; i++) cout << activation1Host[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < l2Size; i++) cout << activation2Host[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < lOutSize; i++) cout << activationOutHost[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < lInSize; i++) cout << sigmoidedInHost[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < l1Size; i++) cout << sigmoided1Host[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < l2Size; i++) cout << sigmoided2Host[i] << " ";
    cout << "\n-----------\n";
    for (int i = 0; i < lOutSize; i++) cout << sigmoidedOutHost[i] << " ";
    cout << "\n-----------\n";

    if (training) {
        ofstream paramOut("paramaters.txt");
        for (float i : bias1Host) paramOut << i << " ";
        for (float i : bias2Host) paramOut << i << " ";
        for (float i : biasOutHost) paramOut << i << " ";
        for (float i : weightInHost) paramOut << i << " ";
        for (float i : weight1Host) paramOut << i << " ";
        for (float i : weight2Host) paramOut << i << " ";
    }
}
