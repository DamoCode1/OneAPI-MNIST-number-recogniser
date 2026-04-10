// MNIST dataset = https://www.kaggle.com/datasets/amineipad/mnist-dataset/data
// Image counts per number 0->9 =  4132, 4684, 4177, 4351, 4072, 3795, 4137, 4401, 4063, 4188 (Minimum = 3795, so 3795 batches with each number) 
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// INITIALISATION --------------------------------------------------------------------------------------
#include <sycl/sycl.hpp>
#include <iostream>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#include "VC_IncludePath\stb_image.h"
using namespace sycl;
using namespace std;
using namespace filesystem;

/* 
    Context and instructions:
        The neural net is fixed at 4 layers(including input and output).
        The input neuron count should be fixed for 28x28 MNIST images, and output neuron count should be fixed for numbers 0->9.
        However, the neuron count of the intermediate layers can be chosen freely.
        Furthermore, each batch contains each of the 10 numbers, and the amount of batches can be lowered.
        The amount of epochs can be chosen freely.
*/
const int lInSize = 784, l1Size = 16, l2Size = 16, lOutSize = 10;
const int batchCount = 3795, epochCount = 1;
const float initialLearningRate = 1, finishLearningRate = 0.01;

// Do not adjust anything below
bool training, debugging;
const float learningRateDecay = std::pow(finishLearningRate / initialLearningRate, 1.0f / (batchCount * epochCount));

//NOTE TO SELF: 1 epoch of 3795 batches of 10 took ~3 minutes = ~21 batches per second = ~210 images per second = ~4.7 milliseconds per image

//GPU memory allocations
queue q;
float *loss, *learningRate;
float *trainingImages;

float *bias1, *bias2, *biasOut;
float *bias1Derivative, *bias2Derivative, *biasOutDerivative;
float *bias1DerivativeSum, *bias2DerivativeSum, *biasOutDerivativeSum;

float *weightIn, *weight1, *weight2;
float *weightInDerivative, *weight1Derivative, *weight2Derivative;
float *weightInDerivativeSum, *weight1DerivativeSum, *weight2DerivativeSum;

float *activationIn, *activation1, *activation2, *activationOut;
float *activationInDerivative, *activation1Derivative, *activation2Derivative, *activationOutDerivative;

float *sigmoidedIn, *sigmoided1, *sigmoided2, *sigmoidedOut;
float *sigmoidedInDerivative, *sigmoided1Derivative, *sigmoided2Derivative, *sigmoidedOutDerivative;


// ChatGPT made a device-based random value generator, which is applied over a list of values. A unique randID is also inputed, for variance over lists.
inline void randomParamaterInit(float* paramaters, int size, int randID, queue& q, bool isBias) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            float randVal = float(((i + randID * 10000000) ^ 0xA3B1C2D3) * 1664525 + 1013904223 & 0xFFFFFF) / float(0xFFFFFF); // Random value between 0 and 1
            paramaters[i] = randVal * 2 - 1;
        });
    });
}

inline void storeImage(float* images, unsigned char* data, int width, int height, int number, int curBatch) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(width * height), [=](id<1> i) {
            images[((number * 10) + curBatch) * batchCount + i] = data[i] / 255.0f;
        });
    });
}

// FORWARD PROPOGATION --------------------------------------------------------------------------------------
// Calculate the layer of sigmoided activations
inline void sigmoidLayer(float* activationLayer, float* sigmoidLayer, int size) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
            sigmoidLayer[i] = 1 / (1 + sycl::exp(-activationLayer[i]));
        });
    }).wait();
}

// Calculate unsigmoided activations of a layer, based on the previous layer and relevant paramaters
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
    }).wait();
}

// Calculate MSE of output layer compared to the expected output
inline void computeLoss(float* lossVal, float* outputLayer, int layerSize, int targetNumber) {
    q.memset(lossVal, 0, sizeof(float)).wait();
    q.submit([&](handler& h) {
        local_accessor<float, 1> lossSum(range<1>(1), h);
        h.parallel_for(range<1>(layerSize), [=](id<1> i) {
            float expectedActivation = (i == targetNumber) ? 1 : 0;
            float mse = sycl::pow(expectedActivation - outputLayer[i], 2);
            //Atomic prevents incorrect sumation for parallel, relaxed means can be in any order, on device memory globally
            atomic_ref<float, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomicSum(*lossVal);
            atomicSum.fetch_add(mse);
        });
    }).wait();
}

// Run functions for forward propogation, to calculate activations of output layer
inline void forwardPropogate() {
    sigmoidLayer(activationIn, sigmoidedIn, lInSize);
    forwardPropogateLayer(sigmoidedIn, lInSize, weightIn, bias1, activation1, l1Size);
    sigmoidLayer(activation1, sigmoided1, l1Size);
    forwardPropogateLayer(sigmoided1, l1Size, weight1, bias2, activation2, l2Size);
    sigmoidLayer(activation2, sigmoided2, l2Size);
    forwardPropogateLayer(sigmoided2, l2Size, weight2, biasOut, activationOut, lOutSize);
    sigmoidLayer(activationOut, sigmoidedOut, lOutSize);
}

// BACKWARD PROPOGATION + GRADIENT DESCENT --------------------------------------------------------------------------------------
inline void initTo1(float* values, int listSize) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(listSize), [=](id<1> i) {
            values[i] = 1;
        });
    });
}

inline void adjustLearningRate(float* LR) {
    const float decay = learningRateDecay;
    q.submit([&](handler& h) {
        h.single_task([=]() {
            LR[0] = LR[0] * decay;
        });
    }).wait();
}

inline void backwardPropogate() {
    
}

// TRAINING --------------------------------------------------------------------------------------
inline void train() {
    // Single-value for computing loss
    loss = malloc_device<float>(1, q);
    learningRate = malloc_device<float>(1, q);
    q.memcpy(learningRate, &initialLearningRate, sizeof(float)).wait();
    trainingImages = malloc_device<float>(10 * batchCount * lInSize, q);

    // Collect and store arrays of images
    float* images = new float[10 * batchCount * lInSize];
    for (int i = 0; i < 10; i++) {
        int curBatch = 0;
        for (const auto& entry : directory_iterator("trainingSet\\" + to_string(i))) {
            int width, height, channels;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 1);
            float* destination = images + (i * batchCount + curBatch) * lInSize;
            for (int x = 0; x < width * height; x++) destination[x] = data[x] / 255.0f;
            stbi_image_free(data);
            curBatch++;
            if (curBatch == batchCount) break;
        }
    }
    q.memcpy(trainingImages, images, 10 * batchCount * lInSize * sizeof(float)).wait();

    // Initialise all paramaters randomly (MAYBE CHANGE FOR A BETTER ALTERNATIVE)
    randomParamaterInit(bias1, l1Size, 1, q, true);
    randomParamaterInit(bias2, l2Size, 2, q, true);
    randomParamaterInit(biasOut, lOutSize, 3, q, true);
    randomParamaterInit(weightIn, lInSize * l1Size, 4, q, false);
    randomParamaterInit(weight1, l1Size * l2Size, 5, q, false);
    randomParamaterInit(weight2, l2Size * lOutSize, 6, q, false);
    q.wait();
    
    //Test forward propogation (TO REMOVE)
    forwardPropogate();
    computeLoss(loss, sigmoidedOut, lOutSize, 0);
    if (debugging) {
        float sigmoidedOutHost[lOutSize], lossHost[1];
        q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float));
        q.memcpy(lossHost, loss, sizeof(float));
        q.wait();
        for (int i = 0; i < lOutSize; i++) cout << sigmoidedOutHost[i] << " ";
        cout << "\nLoss = " << lossHost[0];
        cout << "\n-----------\n";
    }

    for (int epoch = 0; epoch < epochCount; epoch++) {
        for (int batchID = 0; batchID < ((debugging) ? 1 : batchCount); batchID++) {
            for (int i = 0; i < 10; i++) {
                activationIn = trainingImages + (i * batchCount + batchID) * lInSize;

                forwardPropogate();
                computeLoss(loss, sigmoidedOut, lOutSize, i);
                backwardPropogate();
                if (debugging) {
                    float sigmoidedOutHost[lOutSize], lossHost[1];
                    q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float));
                    q.memcpy(lossHost, loss, sizeof(float));
                    q.wait();
                    for (int i = 0; i < lOutSize; i++) cout << sigmoidedOutHost[i] << " ";
                    cout << "\nLoss = " << lossHost[0];
                    cout << "\n-----------\n";
                }
            }
        }
    }

    //Collects paramaters from device
    float bias1Host[l1Size], bias2Host[l2Size], biasOutHost[lOutSize],
        weightInHost[lInSize * l1Size], weight1Host[l1Size * l2Size], weight2Host[l2Size * lOutSize];
    q.memcpy(bias1Host, bias1, l1Size * sizeof(float));
    q.memcpy(bias2Host, bias2, l2Size * sizeof(float));
    q.memcpy(biasOutHost, biasOut, lOutSize * sizeof(float));
    q.memcpy(weightInHost, weightIn, lInSize * l1Size * sizeof(float));
    q.memcpy(weight1Host, weight1, l1Size * l2Size * sizeof(float));
    q.memcpy(weight2Host, weight2, l2Size * lOutSize * sizeof(float));
    q.wait();
    // Stores the collected paramaters in a text file, to be read for testing
    ofstream paramOut("paramaters.txt");
    for (float i : bias1Host) paramOut << i << " ";
    for (float i : bias2Host) paramOut << i << " ";
    for (float i : biasOutHost) paramOut << i << " ";
    for (float i : weightInHost) paramOut << i << " ";
    for (float i : weight1Host) paramOut << i << " ";
    for (float i : weight2Host) paramOut << i << " ";
    cout << "Training Completed!\n";
}

// TESTING --------------------------------------------------------------------------------------
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


// INITIALISATION FUNCTION --------------------------------------------------------------------------------------
int main() {
    q = queue(default_selector_v);
    cout << q.get_device().get_info<info::device::name>() << "\n";
    cout << "(1 = Training, 0 = Testing): ";
    cin >> training;
    cout << "(1 = Debugging, 0 = Not Debugging): ";
    cin >> debugging;
    
    // Allocates memory for activations (sigmoided/unprocessed), and paramaters (weights/biases). See other memory allocations specific to training in the train function.
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
    sigmoidedOut = malloc_device<float>(lOutSize, q);
    
    if (training) train();
    else test();
}