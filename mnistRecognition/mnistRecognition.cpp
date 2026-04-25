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
#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include "VC_IncludePath\stb_image.h"
using namespace sycl;
using namespace std;
using namespace filesystem;
using namespace chrono;

/* 
    Context and instructions:
        The neural net is fixed at 4 layers(including input and output).
        The input neuron count should be fixed for 28x28 MNIST images, and output neuron count should be fixed for numbers 0->9.
        However, the neuron count of the intermediate layers can be chosen freely.
        Furthermore, each batch contains each of the 10 numbers, and the amount of batches can be lowered.
        The amount of epochs can be chosen freely.
*/
const int lInSize = 784, l1Size = 128, l2Size = 128, lOutSize = 10;
const int batchCount = 3795, epochCount = 50;
const float initialLearningRate = 0.1, finishLearningRate = 0.001;

// Do not adjust anything below
bool training, debugging, singleEpoch, findingAccuracy;
constexpr int lossType = 2; 
/* Loss methods:
* 0 = Summed squared errors (SSE)
* 1 = Mean squared errors (MSE)
* 2 = Binary Cross-entropy (BCE)
*/
const float learningRateDecay = std::pow(finishLearningRate / initialLearningRate, 1.0f / (batchCount * epochCount));

//NOTE TO SELF for training 10 epochs: 1 epoch of 3795 batches of 10 took ~126.2s = ~30 batches per second = ~301 images per second = ~3.33 milliseconds per image

//GPU memory allocations
queue q;
float *losses, *learningRate;
float *trainingImages;
float* totalLoss;

float *bias1, *bias2, *biasOut;
float *bias1Derivative, *bias2Derivative, *biasOutDerivative;

float *weightIn, *weight1, *weight2;
float *weightInDerivative, *weight1Derivative, *weight2Derivative;

float *activationIn, *activation1, *activation2, *activationOut;
// No activation derivative, as this is same as bias derivative

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

// FORWARD PROPOGATION --------------------------------------------------------------------------------------
// Calculate the layer of sigmoided activations
inline event sigmoidLayer(float* activationLayer, float* sigmoidLayer, int size, event priorEvent) {
    return q.submit([&](handler& h) {
        h.depends_on(priorEvent);
        h.parallel_for(range<1>(size), [=](id<1> i) {
            sigmoidLayer[i] = 1 / (1 + sycl::exp(-activationLayer[i]));
        });
    });
}

// Calculate unsigmoided activations of a layer, based on the previous layer and relevant paramaters
inline event forwardPropogateLayer(float* prevSigmoidedLayer, int prevSize, float* weights, float* biases, float* curLayer, int curSize, event priorEvent) {
    return q.submit([&](handler& h) {
        h.depends_on(priorEvent);
        h.parallel_for(range<1>(curSize), [=](id<1> i) {
            curLayer[i] = biases[i];
            for (int j = 0; j < prevSize; j++) {
                float lastSigmoidedActivation = prevSigmoidedLayer[j];
                float connectionWeight = weights[i * prevSize + j];
                curLayer[i] += lastSigmoidedActivation * connectionWeight;
            }
        });
    });
}

// Run functions for forward propogation, to calculate activations of output layer
inline event forwardPropogate(int i) {
    event s1 = sigmoidLayer(activationIn + i * lInSize, sigmoidedIn + i * lInSize, lInSize, {});
    event f1 = forwardPropogateLayer(sigmoidedIn + i * lInSize, lInSize, weightIn, bias1, activation1 + i * l1Size, l1Size, s1);
    event s2 = sigmoidLayer(activation1 + i * l1Size, sigmoided1 + i * l1Size, l1Size, f1);
    event f2 = forwardPropogateLayer(sigmoided1 + i * l1Size, l1Size, weight1, bias2, activation2 + i * l2Size, l2Size, s2);
    event s3 = sigmoidLayer(activation2 + i * l2Size, sigmoided2 + i * l2Size, l2Size, f2);
    event f3 = forwardPropogateLayer(sigmoided2 + i * l2Size, l2Size, weight2, biasOut, activationOut + i * lOutSize, lOutSize, s3);
    return sigmoidLayer(activationOut + i * lOutSize, sigmoidedOut + i * lOutSize, lOutSize, f3);
}

// BACKWARD PROPOGATION + GRADIENT DESCENT --------------------------------------------------------------------------------------

inline void adjustLearningRate(float* LR) {
    const float decay = learningRateDecay;
    q.submit([&](handler& h) {
        h.single_task([=]() {
            LR[0] = LR[0] * decay;
        });
    }).wait();
}

inline event deriveOutputSigmoidLayer(float* sigmoidOut, float* sigmoidOutDerivative, int expected, event forwardPropEvent) {
    return q.submit([&](handler& h) {
        h.depends_on(forwardPropEvent);
        h.parallel_for(range<1>(lOutSize), [=](id<1> i) {
            float isExpected = (i == expected) ? 1 : 0;
            sigmoidOutDerivative[i] =
                (lossType == 0) ? 2 * (sigmoidOut[i] - isExpected)
                : (lossType == 1) ? (sigmoidOut[i] - isExpected) / 5
                : (lossType == 2) ? sigmoidOut[i] - isExpected
                : 0;
        });
    });
}

inline event deriveBiases(float* sigmoided, float* biasDerivative, float* sigmoidedDerivative, int listSize, event sigmoidDeriveEvent) {
    return q.submit([&](handler& h) {
        h.depends_on(sigmoidDeriveEvent);
        h.parallel_for(range<1>(listSize), [=](id<1> i) {
            //biasDerivative[i] = sigmoidedDerivative[i] * sycl::exp(-activation[i]) / sycl::pow(1 + sycl::exp(-activation[i]), 2);
            biasDerivative[i] = sigmoidedDerivative[i] * sigmoided[i] * (1 - sigmoided[i]);
        });
    });
}

inline void deriveWeights(float* prevSigmoided, float* biasDerivative, float* weightDerivative, int prevListSize, int listSize, event biasDeriveEvent) {
    q.submit([&](handler& h) {
        h.depends_on(biasDeriveEvent);
        h.parallel_for(range<2>(prevListSize, listSize), [=](id<2> i) {
            weightDerivative[i[1] * prevListSize + i[0]] = biasDerivative[i[1]] * prevSigmoided[i[0]];
        });
    });
}

inline event deriveSigmoid(float* prevSigmoided, float* biasDerivative, float* weight, float* prevSigmoidedDerivative, int prevListSize, int listSize, event biasDeriveEvent) {
    return q.submit([&](handler& h) {
        h.depends_on(biasDeriveEvent);
        h.parallel_for(range<1>(prevListSize), [=](id<1> i) {
            prevSigmoidedDerivative[i] = 0;
            for (int j = 0; j < listSize; j++) {
                prevSigmoidedDerivative[i] += biasDerivative[j] * weight[j * prevListSize + i];
            }
        });
    });
}

inline void backwardPropogate(int i, event forwardPropEvent) {
    event s4 = deriveOutputSigmoidLayer(sigmoidedOut + i * lOutSize, sigmoidedOutDerivative + i * lOutSize, i, forwardPropEvent);
    event b4 = deriveBiases(sigmoidedOut + i * lOutSize, biasOutDerivative + i * lOutSize, sigmoidedOutDerivative + i * lOutSize, lOutSize, s4);
    deriveWeights(sigmoided2 + i * l2Size, biasOutDerivative + i * lOutSize, weight2Derivative + i * l2Size * lOutSize, l2Size, lOutSize, b4);

    event s3 = deriveSigmoid(sigmoided2 + i * l2Size, biasOutDerivative + i * lOutSize, weight2, sigmoided2Derivative + i * l2Size, l2Size, lOutSize, b4);
    event b3 = deriveBiases(sigmoided2 + i * l2Size, bias2Derivative + i * l2Size, sigmoided2Derivative + i * l2Size, l2Size, s3);
    deriveWeights(sigmoided1 + i * l1Size, bias2Derivative + i * l2Size, weight1Derivative + i * l1Size * l2Size, l1Size, l2Size, b3);

    event s2 = deriveSigmoid(sigmoided1 + i * l1Size, bias2Derivative + i * l2Size, weight1, sigmoided1Derivative + i * l1Size, l1Size, l2Size, b3);
    event b2 = deriveBiases(sigmoided1 + i * l1Size, bias1Derivative + i * l1Size, sigmoided1Derivative + i * l1Size, l1Size, s2);
    deriveWeights(sigmoidedIn + i * lInSize, bias1Derivative + i * l1Size, weightInDerivative + i * lInSize * l1Size, lInSize, l1Size, b2);
}

inline void paramaterNegateDerivatives(float* paramaters, float* derivatives, float* LR, int listSize) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(listSize), [=](id<1> i) {
            for (int number = 0; number < 10; number++) {
                // Not divided for average, cause learning rate will adjust anyway
                paramaters[i] -= derivatives[number * listSize + i] * LR[0];
            }
        });
    });
}

// DEBUG --------------------------------------------------------------------------------------
inline void initTo1(float* values, int listSize) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(listSize), [=](id<1> i) {
            values[i] = 1;
        });
    });
}

inline void initTo0(float* values, int listSize) {
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(listSize), [=](id<1> i) {
            values[i] = 0;
        });
    });
}

inline event computeLoss(float* lossVal, float* outputLayer, int layerSize, int targetNumber, bool reset, event forwardPropogateEvent) {
    if (reset) q.memset(lossVal, 0, sizeof(float)).wait();
    return q.submit([&](handler& h) {
        h.depends_on(forwardPropogateEvent);
        h.parallel_for(range<1>(layerSize), [=](id<1> i) {
            float expectedActivation = (i == targetNumber) ? 1 : 0;
            float neuronLoss = 
                (lossType == 0) ? sycl::pow(expectedActivation - outputLayer[i], 2)
                : (lossType == 1) ? sycl::pow(expectedActivation - outputLayer[i], 2) / 10
                : (lossType == 2) ? -expectedActivation * sycl::log(outputLayer[i] + 0.0000001f) - (1 - expectedActivation) * sycl::log(1 - outputLayer[i] + 0.0000001f)
                : 0;
            //Atomic prevents incorrect sumation for parallel, relaxed means can be in any order, on device memory globally
            atomic_ref<float, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomicSum(*lossVal);
            atomicSum.fetch_add(neuronLoss);
        });
    });
}


// TRAINING --------------------------------------------------------------------------------------
inline void train() {
    auto start = high_resolution_clock::now();
    // 10X size to allow parallel compute
    activationIn = malloc_device<float>(10 * lInSize, q);
    activation1 = malloc_device<float>(10 * l1Size, q);
    activation2 = malloc_device<float>(10 * l2Size, q);
    activationOut = malloc_device<float>(10 * lOutSize, q);
    sigmoidedIn = malloc_device<float>(10 * lInSize, q);
    sigmoided1 = malloc_device<float>(10 * l1Size, q);
    sigmoided2 = malloc_device<float>(10 * l2Size, q);
    sigmoidedOut = malloc_device<float>(10 * lOutSize, q);
    // Derivatives and stuff
    bias1Derivative = malloc_device<float>(10 * l1Size, q);
    bias2Derivative = malloc_device<float>(10 * l2Size, q);
    biasOutDerivative = malloc_device<float>(10 * lOutSize, q);
    weightInDerivative = malloc_device<float>(10 * lInSize * l1Size, q);
    weight1Derivative = malloc_device<float>(10 * l1Size * l2Size, q);
    weight2Derivative = malloc_device<float>(10 * l2Size * lOutSize, q);
    sigmoidedInDerivative = malloc_device<float>(10 * lInSize, q);
    sigmoided1Derivative = malloc_device<float>(10 * l1Size, q);
    sigmoided2Derivative = malloc_device<float>(10 * l2Size, q);
    sigmoidedOutDerivative = malloc_device<float>(10 * lOutSize, q);
    // Single-value for computing loss
    losses = malloc_device<float>(10, q);
    learningRate = malloc_device<float>(1, q);
    q.memcpy(learningRate, &initialLearningRate, sizeof(float)).wait();
    trainingImages = malloc_device<float>(10 * batchCount * lInSize, q);
    totalLoss = malloc_device<float>(1, q);

    // Collect and store arrays of images
    float* images = new float[10 * batchCount * lInSize];
    for (int i = 0; i < 10; i++) {
        int curBatch = 0;
        for (const auto& entry : directory_iterator("trainingSet\\" + to_string(i))) {
            int width, height, channels;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 1);
            float* destination = images + (curBatch * 10 + i) * lInSize;
            for (int x = 0; x < width * height; x++) destination[x] = data[x] / 255.0f;
            stbi_image_free(data);
            curBatch++;
            if (curBatch == batchCount) break;
        }
    }
    q.memcpy(trainingImages, images, 10 * batchCount * lInSize * sizeof(float)).wait();
    delete images;

    // Initialise all paramaters randomly (MAYBE CHANGE FOR A BETTER ALTERNATIVE)
    randomParamaterInit(bias1, l1Size, 1, q, true);
    randomParamaterInit(bias2, l2Size, 2, q, true);
    randomParamaterInit(biasOut, lOutSize, 3, q, true);
    randomParamaterInit(weightIn, lInSize * l1Size, 4, q, false);
    randomParamaterInit(weight1, l1Size * l2Size, 5, q, false);
    randomParamaterInit(weight2, l2Size * lOutSize, 6, q, false);
    q.wait();
    auto stop = high_resolution_clock::now();
    cerr << "Data movement and random paramater initialisation took " << duration_cast<seconds>(stop - start).count() << " s\n";
    start = stop;
    
    if (debugging) {
        float biasOutHost[lOutSize];
        q.memcpy(biasOutHost, biasOut, lOutSize * sizeof(float)).wait();
        for (int i = 0; i < lOutSize; i++) cout << biasOutHost[i] << " ";
        cout << " initial bias values\n-----------\n";
    }

    vector<int> batchOrder(batchCount);
    for (int i = 0; i < batchCount; i++) batchOrder[i] = i;
    for (int epoch = 0; epoch < ((singleEpoch) ? 1 : epochCount); epoch++) {
        shuffle(batchOrder.begin(), batchOrder.end(), default_random_engine(epoch));
        for (int batchID : batchOrder) {
            if (batchID && debugging) continue;
            activationIn = trainingImages + batchID * 10 * lInSize;
            for (int i = 0; i < 10; i++) {
                auto forwardPropEvent = forwardPropogate(i);
                backwardPropogate(i, forwardPropEvent);
            }
            q.wait();
            // Negated for gradient descent
            paramaterNegateDerivatives(weightIn, weightInDerivative, learningRate, lInSize * l1Size);
            paramaterNegateDerivatives(weight1, weight1Derivative, learningRate, l1Size * l2Size);
            paramaterNegateDerivatives(weight2, weight2Derivative, learningRate, l2Size * lOutSize);
            paramaterNegateDerivatives(bias1, bias1Derivative, learningRate, l1Size);
            paramaterNegateDerivatives(bias2, bias2Derivative, learningRate, l2Size);
            paramaterNegateDerivatives(biasOut, biasOutDerivative, learningRate, lOutSize);
            q.wait();
            adjustLearningRate(learningRate);
            if (batchID == 0 && findingAccuracy) {
                for (int i = 0; i < 10; i++) {
                    float sigmoidedOutHost[lOutSize];
                    q.memcpy(sigmoidedOutHost, sigmoidedOut + i * lOutSize, lOutSize * sizeof(float)).wait();
                    for (int j = 0; j < lOutSize; j++) cout << sigmoidedOutHost[j] << " ";
                    cout << " Outputs for " << i << "\n---------- - \n";
                }
                for (int i = 0; i < 10; i++) computeLoss(losses + i, sigmoidedOut + i * lOutSize, lOutSize, i, true, {});
                float lossHost[10];
                q.memcpy(lossHost, losses, 10 * sizeof(float)).wait();
                for (int i = 0; i < 10; i++) cout << "Loss = " << lossHost[i] << " ";
                cout << "\n";
            }
            if (debugging) {
                float biasOutHost[lOutSize];
                q.memcpy(biasOutHost, biasOut, lOutSize * sizeof(float)).wait();
                for (int i = 0; i < lOutSize; i++) cout << biasOutHost[i] << " ";
                cout << " new bias values\n-----------\n";
            }
        }
        cout << "Finished epoch " << epoch + 1 << "\n";
        if (!findingAccuracy) continue;
        q.memset(totalLoss, 0, sizeof(float)).wait();
        for (int batchID : batchOrder) {
            if (batchID && debugging) continue;
            activationIn = trainingImages + batchID * 10 * lInSize;
            for (int i = 0; i < 10; i++) {
                event forwardPropogateEvent = forwardPropogate(i);
                computeLoss(totalLoss, sigmoidedOut + i * lOutSize, lOutSize, i, false, forwardPropogateEvent);
            }
        }
        q.wait();
        float totalLossHost[1];
        q.memcpy(totalLossHost, totalLoss, sizeof(float));
        cout << "Accuracy = " << totalLossHost[0] << "\n";
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
    stop = high_resolution_clock::now();
    cout << "Training took " << duration_cast<seconds>(stop - start).count() << " s\n";
}

// TESTING --------------------------------------------------------------------------------------
inline void test() {
    activationIn = malloc_device<float>(lInSize, q);
    activation1 = malloc_device<float>(l1Size, q);
    activation2 = malloc_device<float>(l2Size, q);
    activationOut = malloc_device<float>(lOutSize, q);
    sigmoidedIn = malloc_device<float>(lInSize, q);
    sigmoided1 = malloc_device<float>(l1Size, q);
    sigmoided2 = malloc_device<float>(l2Size, q);
    sigmoidedOut = malloc_device<float>(lOutSize, q);

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

    float* activationInHost = new float[784];
    int width, height, channels;
    auto lastWriteTime = last_write_time("testSample.jpg"); //Make a null value for first time
    while (true) {
        this_thread::sleep_for(milliseconds(200));
        auto currentWriteTime = last_write_time("testSample.jpg");
        if (currentWriteTime == lastWriteTime) continue;
        unsigned char* data = stbi_load("testSample.jpg", &width, &height, &channels, 1);
        if (!data) continue;
        lastWriteTime = currentWriteTime;
        for (int x = 0; x < width * height; x++) activationInHost[x] = data[x] / 255.0f;
        stbi_image_free(data);

        q.memcpy(activationIn, activationInHost, 784 * sizeof(float)).wait();
        forwardPropogate(0);
        q.wait();

        float sigmoidedOutHost[lOutSize];
        q.memcpy(sigmoidedOutHost, sigmoidedOut, lOutSize * sizeof(float)).wait();
        float sum = 0;
        for (int j = 0; j < lOutSize; j++) sum += sigmoidedOutHost[j];
        for (int j = 0; j < lOutSize; j++) sigmoidedOutHost[j] = std::round(sigmoidedOutHost[j] / sum * 100);
        cout << "\033[3;0H";
        for (int j = 0; j < lOutSize; j++) cout << j << " = " << sigmoidedOutHost[j] << "% probability\n";
    }
}


// INITIALISATION FUNCTION --------------------------------------------------------------------------------------
int main() {
    q = queue(default_selector_v);
    cout << q.get_device().get_info<info::device::name>() << "\n";
    cout << "(1 = Training, 0 = Testing): ";
    cin >> training;
    if (training) {
        cout << "(1 = Debugging, 0 = Not Debugging): ";
        cin >> debugging;
        cout << "(1 = Single epoch, 0 = All epochs): ";
        cin >> singleEpoch;
        cout << "(1 = Finding accuracy, 0 = Not finding accuracy): ";
        cin >> findingAccuracy;
    }
    
    // Allocates memory for paramaters (weights/biases)
    bias1 = malloc_device<float>(l1Size, q);
    bias2 = malloc_device<float>(l2Size, q);
    biasOut = malloc_device<float>(lOutSize, q);
    weightIn = malloc_device<float>(lInSize * l1Size, q);
    weight1 = malloc_device<float>(l1Size * l2Size, q);
    weight2 = malloc_device<float>(l2Size * lOutSize, q);
    
    if (training) train();
    else test();
}