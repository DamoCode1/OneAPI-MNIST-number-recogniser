// MNIST dataset = https://www.kaggle.com/datasets/amineipad/mnist-dataset/data
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <iostream>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;
using namespace std;

int main() {
    auto selector = default_selector_v;

    queue q(selector);
    cout << q.get_device().get_info<info::device::name>();

    return 0;
}
