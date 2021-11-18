# YOLOv5-ncnn-arm
An ncnn implementation of YOLOv5 on ARM devices, capable of using GPU to accelerate inference

### Environment

- Ubuntu 18.04 (x86_64)
- Ubuntu 16.04 (aarch64)
- OpenCV 3.2.0
- CMake 3.10.0

### Getting Started with CPU version

1. The compilation of the project should be on the ARM device.

2. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

3. ```
   cd YOLOv5ncnn
   ```

4. Edit "CMakeLists.txt" to configure correctly.

5. Compile and run.

   ```shell
   cd build
   cmake ..
   make
   ./../bin/YOLOv5ncnn
   ```

### Compile ncnn-ARM by yourself

1. The compilation of ncnn should be on the x86 device.

2. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

3. Install protobuf.

   ```shell
   sudo apt install protobuf-compiler libprotobuf-dev 
   ```

4. Download source code of ncnn from https://github.com/Tencent/ncnn/releases.

   ```shell
   unzip ncnn-master.zip
   ```

5. Download [gcc-arm-toolchain](https://developer.arm.com/-/media/Files/downloads/gnu-a/8.2-2018.11/gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu.tar.xz?revision=7a60a425-1aa0-43f5-b9db-1af71bffadc6&la=en) and add to environment variables.

   ```shell
   tar -zxvf gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu.tar.xz
   gedit ~/.bashrc
   export PATH=$PATH:/home/username/gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu/bin
   source ~/.bashrc
   ```

6. Compile ncnn.

   ```shell
   cd ncnn
   mkdir -p build-aarch64-linux
   cd build-aarch64-linux
   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake –DANDROID=ON ..
   make -j8
   make install
   ```

### Getting Started with GPU version

1. The compilation of the project should be on the ARM device.

2. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

3. ```
   cd YOLOv5ncnn-vulkan
   ```

4. Edit "CMakeLists.txt" to configure correctly.

5. Compile and run.

   ```shell
   cd build
   cmake ..
   make
   ./../bin/YOLOv5ncnn-vulkan
   ```

### Compile ncnn-vulkan-ARM by yourself

1. The compilation of ncnn-vulkan should be on the x86 device.

2. Install protobuf.

   ```shell
   sudo apt install protobuf-compiler libprotobuf-dev 
   ```

3. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

4. Download vulkan-sdk from https://vulkan.lunarg.com/sdk/home#sdk/downloadConfirm/1.2.148.0/linux/vulkansdk-linux-x86_64-1.2.148.0.tar.gz. and add to environment variables (reboot may needed).

   ```shell
   export VULKAN_SDK=~/vulkan-sdk-1.2.148.0/x86_64
   export PATH=$PATH:$VULKAN_SDK/bin
   export LIBRARY_PATH=$LIBRARY_PATH$:VULKAN_SDK/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VULKAN_SDK/lib
   export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
   ```

5. Download source code of ncnn from https://github.com/Tencent/ncnn/releases.

   ```shell
   unzip ncnn-master.zip
   ```

6. Download [gcc-arm-toolchain](https://developer.arm.com/-/media/Files/downloads/gnu-a/8.2-2018.11/gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu.tar.xz?revision=7a60a425-1aa0-43f5-b9db-1af71bffadc6&la=en) and add to environment variables.

   ```shell
   tar -zxvf gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu.tar.xz
   gedit ~/.bashrc
   export PATH=$PATH:/home/username/gcc-arm-8.2-2018.11-x86_64-aarch64-linux-gnu/bin
   source ~/.bashrc
   ```

7. Compile ncnn-vulkan.

   ```shell
   cd ncnn
   mkdir -p build-aarch64-linux-vulkan
   cd build-aarch64-linux-vulkan
   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake –DANDROID=ON -DNCNN_VULKAN=ON ..
   make -j8
   make install
   ```

8. In order to compile the project correctly on ARM devices, additional static link libraries (libvulkan-sdk.a  and libvulkan-stub.a) are needed and can be obtained from [here]([ARM-software/vulkan-sdk: Github repository for the Vulkan SDK](https://github.com/ARM-software/vulkan-sdk)).

### Get your own Yolov5 ncnn model

We train a model in Pytorch and first convert to onnx and then to ncnn.

1. For how to train in Pytorch and export to onnx, see https://github.com/ultralytics/yolov5.

2. Because ncnn has limited support for operators, the network definition needs to be modified before training, please modify "common.py".

   from

   ```python
   class Focus(nn.Module):
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
           super(Focus, self).__init__()
           self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
   
       def forward(self, x):
           return self.conv(torch.cat([x[..., ::2, ::2], x[..., ::2, ::2], x[..., ::2, ::2], x[..., ::2, ::2]], 1))
   ```

   to

   ```python
   class Focus(nn.Module):
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
           super(Focus, self).__init__()
           self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
   
       def forward(self, x):
           return self.conv(torch.cat([torch.nn.functional.interpolate(x, scale_factor=0.5),
                                       torch.nn.functional.interpolate(x, scale_factor=0.5),
                                       torch.nn.functional.interpolate(x, scale_factor=0.5),
                                       torch.nn.functional.interpolate(x, scale_factor=0.5)], 1))
   ```

3. When export to onnx, Detect layer should be removed from the graph, please modify  "export.py".

   ```python
   model.model[-1].export = True
   ```

4. Simplify the onnx model by onnx-simplifier.

   ```shell
   pip3 install onnx-simplifier
   python3 -m onnxsim yolov5s.onnx yolov5s.onnx
   ```

5. Convert onnx to ncnn

   ```shell
   ./onnx2ncnn yolov5s.onnx yolov5s.param yolov5s.bin
   ```
