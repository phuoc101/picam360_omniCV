# Dependencies
    ```bash
    sudo apt install cmake
    sudo apt install gcc g++
    sudo apt install python3 python3-dev python3-numpy
    sudo apt install libavcodec-dev libavformat-dev libswscale-dev
    sudo apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
    sudo apt install libgtk-3-dev
    sudo apt install libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev
    ```
# Clone repositories
    ```bash
    mkdir cv_install && cd cv_install
    git clone https://github.com/opencv/opencv_contrib.git
    git clone https://github.com/opencv/opencv_contrib.git
    ```
# Determine cuda architecture
# Building
    ```bash
    cd ~/opencv
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=6.1 -D OPENCV_EXTRA_MODULES_PATH=$HOME/cv_install/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
    ```
Last command will generate compile_config json file, put in head directory for lsp_to work
# Compile OpenCV with CUDA support
    ```bash
    make -j 8
    sudo make install
    sudo ldconfig
    sudo ln -s /usr/local/lib/python3.8/site-packages/cv2 /usr/local/lib/python3.8/dist-packages/cv2
    ```

