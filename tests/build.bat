@echo off

REM This script configures and builds the core libraries
REM Configure and install the model library
cmake -S ../core/model -B ../core/model/build -G Ninja -DCMAKE_INSTALL_PREFIX=../core/model/install
cmake --build ../core/model/build --target install

REM This script configures and builds the util library
cmake -S ../core/util -B ../core/util/build -G Ninja -DCMAKE_INSTALL_PREFIX=../core/util/install
cmake --build ../core/util/build --target install

REM Configure and build the test application
cmake -S . -B build/test -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=../core/model/install;../core/util/install

cmake --build build/test

cd build/test && ctest