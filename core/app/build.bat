@echo off

REM This script configures and builds the core application and its dependencies
REM Configure and install the util library
cmake -S ../util -B ../util/build -G Ninja -DCMAKE_INSTALL_PREFIX=../util/install
cmake --build ../util/build --target install

REM Configure and install the model library
cmake -S ../model -B ../model/build -G Ninja -DCMAKE_INSTALL_PREFIX=../model/install
cmake --build ../model/build --target install

REM Configure and build the core application
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=../util/install;../model/install
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=../util/install;../model/install

cmake --build build/debug
cmake --build build/release