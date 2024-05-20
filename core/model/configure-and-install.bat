@echo off

cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX=install;../util/install
cmake --build build --target install