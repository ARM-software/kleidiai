#rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -DCMAKE_C_FLAGS=-march=armv8.5a+sve2+sme -S . -B build/
cd build/
make
cd ..
cp build/kleidiai_test ../../../../qemu/emulboot/binaries/ 
