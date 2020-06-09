```bash
export DSSM_PATH=`pwd`
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$DSSM_PATH/contrib/libtorch -DTorch_DIR=$DSSM_PATH/contrib/libtorch/share/cmake/Torch/ -DCMAKE_CXX_FLAGS="-Wunused -Werror -O3 -flto -ffast-math -I $SGL_PATH/include/" ..
```
