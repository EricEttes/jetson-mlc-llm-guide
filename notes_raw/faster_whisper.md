Faster-whisper

# Installation of CUDA enabled ctranslate2

git clone --recursive https://github.com/OpenNMT/CTranslate2.git

mkdir -p CTranslate2/build && cd CTranslate2/build

# build C++ libraries
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP -DCMAKE_INSTALL_PREFIX="/home/eric/projects/misty/CTranslate2/build/install/"

make -j$(nproc)
sudo make install

cp -r /home/eric/projects/misty/CTranslate2/build/install/* /usr/local/

# build Python packages
cd CTranslate2/python
pip3 install -r install_requirements.txt
python3 setup.py --verbose bdist_wheel --dist-dir /home/eric/projects/misty/

# install/upload wheels
pip3 install --force-reinstall /home/eric/projects/misty/ctranslate2*.whl



Speed test with turbo model:

Whisper: 2026-01-10 15:36:08,941 - INFO - Total transcription time: 76.37 seconds
Faster whisper (2.7x faster): 2026-01-10 15:42:52,824 - INFO - Processed 89 segments in 28.09 seconds (avg 0.316s per segment)
