FROM ubuntu:18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        g++ \
        make \
        wget \
        unzip \
        vim \
        git \
        libssl-dev \
        time \
        numactl

WORKDIR /tmp
# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6.tar.gz \
    && tar xvf cmake-3.18.6.tar.gz && cd cmake-3.18.6 \
    && ./bootstrap && make -j$(nproc) && make install

# Install szip for HDF5
RUN wget https://support.hdfgroup.org/ftp/lib-external/szip/2.1.1/src/szip-2.1.1.tar.gz \
    && tar xvf szip-2.1.1.tar.gz && cd szip-2.1.1 \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) && make install

# Install HDF5
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz \
    && tar xvf hdf5-1.12.1.tar.gz && cd hdf5-1.12.1 \
    && ./configure --prefix=/usr/local/hdf5 --enable-cxx --with-szlib=/usr/local/lib --enable-threadsafe --with-pthread=/usr/include/ --enable-hl --enable-shared --enable-unsupported \
    && make -j$(nproc) && make install

# Install SEAL
RUN wget https://github.com/microsoft/SEAL/archive/refs/tags/v3.6.6.tar.gz \
    && tar xvf v3.6.6.tar.gz && cd SEAL-3.6.6 \
    && cmake -S . -B build && cmake --build build -- -j$(nproc) && cmake --install build

WORKDIR /
# Install Eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    && tar xvf eigen-3.4.0.tar.gz \
    && cp -r eigen-3.4.0/Eigen /usr/local/include/ \
    && cd eigen-3.4.0 && cmake -S . -B build

ENV PATH $PATH:/usr/local/hdf5/bin
ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/hdf5/lib
ENV LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/hdf5/lib

COPY . /app
WORKDIR /app

RUN Eigen3_DIR=/eigen-3.4.0/build \
    cmake -S . -B build && cmake --build build -- -j$(nproc)
