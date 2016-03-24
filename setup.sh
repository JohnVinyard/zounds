#!/bin/sh
apt-get update
# install packages needed by libsndfile
apt-get install -y \
	g++ \
	autoconf \
	autogen \
	automake \
	libtool \
	pkg-config \
	libogg0 \
	libogg-dev \
	libvorbis0a \
	libvorbis-dev \
	libsamplerate0 \
	libsamplerate0-dev

# get libflac 1.3.1, build and install
wget http://downloads.xiph.org/releases/flac/flac-1.3.1.tar.xz
tar xf flac-1.3.1.tar.xz
cd flac-1.3.1
./configure --prefix=/usr/bin && make && make install
# get libsndfile 1.0.26, build and install
cd ..
wget https://github.com/erikd/libsndfile/archive/1.0.26.tar.gz
tar -xzf 1.0.26.tar.gz
cd libsndfile-1.0.26
./autogen.sh
./configure --prefix=/usr/bin && make && make install