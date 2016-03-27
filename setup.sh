#!/bin/sh
apt-get update
# install packages needed by libsndfile, matplotlib and lmdb
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
	libsamplerate0-dev \
	libx11-dev \
	python-dev \
	libfreetype6-dev \
	libpng12-dev \
	libffi-dev
# get libflac 1.3.1, build and install
wget http://downloads.xiph.org/releases/flac/flac-1.3.1.tar.xz
tar xf flac-1.3.1.tar.xz
cd flac-1.3.1
./configure && make && make install
# get libsndfile 1.0.26, build and install
cd ..
wget http://www.mega-nerd.com/tmp/libsndfile-1.0.26pre5.tar.gz
tar -xzf libsndfile-1.0.26pre5.tar.gz
cd libsndfile-1.0.26pre5
./autogen.sh
./configure --prefix=/usr && make && make install