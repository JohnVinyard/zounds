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

# get libflac, build and install
wget http://downloads.xiph.org/releases/flac/flac-1.3.1.tar.xz
tar xf flac-1.3.1.tar.xz
cd flac-1.3.1
./configure && make && make install
cd ..
# get libsndfile, build and install
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.27.tar.gz
tar -xzf libsndfile-1.0.27.tar.gz
cd libsndfile-1.0.27
./configure --libdir=/usr/lib/x86_64-linux-gnu && make && make install