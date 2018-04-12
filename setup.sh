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