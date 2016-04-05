FROM ubuntu:14.04

MAINTAINER John Vinyard <john.vinyard@gmail.com>

RUN apt-get update --fix-missing && apt-get install -y \
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
	libffi-dev \
	python-pip \
	wget \
	bzip2 \
	ca-certificates \
	libglib2.0-0 \
	libxext6 \
	libsm6 \
	libxrender1 \
    git \
    mercurial \
    subversion

RUN wget http://downloads.xiph.org/releases/flac/flac-1.3.1.tar.xz \
    && tar xf flac-1.3.1.tar.xz \
    && cd flac-1.3.1 \
    && ./configure && make && make install \
    && cd..

RUN wget http://www.mega-nerd.com/tmp/libsndfile-1.0.26pre5.tar.gz \
    && tar -xzf libsndfile-1.0.26pre5.tar.gz \
    && cd libsndfile-1.0.26pre5 \
    && ./autogen.sh \
    && ./configure --libdir=/usr/lib/x86_64-linux-gnu && make && make install

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2.5.0-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH

RUN pip install zounds

CMD zounds-quickstart --datadir data --port 9999