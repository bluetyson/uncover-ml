# Docker file for uncoverml project

FROM ubuntu:16.04
MAINTAINER Lachlan McCalman <lachlan.mccalman@data61.csiro.au>

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
RUN apt-get update && apt-get install -y \
  wget \
  build-essential \
  # Fast blas for numpy and scipy
  libopenblas-base \ 
  libopenblas-dev \
  python3 \
  python3-dev \
  python3-pip \
  # Needed for matplotlib
  libfreetype6-dev \
  libxft-dev \
  # Needed for ipyparallel
  libzmq3-dev \
  # Needed for pytables
  libhdf5-dev \
  libbz2-dev \
  liblzo2-dev \
  zlib1g-dev \
  gfortran \
  # Needed for rasterio
  libgdal-dev \
  gdal-bin \
  # confd for doing in-container configuration
  && mkdir -p /webserver /usr/local/bin etc/confd/conf.d /etc/confd/templates \
  && wget https://github.com/kelseyhightower/confd/releases/download/v0.11.0/confd-0.11.0-linux-amd64 -O /usr/local/bin/confd \
  && chmod +x /usr/local/bin/confd \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  # Make folders
  && mkdir -p /usr/src/python/uncoverml

# pip packages 
RUN pip3 -v install \
  ipython \
  Cython \
  numpy \
  scipy \
  numexpr \
  tables \
  matplotlib \
  scikit-learn \
  ipyparallel


# install prereqs for caching reasons
RUN pip3 -v install \ 
  rasterio \ 
  affine \ 
  pytest \ 
  pytest-cov \ 
  pyshp \ 
  click \
  revrand

WORKDIR /usr/src/python/uncoverml

COPY . /usr/src/python/uncoverml

RUN python3 setup.py develop

