# Base Image to start from 
# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM ubuntu:22.04

# If interactive is on, you will find applications asking to type in keyboard inputs in docker image build, except keyboard does not register during building, and system hangs up in install. Happens to python install.
ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER dockerscade

# timezone & workdir
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# WORK directory from root
WORKDIR /root/project

#link python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Update package lists and install essential packages including Python and pip
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
	git-lfs \
    vim \
    software-properties-common \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \