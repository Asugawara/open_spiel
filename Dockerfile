FROM ubuntu:20.04 as base
RUN apt update
RUN dpkg --add-architecture i386 && apt update
RUN apt-get -y install \
    clang \
    curl \
    git \
    vim \
    screen \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    gobject-introspection \
    gir1.2-gtk-3.0 \
    python3-gi-cairo \
    graphviz \
    graphviz-dev \
    sudo && apt update
RUN mkdir repo
WORKDIR /repo

RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install matplotlib
RUN sudo pip3 install jupyterlab
RUN sudo pip3 install torch
RUN sudo pip3 install hypothesis
RUN sudo pip3 install --install-option="--include-path=/usr/share/graphviz"  --install-option="--library-path=/usr/share/graphviz" pygraphviz

# install
COPY . .
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN ./install.sh
RUN pip3 install --upgrade setuptools testresources
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install --upgrade cmake

# build and test
RUN mkdir -p build
WORKDIR /repo/build
RUN cmake -DPython3_EXECUTABLE=`which python3` -DCMAKE_CXX_COMPILER=`which clang++` ../open_spiel
RUN make -j4
ENV PYTHONPATH=${PYTHONPATH}:/repo
ENV PYTHONPATH=${PYTHONPATH}:/repo/build/python
RUN ctest -j4
WORKDIR /repo/open_spiel

# minimal image for development in Python
FROM python:3.6-slim-buster as python-slim
RUN mkdir repo
WORKDIR /repo
COPY --from=base /repo .
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install matplotlib
RUN sudo pip3 install jupyterlab
ENV PYTHONPATH=${PYTHONPATH}:/repo
ENV PYTHONPATH=${PYTHONPATH}:/repo/build/python
WORKDIR /repo/open_spiel
