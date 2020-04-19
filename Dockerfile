FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL version="1.0" 

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update --fix-missing && \
    apt-get install -y \
        wget bzip2 ca-certificates curl git apt-utils build-essential cmake pkg-config \
        libsm6 libxrender1 libfontconfig1 libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache 

# -- install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV CONDA_AUTO_UPDATE_CONDA=false

# -- install the sfsnet package and its dependences
ADD . /app
WORKDIR /app

RUN pip install --upgrade pip && \
    sh install_requirements.sh && \
    python setup.py develop && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

## -- define the image's main command
#ENTRYPOINT ["/bin/bash", "sfsnet.py"]
