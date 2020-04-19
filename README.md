## Leveraging Shape, Reflectance and Albedo from Shading for Face Presentation Attack Detection (SFSNet)

This software is an official implementation of the paper entitled *Leveraging Shape, Reflectance and Albedo from Shading for Face Presentation Attack Detection* published in the *IEEE Transactions on Information Forensics and Security*. If you use this software, please cite our paper as follow:

> **Plain Text**
> *It will be available soon*

> **BibTeX**
> *It will be available soon*


## License

This software is available under condition of the [AGPL-3.0 Licence](https://github.com/allansp84/sfsnet/blob/master/LICENSE).


## How to install this software?

This reporsitory provides a Python Package to make the instalation and usage of this sotware easy. We also provide a Dockerfile to build Docker containers ready to use this software. To build a docker container for this sotfware please execute the command below. This software was implemented using the Python 3.5 language. The experiments were performed considering a Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz with 12 cores, a Nvidia GTX 1080 TI GPU, and Ubuntu 18.04.3 LTS.
>
>     sh build_docker_image.sh
>      

## Download and Installation

To download this software, please execute the following command:

>     cd ~
>     git clone https://github.com/allansp84/sfsnet.git

Please execute the following command for installing this software:

>     cd ~/SFSNet
>     sh install_requirements.sh
>     pyhton setup.py install
>     sfsnet.py --help

The second line should install all dependencies


## Bulding a Docker Container

To build a Docker Image to run this software, please execute the commands below:

>     docker build --rm --tag allansp-sfsnet:1.0 --file Dockerfile .

After build a Docker Image, please execute the following commands to run a docker container:

>     nvidia-docker run --rm -it --ipc=host --userns=host --name allansp-sfsnet-c1 \
>           -v ${HOME}:${HOME} \
>           allansp-sfsnet:1.0 bash



## Usage

After installing our software, we can use it via command line interface (CLI). To see how to use this software, execute the following command in any directory, since it will be already installed in your system:

>     sfsnet.py --help


## Examples
1. Compute the Albedo, Depth, and Reflectance maps:

> *Replay Attack Dataset:*
> ./scripts/1_map_extraction_replayattack.sh

> *Casia Dataset:*
> ./scripts/1_map_extraction_casia.sh


2. Build the Multi-channel Input Tensors:

> *Replay Attack Dataset:*
> ./scripts/2_build_multichannel_input_replayattack.sh

> *Casia Dataset:*
> ./scripts/2_build_multichannel_input_casia.sh

2. Run the SFSNet Network:

> *Replay Attack Dataset:*
> ./scripts/3_classification_experiment_1_spoofnet_replayattack.sh

> *Casia Dataset:*
> ./scripts/3_classification_experiment_1_spoofnet_casia.sh
