FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
# non interactive mode
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y nginx python3 python3-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/TahaBinhuraib/nomic


# cd into the git repo and run pip3 install .[GPT4All]
# This will install torch=2 sentencepiece and 2 other dependencies: transformers and peft from wheel
RUN cd nomic && pip3 install .[GPT4All]
# Go back to working directory
COPY test.py /app/nomic
WORKDIR /app/nomic
