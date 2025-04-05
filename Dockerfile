FROM nvidia/cuda:12.8.1-runtime-ubuntu20.04
EXPOSE 8572
WORKDIR /app
COPY . .

# Install dependenceis to add PPAs
RUN apt-get update && \
    apt-get install -y wget unzip && apt clean && \
    apt-get install -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA to get Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9 and pip
RUN apt-get update && \
    apt-get install -y build-essential python-dev python3-dev python3.9-distutils python3.9-dev python3.9 curl && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN python3 -m pip install --upgrade pip==24.0
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN wget https://git.apad.pro/github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip
RUN unzip pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip
RUN wget https://git.apad.pro/github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip
RUN unzip hnsep_240512.zip
RUN mv ./vr ./hnsep -f

CMD ["python3", "hifiserver.py"]
