FROM nvcr.io/nvidia/pytorch:22.11-py3
ARG USER_ID
ARG GROUP_ID
ARG USER
 
ENV TZ=Europe/London
 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
 
RUN apt-get update && apt-get install -y sudo \
    libgl1-mesa-glx \
    ffmpeg \
    libimage-exiftool-perl \
    swig

# RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" 

# RUN  curl -sL \
#       "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
#       "Miniconda3.sh"

# RUN  bash Miniconda3.sh


 
# RUN conda env create -f /home/ht23/KFlearn/code/docker/environment.yml
# RUN pip3 install --upgrade pip
# RUN pip install opencv-python
# RUN pip install --upgrade opencv-python
# RUN pip3 install ffmpeg-python
# RUN pip install opencv-contrib-python

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
 
WORKDIR /nfs/home/$USER
EXPOSE 8888