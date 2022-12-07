FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]
RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

WORKDIR /main/
ENV LIBGL_ALWAYS_INDIRECT=1
ENV DEBIAN_FRONTEND=interactive
RUN apt-get update
RUN apt-get install -y wget bzip2 ffmpeg libsm6 libxext6 \ 
	 libxcb-xinerama0 libxcb1-dev libx11-dev libgl1-mesa-dev qt5-default wireshark xvfb 
RUN apt-get clean
RUN apt clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

#online soln
#RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

ENV DEBIAN_FRONTEND=interactive

ENV PATH /opt/conda/bin:$PATH

COPY ./environment.yml ./environment.yml
RUN conda update conda -y 
#RUN conda update -n base -c defaults conda -y
RUN conda env create --name Capstone -f ./environment.yml

RUN echo "conda activate Capstone" >> ~/.bashrc
ENV PATH /opt/conda/envs/Capstone/bin:$PATH
ENV CONDA_DEFAULT_ENV $Capstone


RUN pip install --upgrade pip 
COPY ./docker_requirements.txt ./
RUN conda install -n Capstone  cvxopt==1.2.6
RUN conda install -c anaconda -p /opt/conda/envs/Capstone pyqt qt
RUN pip install pyqt5
RUN pip install --no-cache-dir -r ./docker_requirements.txt

RUN conda install -n Capstone --force-reinstall libxcb -y
USER qtuser
COPY --chown=qtuser:qtuser  ./ ./
#COPY ./ ./
#CMD ["bash"]
