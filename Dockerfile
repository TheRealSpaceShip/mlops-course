FROM quay.io/jupyter/base-notebook:lab-4.2.5

USER root

RUN apt-get clean --yes && \
    apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    curl \
    git \
    # libgl1-mesa-glx \
    # libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    # libasound2 \
    libxi6 \
    libxtst6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
RUN eval "$(/$HOME/miniconda3/bin/conda shell.$SHELL hook)"

RUN conda install --yes --quiet \
    mlflow \
    hyperopt \
    pandas \
    seaborn \
    matplotlib \
    scikit-learn \
    pyarrow
