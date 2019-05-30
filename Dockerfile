FROM python:3.6.8-slim-stretch

ARG NB_USER="ai-lab"
ARG NB_UID="1000"
ARG WORKDIR="/rl"

# install gym dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    xvfb xauth libglu1-mesa libgl1-mesa-dri python-opengl graphviz && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# install requirements
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt && \
    rm -rf /tmp/*

# create user -m (creates home if it doesn't exist) -s (users standard shell) -u (UID)
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER
USER $NB_UID

EXPOSE 8888
# serve jupyter notebooks from this folder
WORKDIR $WORKDIR

CMD xvfb-run -s "-screen 0 600x400x24" \
    jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --NotebookApp.token=''
