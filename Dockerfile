FROM jupyter/scipy-notebook:2022-05-09

ENV VERSION=0.2.5

COPY dist/dswizard-$VERSION.tar.gz /tmp/
COPY dswizard/assets/ /home/jovyan/dswizard/assets/
COPY scripts/ /home/jovyan/dswizard/examples/

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN pip install /tmp/dswizard-$VERSION.tar.gz