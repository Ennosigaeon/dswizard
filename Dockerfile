FROM jupyter/minimal-notebook:e28c630dfc4f

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
RUN pip install -e .