FROM python:3.9-bullseye

RUN mkdir /opt/dswizard/ &\
    mkdir /dswizard/

RUN pip install dswizard

COPY fabos/entrypoint.py dswizard/assets/rf_complete.pkl /opt/dswizard/

RUN chmod +x /opt/dswizard/entrypoint.py

ENTRYPOINT ["python", "/opt/dswizard/entrypoint.py"]