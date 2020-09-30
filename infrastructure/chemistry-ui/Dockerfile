FROM ubuntu:20.04
LABEL maintainer="JJ Ben-Joseph (jbenjoseph@iqt.org)" \
      description="The antiviral UI based from the ChEMBL GlaDOS interface."
EXPOSE 8080
WORKDIR /app
COPY glados-frontend-chembl-main-interface /app
COPY config.yml /app/config.yml
RUN apt-get update && apt-get install -y --no-install-recommends \
       python3-minimal python3-pip python3-dev build-essential \
       libpython3.8 git 
RUN pip3 install --no-cache-dir uwsgi
RUN pip3 install --no-cache-dir -U 'setuptools<46'
RUN pip3 install --no-cache-dir -r /app/requirements.txt 
RUN apt-get remove -y build-essential python3-dev git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*
RUN CONFIG_FILE_PATH='/app/config.yml' python3 manage.py collectstatic --no-input
ENTRYPOINT python3 manage.py runserver 8000