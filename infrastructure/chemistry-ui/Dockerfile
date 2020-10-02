FROM ubuntu:20.04
LABEL maintainer="JJ Ben-Joseph (jbenjoseph@iqt.org)" \
      description="The antiviral UI based from the ChEMBL GlaDOS interface."
EXPOSE 8080
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
ARG CONFIG_FILE_PATH='/app/config.yml'
COPY glados-frontend-chembl-main-interface /app
COPY config.yml /app/config.yml
RUN apt-get update && apt-get install -y --no-install-recommends \
       python3-minimal python3-pip python3-dev build-essential \
       libpython3.8 git nodejs
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata
RUN pip3 install --no-cache-dir uwsgi
RUN pip3 install --no-cache-dir .
RUN apt-get remove -y build-essential python3-dev git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*
RUN python3 manage.py collectstatic --no-input && python3 manage.py compress
ENTRYPOINT uwsgi --http :8080 --module glados.wsgi:APP 