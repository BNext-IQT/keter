FROM ubuntu:20.04
LABEL maintainer="JJ Ben-Joseph (jbenjoseph@iqt.org)" \
      description="Contains the UI for managing RQ jobs."
EXPOSE 8080
CMD uwsgi --http :8080 --wsgi-file app.py
COPY app.py /
WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
       python3-minimal python3-pip python3-dev build-essential \
       libpython3.8 \ 
    && pip3 install --no-cache-dir uwsgi rq-dashboard \
    && apt-get remove -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*