FROM ubuntu:22.04
LABEL maintainer="JJ Ben-Joseph (jbenjoseph@iqt.org)" \
    description="Keter biosecurity platform. Agent container."
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY setup.py requirements.txt README.rst /app/
COPY keter /app/keter
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal python3-pip libopenblas0-openmp cython3 \
    python3-dev build-essential cmake libopenblas-openmp-dev \
    gfortran libffi-dev python3-pkg-resources python3-wheel \
    libpython3.8 python3-rdkit git python3-torch libtorch-dev \
    pkg-config pybind11-dev \
    && CFLAGS="-g0 -O3 -Wl,--strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib" \
    pip3 install --no-cache-dir -e . \
    # COMPILE OPTIONS FOR FASTER PERFORMANCE BUT SLOWER BUILD
    #    && pip3 install --no-cache-dir --compile --global-option=build_ext \
    #    --global-option="-j 4" -e . \
    && apt-get remove -y python3-dev python3-pip build-essential cmake \
    libopenblas-openmp-dev gfortran libffi-dev pkg-config pybind11-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*