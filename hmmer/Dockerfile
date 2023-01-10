FROM python:3.8

ARG HMMER_VERSION="3.3.2"

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    htop \
    vim

WORKDIR root
RUN wget http://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz
RUN tar zxf hmmer-${HMMER_VERSION}.tar.gz
WORKDIR /hmmer-${HMMER_VERSION}
RUN ./configure --prefix /usr/local
RUN make
# optional: run automated tests
RUN make check
# optional: install HMMER programs, man pages
RUN make install
# optional: install Easel tools
RUN (cd easel; make install)
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY app.py /app/app.py
EXPOSE 5000
CMD ["gunicorn","--bind", "0.0.0.0:5000", "--workers", "4","--timeout","300","app:app"]
