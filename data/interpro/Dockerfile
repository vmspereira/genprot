FROM python:3.8


ARG MAFFT_VER="7.475"

# install wget, remove apt garbage
RUN apt-get update && apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# install mafft binary and make /data
RUN wget https://mafft.cbrc.jp/alignment/software/mafft_${MAFFT_VER}-1_amd64.deb && \
    dpkg -i mafft_${MAFFT_VER}-1_amd64.deb && \
    rm mafft_${MAFFT_VER}-1_amd64.deb && \
    mkdir /data

RUN pip install pyhmmer numpy pyfasta
COPY ./inter.py /
COPY ./script.sh /
RUN chmod +x /script.sh
ENTRYPOINT ["/script.sh"]

