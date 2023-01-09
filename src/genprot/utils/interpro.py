#!/usr/bin/python3

import sys
import json
import ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep
import pyhmmer
import pyfasta
import os

HEADER_SEPARATOR = "|"
LINE_LENGTH = 80
PAGE_SIZE = 200
MAX_SEQUENCE_LENGHT = 504
FOLDER = 'data/'


def interpro(assertion, maxlen=None):

    filename = f"{assertion}.fasta"
    total = 0
    # disable SSL verification to avoid config issues
    context = ssl._create_unverified_context()
    BASE_URL = (f"https://www.ebi.ac.uk:443/interpro/api/protein/"
                f"UniProt/entry/InterPro/{assertion}"
                f"/?page_size={PAGE_SIZE}&extra_fields=sequence")
    next = BASE_URL

    attempts = 0
    while next:
        try:
            req = request.Request(next, headers={"Accept": "application/json"})
            res = request.urlopen(req, context=context)
            # If the API times out due a long running query
            if res.status == 408:
                # wait just over a minute
                sleep(61)
                # then continue this loop with the same URL
                continue
            elif res.status == 204:
                # no data so leave loop
                break
            payload = json.loads(res.read().decode())
            next = payload["next"]
            attempts = 0
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            else:
                # If there is a different HTTP error, it wil re-try 3 times
                # before failing
                if attempts < 3:
                    attempts += 1
                    sleep(61)
                    continue
                else:
                    sys.stderr.write("LAST URL: " + next)
                    raise e

        with open(f"./{FOLDER}{filename}", 'a') as f:
            for i, item in enumerate(payload["results"]):

                if ("entries" in item):
                    for entry in item["entries"]:
                        for locations in entry["entry_protein_locations"]:
                            for fragment in locations["fragments"]:
                                start = fragment["start"]
                                end = fragment["end"]

                                header = ">" + \
                                    item["metadata"]["accession"] + \
                                    HEADER_SEPARATOR + \
                                    entry["accession"] + HEADER_SEPARATOR + \
                                    str(start) + "..." + \
                                    str(end) + HEADER_SEPARATOR + \
                                    item["metadata"]["name"] + "\n"

                                seq = item["extra_fields"]["sequence"]

                                fastaSeqFragments = [seq[0+i:LINE_LENGTH+i]
                                                     for i in range(0, len(seq),
                                                     LINE_LENGTH)]
                                sequence = ""
                                for fastaSeqFragment in fastaSeqFragments:
                                    sequence = sequence + fastaSeqFragment + "\n"


                else:
                    header = ">" + item["metadata"]["accession"] + \
                        HEADER_SEPARATOR + item["metadata"]["name"] + "\n"

                    seq = item["extra_fields"]["sequence"]
                    fastaSeqFragments = [seq[0+i:LINE_LENGTH+i]
                                         for i in range(0, len(seq),
                                         LINE_LENGTH)]
                    sequence = ""
                    for fastaSeqFragment in fastaSeqFragments:
                        sequence = sequence + fastaSeqFragment + "\n"
                    

                if maxlen is None or len(sequence.replace('\n', '')) <= maxlen:
                    f.write(header)
                    f.write(sequence)
                    total += 1
                    print(f"Downloading sequences: {total}",end='\r')

                # Don't overload the server, give it time before
                # asking for more
        if next:
            sleep(1)
    print(f"Downloaded sequences: {total}")
    return filename


def filter_size(filename, maxsize, minsize=0):
    f = pyfasta.Fasta(f"./{FOLDER}{filename}")
    
    tokens = filename.split('.')
    out = '.'.join(tokens[:-1])+'-filtered.fasta'
    
    with open(f"./{FOLDER}{out}", "w") as output_file:
        for header in f.keys():
            name = str(header)
            seqt = str(f[header])
            if len(seqt) <= maxsize and len(seqt) >= minsize:
                output_file.write(">"+name+"\n")
                output_file.write(seqt+"\n")
    return out


def hmm(filename):
    seqs = []

    f = pyfasta.Fasta(f"./{FOLDER}{filename}")
    for header in f.keys():
        name = str(header)
        seqt = str(f[header])
        seq = pyhmmer.easel.TextSequence(
            name=bytes(name, 'ascii'), sequence=seqt)
        seqs.append(seq)

    msa = pyhmmer.easel.TextMSA(bytes('msa', 'ascii'), sequences=seqs)
    alphabet = pyhmmer.easel.Alphabet.amino()
    msa_d = msa.digitize(alphabet)

    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa_d, background)

    print('consensus:', hmm.consensus)
    print('cutoffs', hmm.cutoffs)

    tokens = filename.split('.')
    out = '.'.join(tokens[:-1])+'.hmm'

    with open(f"./{FOLDER}{out}", "wb") as output_file:
        hmm.write(output_file)

    return out


def msa(filename):
    tokens = filename.split('.')
    out = '.'.join(tokens[:-1])+'-aln.fasta'
    os.system("docker run -v ${PWD}/" + FOLDER + ":/data " +
                   "staphb/mafft mafft --auto " + filename + " > data/" + out)

    return out
    
if __name__ == "__main__":
    try:
        assertion = sys.argv[1]
        print(f"Downloading fasta file for assertion {assertion}")
        print("==================================================")
        filename = interpro(assertion)
        print(f"out: {filename}\n")
        print(f"Filtering sequences by size {MAX_SEQUENCE_LENGHT}")
        print("==================================================")
        filename = filter_size(filename, MAX_SEQUENCE_LENGHT)
        print(f"out: {filename}\n")
        print("Running MSA")
        print("==================================================")
        filename = msa(filename)
        print(f"out: {filename}\n")
        print("==================================================")
        print("Building HMM")
        hmmfile = hmm(filename)
        print(f"out: {hmmfile}")

    except Exception as e:
        print(e)
