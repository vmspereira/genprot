import pandas as pd
from Bio import SeqIO

'''
Helper method to convert a csv file to fasta
'''
def convert_to_fasta(file):
    file_csv = pd.read_csv(file)
    id = file_csv['id']
    sequences = file_csv['sequence']
    fasta = open ('dataset_gan.fasta', 'w')
    for i in range(len(sequences)):
        fasta.write('>' + id[i] + '\n' + sequences[i] + '\n')
    fasta.close

    return fasta

    
