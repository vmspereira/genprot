import random
import os
import numpy as np
from flask import Flask, request, jsonify

HOME = './temp/'
DATA = '/data/'
app = Flask(__name__)

def evaluateHMM(sequence, hmmfile):
    filename = 'seq'+str(random.randint(1000000000000,999999999999999999999))+'.faa'
    hmmfile = DATA+hmmfile
    with open(HOME+filename,'w') as f:
        f.write('>seq\n')
        f.write(sequence.replace('-','_'))
    
    stream = os.popen('hmmscan -T 0 '+hmmfile+' '+HOME+filename)
    output = stream.read()
    result = -np.inf
    try:
        lines=output.split('\n')
        line = lines[15]
        tokens = line.split()
        result =  float(tokens[1])
    except Exception as e:
        print(e)
    os.system('rm '+HOME+filename)
    return result


@app.route("/")
def index():
    return "<p>hmmer</p>"

@app.route("/eval/", methods=["POST"])
def evaluate():
    seq = request.form.get('seq')
    hmm = request.form.get('hmm')
    value = evaluateHMM(seq,hmm)
    return jsonify({'value':value})


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5010)