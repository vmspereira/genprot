from flask import Flask, request, send_file
from igfold import IgFoldRunner

try:
     from igfold import init_pyrosetta
     init_pyrosetta()
except Exception as e:
    pass


HOME = './temp/'
DATA = '/data/'

app = Flask(__name__)

def get_pdb(VH, VL, filename):
    sequences = {
        "H": VH,
        "L": VL
    }
    pred_pdb = filename

    igfold = IgFoldRunner()
    igfold.fold(
        pred_pdb, # Output PDB file
        sequences=sequences, # Antibody sequences
        do_refine=True, # Refine the antibody structure with PyRosetta
        do_renum=True, # Renumber predicted antibody structure (Chothia)
    )

@app.route("/")
def index():
    return "<p>IgFold</p>"

@app.route("/igfold/", methods=["POST"])
def generate_pdb():
    vh = request.form.get('VH')
    vl = request.form.get('VL')
    filename = "my_antibody.pdb"
    get_pdb(vh,vl,filename)
    try:
        return send_file(filename,attachment_filename=filename)
    except Exception as e:
        return str(e)

@app.route("/igfoldbatch/", methods=["POST"])
def evaluate():
    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)