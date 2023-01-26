from . import postURL

def hmm(url, seq, hmmfile):
    json = {'seq':seq,
            'hmm':hmmfile} 
    res = postURL(url,json=json)
    rdata = json.loads(res.json())
    if rdata and rdata['value']:
        value = rdata['value']
    else:
        value = None
    return value