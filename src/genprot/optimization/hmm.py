import requests
import json

def getURL(url, params=None, **kwargs):
    return requests.get(url, params, **kwargs)


def postURL(url, data=None, json=None, **kwargs):
    return requests.post(url, data, json, **kwargs)


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