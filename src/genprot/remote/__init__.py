import requests

def getURL(url, params=None, **kwargs):
    return requests.get(url, params, **kwargs)


def postURL(url, data=None, json=None, **kwargs):
    return requests.post(url, data, json, **kwargs)