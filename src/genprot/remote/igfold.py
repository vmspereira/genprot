from . import postURL

def igfold(url, vh, vl):
    json = {'VH':vh,
            'VL':vl} 
    res = postURL(url,json=json)
    pass