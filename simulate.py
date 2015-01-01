# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# 
# Simulate data observed by an interferometer

import numpy

def rot(p, ha, dec):
    """
    Rotate x,y,z in earth coordinates to u,v,w relative to direction
    defined by ha and dec

    p: array of antenna positions
    """
    x,y,z=numpy.hsplit(p,3)
    t=x*numpy.cos(ha) - y*numpy.sin(ha)
    u=x*numpy.sin(ha) + y*numpy.cos(ha)
    v=t*numpy.sin(dec)+ z*numpy.cos(dec)
    w=t*numpy.cos(dec)+ z*numpy.sin(dec)
    return numpy.hstack([u,v,w])

def bls(p):
    """
    Compute baseline uvws from station uvw
    """
    res=[]
    for i in range(p.shape[0]):
        for j in range(i+1, p.shape[0]):
            res.append(p[i]-p[j])
    return numpy.array(res)

def genuv(p, ha, dec):
    return(numpy.concatenate([bls(rot(p,hax,dec)) for hax in ha]))
