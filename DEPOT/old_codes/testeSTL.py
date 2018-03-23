import numpy as np
from struct import unpack

def BinarySTL(fname):
    fp = open(fname, 'rb')
    Header = fp.read(80)
    nn = fp.read(4)
    Numtri = unpack('i', nn)[0]
    #print nn
    record_dtype = np.dtype([
        ('normals', np.float32, (3,)),
        ('Vertex1', np.float32, (3,)),
        ('Vertex2', np.float32, (3,)),
        ('Vertex3', np.float32, (3,)),
        ('atttr', '<i2', (1,))
    ])
    data = np.fromfile(fp, dtype=record_dtype, count=Numtri)
    fp.close()

    Normals = data['normals']
    Vertex1 = data['Vertex1']
    Vertex2 = data['Vertex2']
    Vertex3 = data['Vertex3']

    p = np.append(Vertex1, Vertex2, axis=0)
    p = np.append(p, Vertex3, axis=0)  # list(v1)
    Points = np.array(list(set(tuple(p1) for p1 in p)))

    return Header, Points, Normals, Vertex1, Vertex2, Vertex3

#########################################

if __name__ == '__main__':
    #fname = "/home/clayton/Downloads/cilindro1.stl"  # "porsche.stl"
    #fname = "cubo.stl"
    fname = "piramideclarabinary.stl"
    head, p, n, v1, v2, v3 = BinarySTL(fname)
    print(head)
    print(p)
    print(v1)
    print(p.shape)