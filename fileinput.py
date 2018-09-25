import numpy as np

def get_array(filename="somecode.txt"):

    with open(filename,'r') as file:
        while True:
            c=file.read(1)
            if not c:
                break
            dim=ord(c)
            thisarr=np.zeros([256])
            thisarr[dim]=1
            yield thisarr
def main():
    arrgen=get_array()
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
    print(next(arrgen))
