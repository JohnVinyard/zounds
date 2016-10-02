
def tuplify(a):
    try:
        return tuple(a)
    except TypeError:
        return (a,)


