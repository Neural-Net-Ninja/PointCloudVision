# convert decimal to binary, and make sure the input is integer. otherwise raise error
def dec2bin(x):
    if not isinstance(x, int):
        raise TypeError("expected int, got %s" % type(x))
    if x < 0:
        raise ValueError("must supply a positive integer")
    return bin(x)[2:]

print(dec2bin(10))
