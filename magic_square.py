def ramanujan_magic_square(a, b, c, d):
    # create a 4x4 matrix with all zeros
    magic_square = [[0 for x in range(4)] for y in range(4)]
    
    # set user inputs on the first row
    magic_square[0][0] = a
    magic_square[0][1] = b
    magic_square[0][2] = c
    magic_square[0][3] = d
    
    # calculate the remaining elements using Ramanujan's algorithm
    magic_constant = 139
    magic_square[1][0] = int(magic_constant - (a + d))
    magic_square[2][0] = int(magic_constant - (b + c))
    magic_square[3][0] = int(magic_constant - (magic_square[1][0] + magic_square[2][0]))
    
    magic_square[1][1] = int(magic_constant - (magic_square[1][0] + magic_square[0][2]))
    magic_square[2][2] = int(magic_constant - (magic_square[2][0] + magic_square[0][3]))
    magic_square[3][3] = int(magic_constant - (magic_square[3][0] + magic_square[3][1]))
    
    magic_square[1][3] = int(magic_constant - (magic_square[0][0] + magic_square[2][2]))
    magic_square[2][1] = int(magic_constant - (magic_square[0][1] + magic_square[3][1]))
    magic_square[3][2] = int(magic_constant - (magic_square[0][2] + magic_square[1][2]))
    
    magic_square[1][2] = int(magic_constant - (magic_square[1][0] + magic_square[1][1] + magic_square[1][3]))
    magic_square[2][3] = int(magic_constant - (magic_square[2][0] + magic_square[2][1] + magic_square[2][2]))
    magic_square[3][1] = int(magic_constant - (magic_square[3][0] + magic_square[3][2] + magic_square[3][3]))
    
    # return the magic square
    return magic_square

# example usage
inputs = input("Enter 4 unique integers between 1 and 100, separated by spaces: ")
a, b, c, d = [int(x) for x in inputs.split()]
magic_square = ramanujan_magic_square(a, b, c, d)
print("Ramanujan's magic square:")
for row in magic_square:
    print(row)
