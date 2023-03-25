# create a list of 100 numbers
numbers = [x for x in range(1, 101)]

def check_all_numbers_are_even(numbers):
    # check if all numbers in the list are even
    for number in numbers:
        if number % 2 != 0:
            return False
    return True

# check if all nu

