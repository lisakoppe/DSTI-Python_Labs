# Factorial function


def Factorial(n: int):
    result = 1
    if (n > 1):
        for i in range(n, 1, -1):
            result = result * i
    elif (n < 0):
        raise Exception("The value {} cannot be raised to the factorial".format(n))
    return result


try:
    print(Factorial(n=7))
except ValueError:
    print("Could not compute Factorial")
