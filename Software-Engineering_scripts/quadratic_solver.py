# Import everything in the math module
from math import *


def quadratic_solver(a, b, c):
    '''
    This function returns the solution(s) for quadratic equations of standard type: ax**2 + bx + c = 0
    where a, b and c are real numbers
    '''

    # Calculate the discriminant 'delta'
    delta = b**2 - 4*a*c

    # Initialize the variable 'result'
    result = None

    # Find the solution(s) based on the nature of delta:
    # delta = 0 --> 1 solution
    # delta < 0 --> 0 solution
    # delta > 0 --> 2 solutions
    if(delta == 0):
        result = -b / (2*a), None
    elif(delta < 0):
        result = float("nan"), None
    else:
        result = ((-b - sqrt(delta)) / (2*a), (-b + sqrt(delta)) / (2*a))
    return result


# Test the function and display the result(s)
sol1, sol2 = quadratic_solver(a=4, b=7, c=1)
print('The solutions of the quadratic equation given are {0} and {1}.'.format(sol1, sol2))
