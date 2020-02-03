# Fibonacci sequence applied from 0 to 10
a, b = 0, 1
for i in range(0, 10):
    print(a)
    a, b = b, a + b


# Fibonacci generator


def fibo(num):
    for i in range(0, num):
        yield "{}: {}".format(i+1, a)
        a, b = b, a + b


for item in fibo(10):
    print(item)
