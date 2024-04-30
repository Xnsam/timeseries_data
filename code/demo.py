# def decorator(func):
#     def wrapper(*args, **kwargs):
#         print("wrapper start")
#         func(*args, **kwargs)
#         print("wrapper stop")
#     return wrapper

# @decorator
# def test_sum(a, b):
#     print("sum ", a + b)

# test_sum(1, 2)

# gen_1 = (i for i in range(10))
# print(next(gen_1))
# print(next(gen_1))

# f(n) = f(n-1) + f(n-2)

# def fibo(n1, n2, length):
#     if length == 0:
#         return 0
#     print(n1 + n2)
#     return fibo(n2, n2+n1, length-1)


# print(fibo(0, 1, 10))

# from collections import Counter
# import numpy as np

# def check_unique(data_list):
#     if len(data_list) == len(list(set(data_list))):
#         return True
#     else:
#         return False

# print(check_unique([i for i in range(10)]))
# print(check_unique([1, 1, 2, 3, 4, 5, 6]))


from collections import Counter

a = {"k1": 1, "k2": 2}
b = {"k1": 3, "k2": 3, "k3": 4}
new_dict = Counter(a) + Counter(b)
print(new_dict)
