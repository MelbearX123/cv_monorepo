# Day 1
# Write a function that takes another function as an argument and applies it to a list
# — understand first-class functions.
list = [1, 2, 3]


def func2(listy):
    return [x**2 for x in listy]


def func1(func2, listy):
    return func2(listy)


# Rewrite three for-loops as list comprehensions. Then rewrite as dict comprehensions.
# 1. Basic transform
numbers = [1, 2, 3, 4, 5]
doubled = []
for n in numbers:
    doubled.append(n * 2)

# 2. Filtering
words = ["apple", "banana", "fig", "kiwi", "watermelon"]
long_words = []
for word in words:
    if len(word) > 4:
        long_words.append(word)

# 3. Transform + filter together
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
squared_evens = []
for n in numbers:
    if n % 2 == 0:
        squared_evens.append(n**2)

doubled = [num * 2 for num in numbers]
long_words = [word for word in words if len(word) > 4]
squared_evens = [num**2 for num in numbers if num % 2 == 0]

# 1. Basic key-value build
names = ["alice", "bob", "charlie"]
name_lengths = {}
for name in names:
    name_lengths[name] = len(name)

# 2. Filtering
scores = {"alice": 85, "bob": 42, "charlie": 91, "diana": 38}
passing = {}
for name, score in scores.items():
    if score >= 50:
        passing[name] = score

# 3. Transform keys and values
prices = {"apple": 1.0, "banana": 0.5, "cherry": 2.0}
discounted = {}
for item, price in prices.items():
    discounted[item.upper()] = round(price * 0.9, 2)

name_lengths = {name: len(name) for name in names}
passing = {name: score for (name, score) in scores.items() if score >= 50}
discounted = {item.upper(): round(price * 0.9, 2) for (item, price) in prices.items()}

# Write a context manager class using __enter__ and __exit__ that times a code block.
import time


class Timer:
    def __enter__(self):
        self.start = time.time()  # record current time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()  # record current time
        print(f"Elapsed: {self.end - self.start} seconds")


with Timer() as t:
    # simulate work
    for _ in range(1000000):
        pass


# Deliberately trigger the mutable default argument bug, then fix it with None as the default.
def append(number, alist=[]):
    alist.append(number)
    return alist


# Now fix
def append(number, alist=None):
    if alist is None:
        alist = []
    alist.append(number)
    return alist


# Write a decorator @timer that prints execution time of any function — you'll use this to benchmark CV code.


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed: {end - start} seconds")
        return result

    return wrapper


@timer
def slow_func():
    for i in range(1000):
        pass


slow_func()

# Practice reading tracebacks: introduce 5 different errors
# (TypeError, IndexError, AttributeError, ValueError, KeyError)
# and get fast at reading what went wrong and where.
