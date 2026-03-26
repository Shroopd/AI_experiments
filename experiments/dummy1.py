

print(1)
try:
    print(2)
    print(3)
    raise ValueError
    print(4)
except:
    print(5)
    raise ValueError
finally:
    print(6,6)