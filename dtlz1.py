# g = 100 * (len(individual[obj - 1:]) + sum((xi - 0.5)**2 - cos(20 * pi * (xi - 0.5)) for xi in individual[obj - 1:]))
#     f = [0.5 * reduce(mul, individual[:obj - 1], 1) * (1 + g)]
#     f.extend(0.5 * reduce(mul, individual[:m], 1) * (1 - individual[m]) * (1 + g) for m in reversed(range(obj - 1)))
#     return f