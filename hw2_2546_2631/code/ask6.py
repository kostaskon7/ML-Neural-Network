import matplotlib.pyplot as plt


from math import sin, pi, exp
from random import uniform

#Initialize input vectors
p = [-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2]
learning_rate = 0.01

def g_function(p):
    return 1+sin(p*(pi/2))

def logsigmoid(n):
    return 1/(1+exp(-n))

def purelin(n):
    return n

def purelin_der(n):
    return 1

def logsigmoid_der(n):
    return exp(-n)/((1+exp(-n))*(1+exp(-n)))

S = 3

#Initialize weights and biases
w1 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
b1 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
w2 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
b2 = uniform(-0.5, 0.5)

#Start training
while True:
    s = 0
    for i in range(11):
        n1 = []
        a1 = []
        n2 = b2
        for j in range(S):
            n = p[i]*w1[j]+b1[j]
            n1.append(n)
            a = logsigmoid(n)
            a1.append(a)
            n2 += a * w2[j]
        a2 = purelin(n2)

        #Calculate error
        e = g_function(p[i])-a2
        s = s + e*e

        #calculate sensitivities and recalculate weghts and biases
        s2 = -2*purelin_der(n2)*(e)
        s1 = []
        for j in range(S):
            s1.append(logsigmoid_der(n1[j])*w1[j]*s2)

            w2[j] -= learning_rate*s2*a1[j]
        b2 -= learning_rate*s2

        for j in range(S):
            w1[j] -= learning_rate*s1[j]*p[i]
            b1[j] -= learning_rate*s1[j]

    #Check sum square error threshold
    if s <= 1.5:
        break

print (f"Final weight1: {w1} and bias1: {b1}")
print (f"Final weight2: {w2} and bias2: {b2}")

#Classify vectors
result = []
g = []
for i in range(11):
    for j in range(S):
        n = p[i]*w1[j]+b1[j]
        a = logsigmoid(n)
        n2 += a * w2[j]
    a2 = purelin(n2)
    result.append(a2)
    g.append(g_function(p[i]))
    print (f"P{i} is {a2} and is supposed to be {g_function(p[i])}")

#Design plot
plt.plot(p, g)
plt.plot(p, result)
plt.show()


S = 15

#Initialize weights and biases
w1 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
b1 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
w2 = [uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)]
b2 = uniform(-0.5, 0.5)

#Start training
while True:
    s = 0
    for i in range(11):
        n1 = []
        a1 = []
        n2 = b2
        for j in range(S):
            n = p[i]*w1[j]+b1[j]
            n1.append(n)
            a = logsigmoid(n)
            a1.append(a)
            n2 += a * w2[j]
        a2 = purelin(n2)

        #Calculate error
        e = g_function(p[i])-a2
        s = s + e*e

        #calculate sensitivities and recalculate weghts and biases
        s2 = -2*purelin_der(n2)*(e)
        s1 = []
        for j in range(S):
            s1.append(logsigmoid_der(n1[j])*w1[j]*s2)

            w2[j] -= learning_rate*s2*a1[j]
        b2 -= learning_rate*s2

        for j in range(S):
            w1[j] -= learning_rate*s1[j]*p[i]
            b1[j] -= learning_rate*s1[j]

    #Check sum square error threshold
    if s <= 1.6:
        break

print (f"Final weight1: {w1} and bias1: {b1}")
print (f"Final weight2: {w2} and bias2: {b2}")

#Classify vectors
result = []
g = []
for i in range(11):
    for j in range(S):
        n = p[i]*w1[j]+b1[j]
        a = logsigmoid(n)
        n2 += a * w2[j]
    a2 = purelin(n2)
    result.append(a2)
    g.append(g_function(p[i]))
    print (f"P{i} is {a2} and is supposed to be {g_function(p[i])}")

#Design plot
plt.plot(p, g)
plt.plot(p, result)
plt.show()
