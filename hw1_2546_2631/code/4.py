import matplotlib.pyplot as plt

def Main():



    x = [0.3]
    for i in range(100):
        x_new = x[i]*(2-x[i]*x[i])
        x.append(x_new)

    print (f"x = {x}")
    #Design graph
    plt.plot(x)
    plt.show()



    x = [0.3]
    for i in range(100):
        x_new =  x[i]*(3-x[i]*x[i])
        x.append(x_new)

    print (f"x = {x}")
    #Design graph
    plt.plot(x)
    plt.show()

    x = [0.3]
    for i in range(100):
        x_new = x[i]*(4-x[i]*x[i])
        x.append(x_new)

    print (f"x = {x}")
    #Design graph
    plt.plot(x)
    plt.show()

if __name__ == "__main__":
    Main()