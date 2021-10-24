import matplotlib.pyplot as plt
import numpy as np


def function(x1, x2):
    return 3/2*x1*x1 + 3/2*x2*x2 + x1*x2 + x1 + 2*x2 +2

def dfx1(x1, x2):
    return 3*x1 + x2 + 1

def dfx2(x1, x2):
    return 3*x2 + x1 + 2

def dx1(gamma,xd_old,a,g):
    return   gamma * xd_old[0] -(1-gamma)*a*g[0]

def dx2(gamma,xd_old,a,g):
    return   gamma * xd_old[1] -(1-gamma)*a*g[1]


def Main():
    g=[0,0]
    dx=[0,0]
    x_new=[0,0]
    a=0.5
    gamma=0.1
    h=1.5
    r=0.5
    z=0.04

    #Create vector x for x0 = 0.5
    x = [-2 ,-1.5]
    xd_old=[0,0]

    for i in range(2):

       fx=function(x[0],x[1])

       g[0]=dfx1(x[0],x[1])
       g[1]=dfx2(x[0],x[1])

       dx[0]=dx1(gamma,xd_old,a,g)
       dx[1]=dx2(gamma,xd_old,a,g)

       x_new[0]=x[0]+dx[0]
       x_new[1]=x[1]+dx[1]

       fx_new=function(x_new[0],x_new[1])
       print(fx_new)
      
       if fx_new > fx * z:
           x_new=x
           a=r*a
           gamma=0
       elif fx_new < fx:
            x=x_new
            a=n*a
            gamma=0.1
       else :
            x=x_new
            gamma=0.1
        
       xd_old=dx

    print (f"x{i} = {x}")
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    start, stop, n_values = -8, 8, 800

    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)


    Z = 3/2*X*X + 3/2*Y*Y + X*Y + X + 2*Y +2

    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)

    ax.set_title('Contour Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()



if __name__ == "__main__":
    Main()
