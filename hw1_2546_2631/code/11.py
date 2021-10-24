from random import randint

def vector_mul(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]

def hardlim(n):
    if n < 0:
        return 0
    return 1

def Main():
    #Define vectors, bias, weights and target
    p = [[1,0],[1,-1],[0,-1],[0,1],[-1,0], [1,1]]
    t = [0,0,0,1,1,1]
    w = [0,0]
    b = 1

    #Start training
    while True:
        counter = 0
        for i in range(6):
            a = hardlim(vector_mul(w,p[i]) + b)
            #Calculate error
            e = t[i] - a
            if e == 0:
                counter += 1
            #Recalculating weight and bias
            w[0] = w[0] + e*p[i][0]
            w[1] = w[1] + e*p[i][1]
            b = b + e
        #If error is equal to 0 for all vectors the training is over 
        if counter == 6:
            break

    print (f"Final weight: {w} and bias: {b}")

    p = [[1,0],[1,-1],[0,-1],[0,0],[0,1],[-1,0], [1,1]]
    t = [0,0,0,0,1,1,1]


    #Start training
    while True:
        counter = 0
        for i in range(7):
            a = hardlim(vector_mul(w,p[i]) + b)
            #Calculate error
            e = t[i] - a
            if e == 0:
                counter += 1
            #Recalculating weight and bias
            w[0] = w[0] + e*p[i][0]
            w[1] = w[1] + e*p[i][1]
            b = b + e
        #If error is equal to 0 for all vectors the training is over 
        if counter == 7:
            break

    print (f"Final weight: {w} and bias: {b}")



if __name__ == "__main__":
    Main()
