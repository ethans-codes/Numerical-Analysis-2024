import numpy as np 
  
a = 0
b = np.pi 
N = 10
  
ar = np.zeros(N) 
  
print("Random numbers :")
for i in range (len(ar)): 
    ar[i] = np.random.uniform(a, b)
    print(ar[i]) 
  
integral = 0.0
  
def f(x): 
    return np.sin(x) 

print("value")  
for i in ar: 
    print(f(i))
    integral += f(i)
  
print(integral)
ans = ((b-a)/float(N))*integral 
  
error = np.sqrt((b-a)*(np.sin(ar)*np.sin(ar)).sum()/N - (b-a)*np.sin(ar).mean()**2)/np.sqrt(N)

print ("The value calculated by monte carlo integration is {}.".format(ans))
print("The error in the monte carlo integration is {}.".format(error))