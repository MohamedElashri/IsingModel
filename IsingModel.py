 #coding: utf-8
#----------------------------------------------------------------------
## 2d Ising model Monte-Carlo Simulation
## Author: Mohamed Elashri 
## Email: elashrmr@mail.uc.edu
##  Algorithm 
##  1- Prepare some initial configrations of N spins. 
##  2- Flip spin of a lattice site chosen randomly 
##  3- Calculate the change in energy due to that 
##  4- If this change is negative, accept such move. If change is positive, accept it with probability exp^{-dE/kT}
##  5- repeat 2-4. 
##  6- calculate Other parameters and plot them 
#----------------------------------------------------------------------

'''
Lattice is a periodical structure of points that align one by one. 2D lattice can be plotted as: 

* * * * * * * *   
* * * * * * * * 
* * * * * * * *
* * * * * * * *
* * * * * * * *

The points in lattice are called lattice points, neareast lattice points of point ^ are those lattice points denoted by (*) shown in the graph below:

* * *(*)* * * *
* *(*)^(*)* * *
* * *(*)* * * *
* * * * * * * *

Each lattice point is denoted by a number i in the Harmitonian.

The expression for the Energy of the total system is (online latex formula)
http://melashri.net/url/a or (H = - J \sum_{ i = 0 }^{ N-1 } \sum_{ j = 0 }^{ N-1 } (s_{i,j}s_{i,j+1}+s_{i,j}s_{i+1,j}) )

* * * * * * * * 
* * * * * * * *
* * * * * * * * <-the i-th lattice point
* * * * * * * *
* * * * * * * *

Periodical strcture means that lattice point at(1,1) is the same as that at(1,9) if the lattice is 5 by 8. more e.g.(1,1)<=>(6,1),
(2,3)<=>(2,11). A 2D lattice can be any Nx by Ny. The location (x,y) here is another denotion of lattice point that 
is fundementally same as i-th lattice point denotation above.s

* * * * * * * * 4
* * * * * * * * 3
* * * * * * * * 2
* * * * * * * * 1
1 2 3 4 5 6 7 8 

'''


'''
I spent many nights working on this work, most of time I needed to opptimize my code, 
I even tried to move to matlab (last time I used it was like 5 years ago). 
But I learned a nice thing from my desire to optimize code speed. it is the usage of Numba’s JIT compiler. read more about that here [link](http://melashri.net/url/b).
I also instead of using multiple nested loops I dragged all these into just one.
Imagine running 50x50 lattice simulation in my older codes for hours (one took 6 hours) vs 15 minutes for the currect script. (On my Mac m1 Machine). 
also I made the code avilable on colab and can be accssed here (without much comments) [link](http://melashri.net/url/c).
'''


#----------------------------------------------------------------------
##  Import needed python libraries 
#----------------------------------------------------------------------
import matplotlib.pyplot as plt
from numba import jit # wonderful optimization compiler. 
import numpy as np  # we can't work in python without that in physics (maybe we can if we are not lazy enough)
import random # we are doing MC simulation after all!!
import time # for time estimation 
from tqdm import tqdm # fancy progress bars for loops , use this line if working with .py script
#from tqdm.notebook import tqdm # use this if working with jupyter notebook to avoid printing a new line for each iteration
from rich.progress import track # we can use amazin rich library to do the same (more interesting output) change "tqdm" later to "track"

# Define parameters 

B = 0; # Magnetic field strength
L = 50; # Lattice size (width)
s = np.random.choice([1,-1],size=(L,L)) # Begin with random spin sites with values (+1 or -1) for up or down spins. 
n= 1000 * L**2 # number of MC sweeps 
Temperature = np.arange(1.6,3.25,0.01) # Initlaize temperature range (the range includes critical temperature) > takes form np.arange(start,stop,step)
  


'''
Energy of the lattice calculations. 
The energy here is simply the sum of interactions between spins divided by the total number of spins
'''
@jit(nopython=True, cache=True) # wonderful jit optimization compiler in its high performance mode
def calcE(s):
    E = 0
    for i in range(L):
        for j in range(L):
            E += -dE(s,i,j)/2
    return E/L**2

'''
Calculate the Magnetization of a given configuration
Magnetization is the sum of all spins divided by the total number of spins

'''
@jit(nopython=True, cache=True) # wonderful jit optimization compiler in its high performance mode
def calcM(s):
    m = np.abs(s.sum())
    return m/L**2

# Calculate interaction energy between spins. Assume periodic boundaries
# Interaction energy will be the difference in energy due to flipping spin i,j 
# (Example: 2*spin_value*neighboring_spins)
@jit(nopython=True, cache=True) # wonderful jit optimization compiler in its high performance mode
def dE(s,i,j): # change in energy function
    #top
    if i == 0:
        t = s[L-1,j]  # periodic boundary (top)
    else:
        t = s[i-1,j]
    #bottom
    if i == L-1:
        b = s[0,j]  # periodic boundary (bottom)
    else:
        b = s[i+1,j]
    #left
    if j == 0:
        l = s[i,L-1]  # periodic boundary (left)
    else:
        l = s[i,j-1]
    #right
    if j == L-1:
        r = s[i,0]  # periodic boundary  (right)
    else:
        r = s[i,j+1]
    return 2*s[i,j]*(t+b+r+l)  # difference in energy is i,j is flipped

# Monte-carlo sweep implementation
@jit(nopython=True, cache=True) # wonderful jit optimization compiler in its high performance mode
def mc(s,Temp,n):   
    for m in range(n):
        i = random.randrange(L)  # choose random row
        j = random.randrange(L)  # choose random column
        ediff = dE(s,i,j)
        if ediff <= 0: # if the change in energy is negative
            s[i,j] = -s[i,j]  # accept move and flip spin
        elif random.random() < np.exp(-ediff/Temp): # if not accept it with probability exp^{-dU/kT}
            s[i,j] = -s[i,j]
    return s

# Compute physical quantities
@jit(nopython=True, cache=True) # wonderful jit optimization compiler in its high performance mode
def physics(s,T,n):
    En = 0
    En_sq = 0
    Mg = 0
    Mg_sq = 0
    for p in range(n):
        s = mc(s,T,1)
        E = calcE(s)
        M = calcM(s)
        En += E
        Mg += M
        En_sq += E*E
        Mg_sq += M*M
    En_avg = En/n
    mag = Mg/n
    CV = (En_sq/n-(En/n)**2)/(T**2)
    return En_avg, mag, CV

# Inititalize magnetization, average energy and heat capacity
mag = np.zeros(len(Temperature))
En_avg = np.zeros(len(Temperature))
CV = np.zeros(len(Temperature))

start = time.time()

# Simulate at particular temperatures (T) and compute physical quantities (Energy, heat capacity and magnetization)
for ind, T in enumerate(track(Temperature)):
    # Sweeps spins
    s = mc(s,T,n)
    # Compute physical quanitites with 1000 sweeps per spin at temperature T
    En_avg[ind], mag[ind], CV[ind] = physics(s,T,n)
end = time.time()
print("Time it took in seconds is = %s" % (end - start))

time = (end - start)/60
print('It took ' + str(time) + ' minutes to execute the code')

# It took about 30 minutes on my Mac meachine (not bad) for n =1000* L^2
# and abput 2 hours for n =2000*L^2 with double T points



#----------------------------------------------------------------------
#  Plotting area
#----------------------------------------------------------------------


f = plt.figure(figsize=(18, 10)); # plot the calculated values  
sp =  f.add_subplot(2, 2, 1 );
plt.plot(Temperature, En_avg, marker='.', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(2, 2, 2 );
plt.plot(Temperature, abs(mag), marker='.', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

sp =  f.add_subplot(2, 2, 3 );
plt.plot(Temperature, CV, marker='.', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

plt.subplots_adjust(0.12, 0.11, 0.90, 0.81, 0.26, 0.56)
plt.suptitle("Simulation of 2D Ising Model by Metropolis Algorithm\n" + "Lattice Dimension:" + str(L) + "X" + str(
    L) + "\n" + "External Magnetic Field(B)=" + str(B) + "\n" + "Metropolis Step=" + str(n))



plt.show() # function to show the plots
