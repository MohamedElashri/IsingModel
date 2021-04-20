# IsingModel

## Project 
2d Ising model Monte-Carlo Simulation

Author: Mohamed Elashri 

Email: elashrmr@mail.uc.edu

## Goal 

Apply the MC methods using Metropolis Algorithm to ising model and extract physical paramters (Energy, Specific heat and Magnetization)

## Algorithm 
  1. Prepare some initial configrations of N spins. 
  2. Flip spin of a lattice site chosen randomly 
  3. Calculate the change in energy due to that 
  4. If this change is negative, accept such move. If change is positive, accept it with probability exp^{-dE/kT}
  5. repeat 2-4. 
  6. calculate Other parameters and plot them 

My code is very well commented with almost every part have comment that explain its function. reading the code should be something easy, just grab a coffee and enjoy. I always hated people who don't write enough comments or worse than that don't write any. This encouraged me to always to write comments with much details as much as possible. 

## Physical Model
Lattice is a periodical structure of points that align one by one. 2D lattice can be plotted as: 

```
* * * * * * * *   
* * * * * * * * 
* * * * * * * *
* * * * * * * *
* * * * * * * *
```

The points in lattice are called lattice points, neareast lattice points of point ^ are those lattice points denoted by (*) shown in the graph below:
```
* * *(*)* * * *
* *(*)^(*)* * *
* * *(*)* * * *
* * * * * * * *
```
Each lattice point is denoted by a number i in the Harmitonian.

The expression for the Energy of the total system is 

<img align="left" src="https://latex.elashri.xyz/cgi-bin/mimetex.cgi?H%20=%20-%20J%20%5Csum_%7B%20i%20=%200%20%7D%5E%7B%20N-1%20%7D%20%5Csum_%7B%20j%20=%200%20%7D%5E%7B%20N-1%20%7D%20(s_%7Bi,j%7Ds_%7Bi,j+1%7D+s_%7Bi,j%7Ds_%7Bi+1,j%7D)">  

```
(H = - J \sum_{ i = 0 }^{ N-1 } \sum_{ j = 0 }^{ N-1 } (s_{i,j}s_{i,j+1}+s_{i,j}s_{i+1,j}) )
```



```
* * * * * * * * 
* * * * * * * *
* * * * * * * * <-the i-th lattice point
* * * * * * * *
* * * * * * * *
```

Periodical strcture means that lattice point at(1,1) is the same as that at(1,9) if the lattice is 5 by 8. more e.g.(1,1)<=>(6,1),
(2,3)<=>(2,11). A 2D lattice can be any Nx by Ny. The location (x,y) here is another denotion of lattice point that 
is fundementally same as i-th lattice point denotation above.

```
* * * * * * * * 4
* * * * * * * * 3
* * * * * * * * 2
* * * * * * * * 1
1 2 3 4 5 6 7 8 
```

## Results 

These are plots of the physical quantites for different MC steps. 

![250000 steps](./plots/plot_1.png)
![500000 steps](./plots/plot_2.pdf)


## Reproduction  
You can run the Jupyter Notebook provided on Colab directly or you can download and run locally. Also there is python script that you can run. I'm using Numba cache so that it produces cache folder (specific to machine CPU and configration) so that you will need to produce yours by running it for one time and susequent runs will be about double faster.  

  <tr>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/MohamedElashri/IsingModel/blob/main/Ising.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" height = '23px' >
    </a></td>
  </tr>



## Optimization
I spent many nights working on this work, most of time I needed to opptimize my code, 
I even tried to move to matlab (last time I used it was like 5 years ago). 
But I learned a nice thing from my desire to optimize code speed. it is the usage of Numba’s JIT compiler. read more about that here [link](http://melashri.net/url/b).
I also instead of using multiple nested loops I dragged all these into just one.
Imagine running 50x50 lattice simulation in my older codes for hours (one took 6 hours) vs 15 minutes for the currect script. (On my Mac m1 Machine). 
also I made the code avilable on colab and can be accssed here (without much comments) [link](http://melashri.net/url/c).

## Numba on Apple silicon (Mac m1)
Assuming that we are using Python3 version from homebrew not the one comes with OS which we shouldn't work with or try to modify except for Mac OS stuff. This can be done by adding the path to `.bashrc` or `.zshrc` by adding `export PATH="/usr/local/opt/python/libexec/bin:$PATH"` line to the either files.
 
 
To install Numba on Mac m1 we do the following 

```
python3 -m pip install  conda 
```

```
python3 -m pip install  cytoolz
```

```
python3 -m conda config --add channels conda-forge
```


```
python3 -m  conda install -c numba numba
```



