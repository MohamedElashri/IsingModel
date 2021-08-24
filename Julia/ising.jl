# ----------------------------------------------------------------
# The required packages
using Printf
using Plots
using ProgressBars
#----------------------------------------------------------------


# ------------------------------------------------------------------------------
# To download packages from this scripts directly uncomment the following lines:
#using Pkg
#Pkg.add("Plots")
#Pkg.add("ProgressBars")
# ------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# lets define the constants
const L       = 50             # Define size of lattice
const n_sweep = 20             # Define number of sweeps between sampling
const n_therm = 100       # Define number of sweeps 
const n_data  = 10000            # Define number of data samples per temperature
const temps   = 4.0:-0.3:0.1   # Define temperatures range 
# ------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# This function measures the i'th sample of energy and magnetization
function measure(i, energy, magnetization, s)      
    en = 0
    m = 0
    for x = 1:L
        for y = 1:L
            u = 1+mod(y,L) # up 
            r = 1+mod(x,L) # right 
            en -= s[x,y]*(s[x,u]+s[r,y]) # energy
            m  += s[x,y]                 # magnetization
        end
    end
    energy[i] = en
    magnetization[i] = abs(m)
end
# ---------------------------------------------------------------------------------------------------------------------

# 
# ----------------------------------------------------------------------------------------
 # implementing flip function to apply metropolis spin flip algorithm to site (x,y) / temp T
function flip(x, y, T, s)
    u = 1+mod(y,L)   # up
    d = 1+mod(y-2,L) # down
    r = 1+mod(x,L)   # right
    l = 1+mod(x-2,L) # left
    de = 2*s[x,y]*(s[x,u]+s[x,d]+s[l,y]+s[r,y])
    if (de < 0)
        s[x,y] = -s[x,y]
    else
        p = rand()
        if (p < exp(-de/T))
            s[x,y] = -s[x,y]
        end
    end
end
# ----------------------------------------------------------------------------------------


# --------------------------------------------------------------------
# Implementing sweeb function to apply  flip function to every site on the lattice
function sweep(n, T, s) 
    for i = 1:n
        for x = 1:L
            for y = 1:L
                flip(x,y,T, s)
            end
        end
    end
end
# ----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Implement main julia function for algorithm opreating and calculation of physical quantities
function main()
    e1 = Array(1:n_data)     # This array purpose is to hold energy measurements (fixed T)
    m1 = Array(1:n_data)     # This array purpose is to hold magnetization measurements (fixed T)
    en = []                  # This array purpose is to append average energy at each T
    mz = []                  # This is magnetization array
    s  = ones(Int32,L,L)     # lattice of Ising spins (+/-1)
    for T in tqdm(temps)              # Lets loop over temperatures range
        sweep(n_therm, T, s)    # Sweebs over the lattice to assign tempreatures
        energy        = e1      # Reset energy measurement array
        magnetization = m1      # Reset magnetization measurement array
        for i = 1:n_data        # Take n_data measurements w/ n_sweep 
            sweep(n_sweep, T, s)   
            measure(i, energy, magnetization, s)
        end
        avg_en = sum(energy)/n_data           # calculate average energy
        avg_ma = sum(magnetization)/n_data    # calculate average magnetization
        push!(en,avg_en/(L*L))                # Add those to the list
        push!(mz,avg_ma/(L*L))
        # @printf("%8.3f  %8.3f \n", avg_en/(L*L), avg_ma/(L*L)) # using this for debugging code purposes to see values.
    end
# ----------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting section 
    gr() # Set the backend to GR
    plot(temps,mz,title = "Magnetization vs Temperature",xlabel = "Temperature",ylabel = "Magnetization",label = "Magnetization") # plot Magnetization vs Temperature
    savefig("Magnetization.pdf")
    plot(temps,en,title = "Energy vs Temperature",xlabel = "Temperature",ylabel = "Energy",label="Energy") # plot Energy vs. Temperature
    savefig("Energy.pdf")
end
# ------------------------------------------------------------------------------------------------------------------------------------------------

main()
