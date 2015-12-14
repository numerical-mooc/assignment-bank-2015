import numpy
""" This script contains all of the free stream values necessary to initialize 
the problem and the dimensions and number of grid points of the computational domain.
If you wish to see the contents of this script, run the following command in a cell of 
your notebook: %load parameters.py
"""
# Initial Conditions
#
#Velocity
M = 4.0 #mach number
c_inf = 340.28 #speed of sound m/s
u_inf = M*c_inf
v_inf = 0.0
V_inf = numpy.sqrt(u_inf**2 + v_inf**2 )
#
#Eqn state
R = 287 # J/(kg*K)
T_inf = 288.16 #Kelvin
P_inf = 101325 #N/m^2
rho_inf = P_inf/(R*T_inf)
Pr = 0.71 #prandtl no.
#
#viscosity
dyn_visc = 1.7894e-5 #kg/(m*s)
lmbda = -(2./3.)*dyn_visc # second viscosity coefficient
#
#specific heat
gamma = 1.4
cv = R/(gamma-1.) #pg 88 anderson
cp = gamma*cv
#
#Internal Energy
e_inf = cv*T_inf
#Boundary Conditions
T_wall = T_inf #Kelvin
#
# #Parameters for GRID
nx = 70 #number of x grid points
ny = 70 #number of y grid points 
Re = 1000
L = (Re*dyn_visc)/(rho_inf*V_inf) #length of plate
Y = (25*L)/numpy.sqrt(Re)
#space steps:
dx = L/(nx-1)
dy = Y/(ny-1)

#
nt = 5000 #used to run for-loop (can be changed, large enough to ensure convergence)

#
# L = 0.00001
# dx = 0.145e-6
# dy = 0.119e-6
# Y = 69*dy
# Re = (rho_inf*u_inf*L)/dyn_visc
