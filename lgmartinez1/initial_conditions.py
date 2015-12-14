import numpy
def get_Uinitial(numx, numy, rho_fs, ufs, vfs, Vfs, Tfs, efs, Tw, c_v, R):
    """
    Parameters:
    -----------
    free stream values of:
    density, xvel, yvel, vel magnitude, temperature, 
    internal energy, and wall temperature, and
    constant volume spefific heat
    R - gas constant
    """
    #---------------------------------------------
    U_start = numpy.zeros((4,numx,numy),dtype=float)
    #
    #DENSITY
    U_start[0,:,1:] = rho_fs 
    U_start[0,0,0] = rho_fs #leading edge
    U_start[0,1:,0] = (rho_fs*R*Tfs)/(R*Tw) #plate
    #
    #X-MOMENTUM
    U_start[1,0,1:] = rho_fs*ufs #inflow
    #U_start[1,1:,1:] = 0.0 #the plate is stationary
    U_start[1,1:,1:] = rho_fs*ufs 
    U_start[1,:,0] = 0.0 #no slip on plate including leading edge
    #
    #Y-MOMENTUM
    U_start[2,0,1:] = rho_fs*vfs #inflow
    #U_start[2,1:,1:] = 0.0 #the plate is stationary
    U_start[2,1:,1:] = rho_fs*vfs
    U_start[2,:,0] = 0.0 #no slip on plate including leading edge
    #
    #ENERGY
    U_start[3,0,1:] = (efs + 0.5*(Vfs**2))*rho_fs #inflow
    #U_start[3,1:,1:] = (efs)*rho_fs
    U_start[3,1:,1:] = (efs + 0.5*(Vfs**2))*rho_fs
    U_start[3,1:,0] = (Tw*c_v)*rho_fs #no slip plate
    U_start[3,0,0] = (Tfs*c_v)*rho_fs #no slip plate
    
    return U_start   
    