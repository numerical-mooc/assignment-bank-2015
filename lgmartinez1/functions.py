import numpy
#-----------------------------------------------------------------------------------------
def get_Temperature(Uv, numx, numy, Tw, Tfs, c_v):
    """
    This function returns the case of constant plate temperature
    
    Parameters
    ----------------
    Temp - Temeprature in domain
    numx - x grid points
    numy - y grid points
    Tfs - freestream temperature
    Tw - plate temperature
    c_v - constant volume specific heat
    """
    TT = numpy.zeros((numx,numy))
    #
    #interior points
    TT[1:-1,1:-1] = (1/c_v)*\
                    ((Uv[3,1:-1,1:-1]/Uv[0,1:-1,1:-1]) -\
                     0.5*( (Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])**2 +\
                          (Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])**2 ))
    #
    #enforce BCS:
    TT[1:,0] = Tw  #at plate x-line, constant temperature case
    TT[0,0] = Tfs #leading edge
    TT[:,-1] = Tfs #at top boundary of domain
    TT[0,1:] = Tfs  #at inflow y-line
    
    #at outflow:
    TT[-1,1:-1] = 2*TT[-2,1:-1] - TT[-3,1:-1] #at outflow y-line, extrapolate
    #
    return TT
#-----------------------------------------------------------------------------------------
def get_visc(Temp, viscfs, Tfs):
    """
    Sutherland's law
    ----------------
    Temp - Temeprature in domain
    viscfs  - freestream viscosity
    Tfs - freestream temperature
    """
    v = numpy.zeros_like(Temp)
    #
    v[:,1:] = viscfs*((Temp[:,1:]/Tfs)**(3./2.))*(Tfs+110.)/(Temp[:,1:]+110)
    v[1:,0] = viscfs*((Temp[1:,0]/Tfs)**(3./2.))*(Tfs+110.)/(Temp[1:,0]+110) #plate
    v[0,0] = viscfs #leading edge
    #
    return v
#-----------------------------------------------------------------------------------------
def get_k(visc, c_p, Prt):
    """
    Function to obtain thermal conductivity in the flow field
    Parameters
    ----------------
    visc  - viscosity
    c_p - constant pressure specific heat
    Prt - Prandtl number
    """
    #
    kk = numpy.zeros_like(visc)
    kk[:,:] = visc[:,:]*c_p/Prt
    #
    return kk
#-----------------------------------------------------------------------------------------
def get_BC(Uv, Temp, numy, rho_fs, Tw, ufs, c_v, Tfs, R):
    """
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    Temp - Temperature in domain
    numy - y grid points
    rho_fs - free stream density
    Tw - wall temperature
    ufs - freestream x velocity
    c_v - constant volume specific heat
    Tfs - freestream temperature
    R - gas constant
    """
    #
    Ubc = Uv.copy()
    #
    #at outflow, u,v, P, and T are extrapolated, then:
    uout = numpy.zeros((numy-2))
    vout = numpy.zeros((numy-2))
    Pout = numpy.zeros((numy-2))
    #
    uout[:] = (2*Uv[1,-2,1:-1]/Uv[0,-2,1:-1]) - Uv[1,-3,1:-1]/Uv[0,-3,1:-1]
    vout[:] = (2*Uv[2,-2,1:-1]/Uv[0,-2,1:-1]) - Uv[2,-3,1:-1]/Uv[0,-3,1:-1]
    Pout[:] = (2*Uv[0,-2,1:-1]*R*Temp[-2,1:-1]) - Uv[0,-3,1:-1]*R*Temp[-3,1:-1]
    #
    Ubc[0,-1,1:-1] = Pout[:]/(R*Temp[-1,1:-1])
    Ubc[1,-1,1:-1] = Ubc[0,-1,1:-1] * uout[:]
    Ubc[2,-1,1:-1] = Ubc[0,-1,1:-1] * vout[:]
    Ubc[3,-1,1:-1] = Ubc[0,-1,1:-1]*(Temp[-1,1:-1]*c_v +\
                                     0.5*(uout[:]**2 + vout[:]**2))
    
    #density
    Ubc[0,0,0] = rho_fs #leading edge
    Ubc[0,1:-1,0] = (2*R*Uv[0,1:-1,1]*Temp[1:-1,1] -\
                   R*Uv[0,1:-1,2]*Temp[1:-1,2])/(R*Tw) #density at plate[i=1:-1]
    Ubc[0,-1,0] = (2*Pout[0]-Pout[1])/(R*Tw)
    Ubc[0,0,1:] = rho_fs #inflow
    Ubc[0,1:,-1] = rho_fs #top boundary of domain
    #
    #x-mom:
    Ubc[1,:,0] = 0.0 #plate and leading edge
    Ubc[1,0,1:] = rho_fs*ufs #inflow
    Ubc[1,1:,-1] = rho_fs*ufs #top
    #y-mom:
    Ubc[2,:,0] = 0.0 #plate and leading edge
    Ubc[2,0,1:] = 0.0 #inflow
    Ubc[2,1:,-1] = 0.0 #top
    #
    #E_total:
    Ubc[3,0,0] = rho_fs*(c_v*Tfs) #leading edge
    Ubc[3,1:,0] = rho_fs*(c_v*Tw) #plate
    Ubc[3,0,1:] = rho_fs*(c_v*Tfs + 0.5*(ufs**2)) #inflow
    Ubc[3,1:,-1] = rho_fs*(c_v*Tfs + 0.5*(ufs**2)) #top
    #
    #
    return Ubc
#-----------------------------------------------------------------------------------------
def get_tau_xy_Epredict(Uv, visc, numx, numy, delx, dely ):
    """
    xy shear stress for E flux predictor step
    x-backward difference, y-central difference
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    """
    #don't need values for shear for upper boundary, plate, inflow
    #
    tau_xy = numpy.zeros((numx,numy))
    #
    #inner points:
    #i back j central
    tau_xy[1:-1,1:-1] = visc[1:-1,1:-1]*\
                       ( (1/(2*dely))*((Uv[1,1:-1,2:]/Uv[0,1:-1,2:])-\
                                       (Uv[1,1:-1,:-2]/Uv[0,1:-1,:-2])) +\
                       (1/delx)*((Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])-\
                                 (Uv[2,:-2,1:-1]/Uv[0,:-2,1:-1])) )
    #
    #upper outflow
    #i back j back
    tau_xy[-1,-1] = visc[-1,-1]*((Uv[1,-1,-1]/Uv[0,-1,-1] -\
                                  Uv[1,-1,-2]/Uv[0,-1,-2])/dely +\
                                   (Uv[2,-1,-1]/Uv[0,-1,-1] -\
                                    Uv[2,-2,-1]/Uv[0,-2,-1])/delx) 
    #
    #outflow:
    #i back j central
    tau_xy[-1,1:-1] = visc[-1,1:-1]*((Uv[1,-1,2:]/Uv[0,-1,2:] -\
                                      Uv[1,-1,:-2]/Uv[0,-1,:-2])/(2*dely) +\
                                   (Uv[2,-1,1:-1]/Uv[0,-1,1:-1] -\
                                    Uv[2,-2,1:-1]/Uv[0,-2,1:-1])/delx)
    #
    return tau_xy
#-----------------------------------------------------------------------------------------
def get_tau_xy_Fpredict(Uv, visc, numx, numy, delx, dely ):
    """
    xy shear stress for F flux predictor step
    
    y-backward difference, x-central difference
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    """
    #Don't need shear values at inflow, outflow and plate
    #
    tau_xy = numpy.zeros((numx,numy))
    #
    #at upper boundary, excluding outflow point and inflow point:
    # i back j back
    tau_xy[1:-1,-1] = visc[1:-1,-1]*\
                      ((Uv[1,1:-1,-1]/Uv[0,1:-1,-1] -\
                        Uv[1,1:-1,-2]/Uv[0,1:-1,-2])/dely +\
                      (Uv[2,1:-1,-1]/Uv[0,1:-1,-1] -\
                       Uv[2,:-2,-1]/Uv[0,:-2,-1])/delx)

    #inner points:
    #i central j backward
    tau_xy[1:-1,1:-1] = visc[1:-1,1:-1]*\
                       ( (1/(dely))*((Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])-\
                                       (Uv[1,1:-1,:-2]/Uv[0,1:-1,:-2])) +\
                       (1/(2*delx))*((Uv[2,2:,1:-1]/Uv[0,2:,1:-1])-\
                                 (Uv[2,:-2,1:-1]/Uv[0,:-2,1:-1])) )
   
    return tau_xy
 #-----------------------------------------------------------------------------------------     
def get_tau_xx_Epredict(Uv, visc, numx, numy, delx, dely, lmbda):
    """
    xx normal stress for E flux predictor step
    x-backward, y-central
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    lmbda - second viscosity
    """
    #don't need values for upper boundary, plate, and inflow
    #
    tau_xx = numpy.zeros((numx,numy))
    #
    #inner points
    # i backward, j central
    tau_xx[1:-1,1:-1] = lmbda*\
    					((Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1] - Uv[1,:-2,1:-1]/Uv[0,:-2,1:-1])/delx +\
                         (Uv[2,1:-1,2:]/Uv[0,1:-1,2:] - Uv[2,1:-1,:-2]/Uv[0,1:-1,:-2])/(2*dely))+\
                       (2*visc[1:-1,1:-1]*(1/delx)*\
                        ((Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])-(Uv[1,:-2,1:-1]/Uv[0,:-2,1:-1])))
    #
    #upper outflow
    # i and j back
    tau_xx[-1,-1] = lmbda*\
    					((Uv[1,-1,-1]/Uv[0,-1,-1] - Uv[1,-2,-1]/Uv[0,-2,-1])/delx +\
                         (Uv[2,-1,-1]/Uv[0,-1,-1] - Uv[2,-1,-2]/Uv[0,-1,-2])/(dely))+\
                       (2*visc[-1,-1]*(1/delx)*\
                        ((Uv[1,-1,-1]/Uv[0,-1,-1])-(Uv[1,-2,-1]/Uv[0,-2,-1])))
    #
    #outflow:
    #i back j central
    tau_xx[-1,1:-1] = lmbda*\
    					((Uv[1,-1,1:-1]/Uv[0,-1,1:-1] - Uv[1,-2,1:-1]/Uv[0,-2,1:-1])/delx +\
                         (Uv[2,-1,2:]/Uv[0,-1,2:] - Uv[2,-1,:-2]/Uv[0,-1,:-2])/(2*dely))+\
                       (2*visc[-1,1:-1]*(1/delx)*\
                        ((Uv[1,-1,1:-1]/Uv[0,-1,1:-1])-(Uv[1,-2,1:-1]/Uv[0,-2,1:-1])))
    #
    return tau_xx
#-----------------------------------------------------------------------------------------
def get_tau_yy_Fpredict(Uv, visc, numx, numy, delx, dely, lmbda):
    """
    yy normal stress for F flux predictor step
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    lmbda - second viscosity
    """
    #Don't need values for inflow, outflow, plate
    #
    tau_yy = numpy.zeros((numx,numy))
    #
    #upper boundary, excluding outflow and inflow points:
    #i and j backward
    tau_yy[1:-1,-1] = lmbda*\
    					((Uv[1,1:-1,-1]/Uv[0,1:-1,-1] - Uv[1,:-2,-1]/Uv[0,:-2,-1])/delx +\
                         (Uv[2,1:-1,-1]/Uv[0,1:-1,-1] - Uv[2,1:-1,-2]/Uv[0,1:-1,-2])/(dely))+\
                       (2*visc[1:-1,-1]*(1/dely)*\
                        ((Uv[2,1:-1,-1]/Uv[0,1:-1,-1])-(Uv[2,1:-1,-2]/Uv[0,1:-1,-2])))
    #
    #inner points
    # i central, j backward
    tau_yy[1:-1,1:-1] = lmbda*\
    					((Uv[1,2:,1:-1]/Uv[0,2:,1:-1] - Uv[1,:-2,1:-1]/Uv[0,:-2,1:-1])/(2*delx) +\
                         (Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1] - Uv[2,1:-1,:-2]/Uv[0,1:-1,:-2])/(dely))+\
                       (2*visc[1:-1,1:-1]*(1/dely)*\
                        ((Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])-(Uv[2,1:-1,:-2]/Uv[0,1:-1,:-2])))
    #
    return tau_yy
#-----------------------------------------------------------------------------------------
def get_E_flux_predictor(Uv, numx, numy, delx, visc, Temp, kc, txx, txy, R):
    """
    E flux vector for predictor step
    
    The values needed here are E[i=1:, j=1:-1]
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    Temp - temperature in domain,
    kc - thermal conductivity
    txx, txy - stresses
    R - gas constant
    """
    Ev= numpy.zeros_like(Uv)
    qx = numpy.zeros((numx,numy))
    #
    # x heat flux, i backward
    qx[1:,1:-1] = -kc[1:,1:-1]*(Temp[1:,1:-1]-Temp[:-1,1:-1])/delx 
    #
    Ev[0,:,:] = Uv[1,:,:]
    Ev[1,:,:] = (Uv[1,:,:]**2)/Uv[0,:,:] + Uv[0,:,:]*R*Temp[:,:] - txx[:]
    Ev[2,:,:] = Uv[1,:,:]*Uv[2,:,:]/Uv[0,:,:] - txy[:]
    Ev[3,:,:] = (Uv[1,:,:]/Uv[0,:,:])*(Uv[3,:,:] + Uv[0,:,:]*R*Temp[:,:]) -\
                (Uv[1,:,:]/Uv[0,:,:])*txx[:] - (Uv[2,:,:]/Uv[0,:,:])*txy[:] + qx[:,:]
    return Ev
#-----------------------------------------------------------------------------------------
def get_F_flux_predictor(Uv, numx, numy, dely, visc, Temp, kc, txy, tyy, R):
    """
    F flux vector for predictor step
    
    The values needed here are F[i=1:-1, j=1:]
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    Temp - temperature in domain,
    kc - thermal conductivity
    txx, txy - stresses
    R - gas constant
    """
    Fv= numpy.zeros_like(Uv)
    qy = numpy.zeros((numx,numy))
    #
    # y heat flux, j back
    qy[1:-1,1:] = -kc[1:-1,1:]*(Temp[1:-1,1:]-Temp[1:-1,:-1])/dely
    #
    Fv[0,:,:] = Uv[2,:,:]
    Fv[1,:,:] = Uv[1,:,:]*Uv[2,:,:]/Uv[0,:,:] - txy[:]
    Fv[2,:,:] = (Uv[2,:,:]**2)/Uv[0,:,:] + Uv[0,:,:]*R*Temp[:,:] - txy[:]
    Fv[3,:,:] = (Uv[2,:,:]/Uv[0,:,:])*(Uv[3,:,:] + Uv[0,:,:]*R*Temp[:,:]) -\
                (Uv[1,:,:]/Uv[0,:,:])*txy[:] - (Uv[2,:,:]/Uv[0,:,:])*tyy[:] + qy[:,:]
    return Fv
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def get_dt(Uv, numx, numy, delx, dely, visc, Temp, gamma, R, Prt):
    """
    Adaptive time step function
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    Temp - temperature in domain,
    gamma - ratio of specific heats
    R - gas constant
    Prt - Prandtl number
    """
    K= 0.6 #fudge factor for time step to ensure stability
    #
    dt_cfl = numpy.zeros((numx-2,numy-2))
    #
    dt_cfl[:,:] = ((numpy.abs(Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1]))/delx +\
                   (numpy.abs(Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1]))/dely +\
                   numpy.sqrt(gamma*R*Temp[1:-1,1:-1])*numpy.sqrt( 1/(delx**2) +\
                                                                  1/(dely**2) ) +\
                   2*numpy.max( ((4/3)*visc[1:-1,1:-1]*\
                                 (gamma*visc[1:-1,1:-1]/Prt))/Uv[0,1:-1,1:-1] )*\
                    ( 1/(delx**2) + 1/(dely**2) ) )**(-1)
    #
    delt = numpy.min(K*dt_cfl[:,:])
    #
    return delt
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def get_tau_xy_Ecorrect(Uv, visc,numx, numy, delx, dely ):
    """
    xy shear stress for E flux corrector step
    x-forward difference, y-central difference
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    """
    #Don't need values for outflow, upper boundary, plate
    #
    tau_xy = numpy.zeros((numx,numy))
    #
    #inner points:
    # i forward, j central
    tau_xy[1:-1,1:-1] = visc[1:-1,1:-1]*\
                        ((1/(2*dely))*((Uv[1,1:-1,2:]/Uv[0,1:-1,2:])-\
                                       (Uv[1,1:-1,:-2]/Uv[0,1:-1,:-2])) +\
                        (1/delx)*((Uv[2,2:,1:-1]/Uv[0,2:,1:-1])-\
                                  (Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])) )
    #
    #at inflow, including leading edge, excluding upper boundary:
    # i forward, j forward
    tau_xy[0,:-1] = visc[0,:-1]*((Uv[1,0,1:]/Uv[0,0,1:] -\
                                  Uv[1,0,:-1]/Uv[0,0,:-1])/dely +\
                                   (Uv[2,1,:-1]/Uv[0,1,:-1] -\
                                    Uv[2,0,:-1]/Uv[0,0,:-1])/delx)
    #upper inflow:
    # i forward , j backward
    tau_xy[0,-1] = visc[0,-1]*((Uv[1,0,-1]/Uv[0,0,-1] - Uv[1,0,-2]/Uv[0,0,-2])/dely +\
                               (Uv[2,1,-1]/Uv[0,1,-1] - Uv[2,0,-1]/Uv[0,0,-1])/delx)
	#
    return tau_xy
#-----------------------------------------------------------------------------------------
def get_tau_xy_Fcorrect(Uv, visc,numx, numy, delx, dely ):
    """
    xy shear stress for F flux corrector step
    y-forward difference, x-central difference
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    """
    #don't need values for inflow, outflow, and upper boundary
    #
    tau_xy = numpy.zeros((numx,numy))
    #
    #at plate, excluding outflow point and inflow point:
    # i central, j forward
    tau_xy[1:-1,0] = visc[1:-1,0]*((Uv[1,1:-1,1]/Uv[0,1:-1,1] -\
                                    Uv[1,1:-1,0]/Uv[0,1:-1,0])/dely +\
                                   (Uv[2,2:,0]/Uv[0,2:,0] -\
                                    Uv[2,:-2,0]/Uv[0,:-2,0])/(2*delx))
    #
    #inner points:
    # i central, j forward
    tau_xy[1:-1,1:-1] = visc[1:-1,1:-1]*\
                        ((1/dely)*((Uv[1,1:-1,2:]/Uv[0,1:-1,2:])-\
                                       (Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])) +\
                        (1/(2*delx) )*((Uv[2,2:,1:-1]/Uv[0,2:,1:-1])-\
                                  (Uv[2,:-2,1:-1]/Uv[0,:-2,1:-1])) )
    #
    return tau_xy
#-----------------------------------------------------------------------------------------
def get_tau_xx_Ecorrect(Uv, visc, numx, numy, delx, dely, lmbda):
    """
    xx normal stress for E flux corrector step
    x-forward, y-central
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    lmbda - second viscosity
    """
 	#don't need values for outflow, upper boundary, and plate
 	#
    tau_xx = numpy.zeros((numx,numy))
    #
    #inner points
    # i forward, j central
    tau_xx[1:-1,1:-1] = lmbda*\
    					((Uv[1,2:,1:-1]/Uv[0,2:,1:-1] - Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])/delx +\
                         (Uv[2,1:-1,2:]/Uv[0,1:-1,2:] - Uv[2,1:-1,:-2]/Uv[0,1:-1,:-2])/(2*dely))+\
                       (2*visc[1:-1,1:-1]*(1/delx)*\
                        ((Uv[1,2:,1:-1]/Uv[0,2:,1:-1])-(Uv[1,1:-1,1:-1]/Uv[0,1:-1,1:-1])))
    #
    #at inflow, including leading edge, excluding upper boundary:
    # i forward, j forward
    tau_xx[0,:-1] = lmbda*\
    					((Uv[1,1,:-1]/Uv[0,1,:-1] - Uv[1,0,:-1]/Uv[0,0,:-1])/delx +\
                         (Uv[2,0,1:]/Uv[0,0,1:] - Uv[2,0,:-1]/Uv[0,0,:-1])/(dely))+\
                       (2*visc[0,:-1]*(1/delx)*\
                        ((Uv[1,1,:-1]/Uv[0,1,:-1])-(Uv[1,0,:-1]/Uv[0,0,:-1])))
    #upper inflow
    # i forward, j backward
    tau_xx[0,-1] = lmbda*\
    					((Uv[1,1,-1]/Uv[0,1,-1] - Uv[1,0,-1]/Uv[0,0,-1])/delx +\
                         (Uv[2,0,-1]/Uv[0,0,-1] - Uv[2,0,-2]/Uv[0,0,-2])/(dely))+\
                       (2*visc[0,-1]*(1/delx)*\
                        ((Uv[1,1,-1]/Uv[0,1,-1])-(Uv[1,0,-1]/Uv[0,0,-1])))
   #
    return tau_xx
#-----------------------------------------------------------------------------------------
def get_tau_yy_Fcorrect(Uv, visc, numx, numy, delx, dely, lmbda):
    """
    yy normal stress for F flux predictor step
    y-forward, x-central
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    lmbda - second viscosity
    """
	#Don't need values for inflow, outflow, and upper boundary
    #
    tau_yy = numpy.zeros((numx,numy))
    #
    #at plate, exclude leading edge and ouflow point
    #i central, j forward
    tau_yy[1:-1,0] = lmbda*\
    					((Uv[1,2:,0]/Uv[0,2:,0] - Uv[1,:-2,0]/Uv[0,:-2,0])/(2*delx) +\
                         (Uv[2,1:-1,1]/Uv[0,1:-1,1] - Uv[2,1:-1,0]/Uv[0,1:-1,0])/(dely))+\
                       (2*visc[1:-1,0]*(1/dely)*\
                        ((Uv[2,1:-1,1]/Uv[0,1:-1,1])-(Uv[2,1:-1,0]/Uv[0,1:-1,0])))
    #
    #inner points
    #i central, j forward
    tau_yy[1:-1,1:-1] = lmbda*\
    					((Uv[1,2:,1:-1]/Uv[0,2:,1:-1] - Uv[1,:-2,1:-1]/Uv[0,:-2,1:-1])/(2*delx) +\
                         (Uv[2,1:-1,2:]/Uv[0,1:-1,2:] - Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])/(dely))+\
                       (2*visc[1:-1,1:-1]*(1/dely)*\
                        ((Uv[2,1:-1,2:]/Uv[0,1:-1,2:])-(Uv[2,1:-1,1:-1]/Uv[0,1:-1,1:-1])))
    return tau_yy
#-----------------------------------------------------------------------------------------
def get_E_flux_correct(Uv, numx, numy, delx, visc, Temp, kc, txx, txy, R):
    """
    E flux vector for corrector step
    
    The values needed here are E[i=:-1, j=1:-1]
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    Temp - temperature in domain,
    kc - thermal conductivity
    txx, txy - stresses
    R - gas constant
    """
    Ev= numpy.zeros_like(Uv)
    qx = numpy.zeros((numx,numy))
    qx[:-1,1:-1] = -kc[:-1,1:-1]*(Temp[1:,1:-1]-Temp[:-1,1:-1])/delx
    #
    Ev[0,:,:] = Uv[1,:,:]
    Ev[1,:,:] = (Uv[1,:,:]**2)/Uv[0,:,:] + Uv[0,:,:]*R*Temp[:,:] - txx[:]
    Ev[2,:,:] = Uv[1,:,:]*Uv[2,:,:]/Uv[0,:,:] - txy[:]
    Ev[3,:,:] = (Uv[1,:,:]/Uv[0,:,:])*(Uv[3,:,:] + Uv[0,:,:]*R*Temp[:,:]) -\
                (Uv[1,:,:]/Uv[0,:,:])*txx[:] - (Uv[2,:,:]/Uv[0,:,:])*txy[:] + qx[:,:]
    return Ev
#-----------------------------------------------------------------------------------------
def get_F_flux_correct(Uv, numx, numy, dely, visc, Temp, kc, txy, tyy, R):
    """
    F flux vector for corrector step
    
    The values needed here are F[i=1:-1, j=:-1]
    
    Parameters
    ----------------
    Uv - Vector of density, momentum, and energy values
    visc - viscosity in domain
    numx - x grid points
    numy - y grid points
    delx - x direction spacing
    dely - y direction spacing
    Temp - temperature in domain,
    kc - thermal conductivity
    txx, txy - stresses
    R - gas constant
    """
    Fv= numpy.zeros_like(Uv)
    qy = numpy.zeros((numx,numy))
    qy[1:-1,:-1] = -kc[1:-1,:-1]*(Temp[1:-1,1:]-Temp[1:-1,:-1])/dely
    #
    Fv[0,:,:] = Uv[2,:,:]
    Fv[1,:,:] = Uv[1,:,:]*Uv[2,:,:]/Uv[0,:,:] - txy[:]
    Fv[2,:,:] = (Uv[2,:,:]**2)/Uv[0,:,:] + Uv[0,:,:]*R*Temp[:,:] - txy[:]
    Fv[3,:,:] = (Uv[2,:,:]/Uv[0,:,:])*(Uv[3,:,:] + Uv[0,:,:]*R*Temp[:,:]) -\
                (Uv[1,:,:]/Uv[0,:,:])*txy[:] - (Uv[2,:,:]/Uv[0,:,:])*tyy[:] + qy[:,:]
    return Fv
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def maccormack(U_init,numt,numx,numy,delx,dely,Tw,Tfs,rho_fs,ufs,c_v,c_p,viscfs,Prt,lmbda,R,gamma):
    """
    - The MacCormack scheme is used to advance the solution in time.
    - The scheme is second order accurate in space and time.
    - This function uses many inputs, all of which have already been described in all of
       the previous functions.
    - The program ends once all density values in the flow field change no more than
      1e-8 with respect to the previous time step
    - Conservation of mass is performed by analyzing mass flow rate across inflow and outflow
    Parameters:
    -----------
    """
    Un = numpy.zeros((numt+1,4,numx,numy))
    Un[0,:,:,:] = U_init.copy()
    #
    U = U_init.copy()
    #
    Us = U_init.copy()
    #
    for t in range(1,numt+1):
    	#get properties to calculate fluxes:
    	T = get_Temperature(U, numx, numy, Tw, Tfs, c_v)
    	mu = get_visc(T, viscfs, Tfs)
    	k = get_k(mu, c_p, Prt)
    	#get shear:
    	t_xyE = get_tau_xy_Epredict(U, mu, numx, numy, delx, dely )
    	t_xyF = get_tau_xy_Fpredict(U, mu, numx, numy, delx, dely )
    	t_xx = get_tau_xx_Epredict(U, mu, numx, numy, delx, dely, lmbda)
    	t_yy = get_tau_yy_Fpredict(U, mu, numx, numy, delx, dely, lmbda)
    	#calculate fluxes E, F:
    	E = get_E_flux_predictor(U, numx, numy, delx, mu, T, k, t_xx, t_xyE, R)
    	F = get_F_flux_predictor(U, numx, numy, dely, mu, T, k, t_xyF, t_yy, R)
    	#dt:
    	dt = get_dt(U, numx, numy, delx, dely, mu, T, gamma, R, Prt)
    	#Predictor Step:
    	Us[:,1:-1,1:-1] = U[:,1:-1,1:-1] -\
    							(dt/delx)*(E[:,2:,1:-1] - E[:,1:-1,1:-1]) -\
    							(dt/dely)*(F[:,1:-1,2:] - F[:,1:-1,1:-1])
    	Ustar = get_BC(Us, T, numy, rho_fs, Tw, ufs, c_v, Tfs, R)
    	#update properties:
    	T2 = get_Temperature(Ustar, numx, numy, Tw, Tfs, c_v)
    	mu2 = get_visc(T2, viscfs, Tfs)
    	k2 = get_k(mu2, c_p, Prt)
    	#update shear:
    	t_xyE2 = get_tau_xy_Ecorrect(Ustar,mu2,numx, numy, delx, dely)
    	t_xyF2 = get_tau_xy_Fcorrect(Ustar,mu2,numx, numy, delx, dely)
    	t_xx2 = get_tau_xx_Ecorrect(Ustar, mu2, numx, numy, delx, dely, lmbda)
    	t_yy2 = get_tau_yy_Fcorrect(Ustar, mu2, numx, numy, delx, dely, lmbda)
    	#update fluxes:
    	E2 = get_E_flux_correct(Ustar, numx, numy, delx, mu2, T2, k2, t_xx2, t_xyE2, R)
    	F2 = get_F_flux_correct(Ustar, numx, numy, dely, mu2, T2, k2, t_xyF2, t_yy2, R)
    	#corrector step:
    	Un[t,:,1:-1,1:-1] = 0.5*( U[:,1:-1,1:-1] + Ustar[:,1:-1,1:-1] -\
    							(dt/delx)*(E2[:,1:-1,1:-1]-E2[:,:-2,1:-1]) -\
    							(dt/dely)*(F2[:,1:-1,1:-1]-F2[:,1:-1,:-2] ))
    	#
    	Un[t,:,:,:] = get_BC(Un[t,:,:,:], T2, numy, rho_fs, Tw, ufs, c_v, Tfs, R)
    	U = Un[t,:,:,:].copy()
    	#print(t)
    	if( numpy.all(numpy.abs(Un[t,0,:,:]-Un[t-1,0,:,:]) < 1e-8) == True ):
    		tt=t+1
    		Un = Un[:tt,:,:,:].copy()
    		mscn = (numpy.trapz(Un[t,1,0,:])/numpy.trapz(Un[t,1,-1,:]))*100
    		print('Mass is conserved by %.2f percent' % mscn)
    		break
         
    return Un