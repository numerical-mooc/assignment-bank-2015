import numpy
from matplotlib import pyplot 


def get_grid(Lg, Yg, numx, numy):
	#
	xg = numpy.linspace(0,Lg,numx)
	yg = numpy.linspace(0,Yg,numy)
	mxg, myg = numpy.meshgrid(xg,yg)
	#
	return xg,yg, mxg, myg
#----------------------------------------------------------------------
def get_Pressure(Uv, Temp, numx, numy,R):
	#
    P = numpy.zeros((numx,numy))
    P[:,:] = Uv[0,:,:]*R*Temp[:,:]
    #
    return P
#----------------------------------------------------------------------
def get_Pressure_plot(xg,yg, Ren, Pg, P_inf):
	""" Plots the non-dimensional surface pressure at converged solution,
	and the non-dimensional trailing edge pressure at converged solution
	"""
	fig = pyplot.figure(figsize=(8,5))
	pyplot.subplots_adjust(hspace=0.5)
	#
	pyplot.subplot(2,1,1)
	pyplot.plot(xg, Pg[:,0]/P_inf )
	pyplot.ylabel('P\Pinf ')
	pyplot.xlabel('Plate (m)') 
	pyplot.title('Figure 1: Non-Dimensional Surface Pressure');
	
	pyplot.subplot(2,1,2)
	pyplot.plot(Pg[-1,:]/P_inf, yg[:]*numpy.sqrt(Ren)/xg[-1] )
	pyplot.ylabel('Non-Dimensional Y distance')
	pyplot.xlabel('P\Pinf ') 
	pyplot.title('Figure 2: Non-Dimensional Trailing Edge Pressure');
	
	return fig
#----------------------------------------------------------------------
def get_Temperature_plot(xg,yg,Ren,Tg,T_inf):
	"""Plots Non-dimensionalized temperature at trailing edge
	"""
	fig = pyplot.figure(figsize=(8,5))
	pyplot.plot(Tg[-1,:]/T_inf, yg[:]*numpy.sqrt(Ren)/xg[-1])
	pyplot.xlabel('T\Tinf')
	pyplot.ylabel('Non-Dimensional Y distance')
	pyplot.title('Figure 3: Non-Dimensional Temperature at Trailing Edge');
	
	return fig
#----------------------------------------------------------------------
#----------------------------------------------------------------------
def get_Velocity_plot(xg,yg,Ren, Uf, u_inf):
	"""Plots Non-dimensionalized velocity at trailing edge
	"""
	ut = numpy.zeros_like(yg)
	ut = (Uf[1,-1,:]/Uf[0,-1,:])
	fig = pyplot.figure(figsize=(8,5))
	pyplot.plot(ut[:]/u_inf, yg[:]*numpy.sqrt(Ren)/xg[-1])
	pyplot.xlabel('u/u_inf')
	pyplot.ylabel('Non-Dimensional Y distance')
	pyplot.title('Figure 4: Non-Dimensional Velocity at Trailing Edge');
	
	return fig
#----------------------------------------------------------------------
def get_cplots(mxg,myg,Uv, Tv, R):
    
    fig = pyplot.figure(figsize=(10,10))

    pyplot.subplot(2,2,1)
    pyplot.contourf(myg,mxg,Uv[1,:,:])
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.xlabel('$Plate (m)$')
    pyplot.ylabel('$Height (m)$')
    pyplot.colorbar()
    pyplot.title('x-momentum (kg / (m^2 s) )')

    pyplot.subplot(2,2,2)
    pyplot.contourf(myg,mxg,Uv[2,:,:])
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.xlabel('$Plate (m)$')
    pyplot.ylabel('$Height (m)$')
    pyplot.colorbar()
    pyplot.title('y-momentum (kg / (m^2 s) )')

    pyplot.subplot(2,2,3)
    pyplot.contourf(myg,mxg,Uv[3,:,:])
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.xlabel('$Plate (m)$')
    pyplot.ylabel('$Height (m)$')
    pyplot.colorbar()
    pyplot.title('Total Energy (Joules)')

    pyplot.subplot(2,2,4)
    pyplot.contourf(myg,mxg,Uv[0,:,:]*R*Tv[:,:])
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.xlabel('$Plate (m)$')
    pyplot.ylabel('$Height (m)$')
    pyplot.colorbar()
    pyplot.title('Total Pressure (Pa)');
    
    return fig
#----------------------------------------------------------------------