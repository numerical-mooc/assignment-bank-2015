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
def get_Pressure_plot(xg, Pg, P_inf):
    #
    fig = pyplot.figure(figsize=(8,5))
    pyplot.plot(xg, Pg[:,0]/P_inf )
    pyplot.ylabel('P\Pinf ')
    pyplot.xlabel('Plate (m)') 
    pyplot.title('Non-Dimensional Surface Pressure at Converged Solution');
    
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