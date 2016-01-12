#Import these libraries in one cell:

from matplotlib import animation
from JSAnimation import IPython_display
from JSAnimation.IPython_display import display_animation
from moviepy.editor import *

import pylab 
import types

##############################################################

#Use these animation functions:

def animate(data):
    im = ax.contourf(my,mx,data,cmap='jet')
    return im

fig = pyplot.figure(figsize=(8,5))
ax = pyplot.axes()
im = ax.contourf(my, mx, US[0,2,:,:], cmap='jet')

anim = animation.FuncAnimation(fig, animate, frames=US[:1500:10,2,:,:], interval=100)
#to display:
# display_animation(anim, default_mode='once')
#to save mp4 file
# anim.save('ymom.mp4', writer='ffmpeg')


##############################################################

#Make a gif!

clip = VideoFileClip("ymom.mp4").subclip(0,10)
clip.write_gif("ymom.gif", fps=10)

#Have Fun!