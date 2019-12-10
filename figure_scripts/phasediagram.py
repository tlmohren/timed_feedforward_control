import numpy as np
import matplotlib.pyplot as plt
from operator import sub
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
 
#-------------------------------------------------------------------
# plot arrows 
#-------------------------------------------------------------------
def get_aspect(ax_A):
    # Total figure size
    figW, figH = ax_A.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax_A.get_position().bounds
    
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units 
    data_ratio = sub(*ax_A.get_ylim()) / sub(*ax_A.get_xlim())

    return disp_ratio / data_ratio
    
def vec_rotation(vector,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]] )
    return np.dot(R,vector)
  
def draw_arrow( x,y,angle, 
                      ax_pl , 
                      obj_H=1.,obj_A=1., facecolor = 'black', edgecolor='black', 
                      arrow_type = 1 ,angle_local_aspect = True):
    
    x_loc =  np.array([x,y]) 
    
    # acquire information on aspect  
    ax_aspect = get_aspect(ax_pl)   
    
    # determine which angle, and what the rotation point is 
    if arrow_type == 1: 
        X = np.array ([ [0,-0.5  ],  [0,  0.5 ],  [1,0]]) 
        X_c = np.array([0,0]) 
    elif arrow_type == 2: 
        X = np.array ([ [ 0.,0. ],  [-0.5, 0.5 ],  [1.,  0], [-0.5,-0.5 ]])
        X_c = np.array([0,0]) 
    elif arrow_type == 3: 
        X = np.array ([ [-.5,-.5 ],  [-.5, .5],  [.5,  .5], [.5,-.5 ]])
        X_c = np.array([0,0]) 
    
    # scale arrow to specifications 
    H = obj_H   
    W = H*  ax_aspect/obj_A 
    X_dim = X.copy() 
    X_dim[:,1] = X_dim[:,1]  * H
    X_dim[:,0] = X_dim[:,0]  * W
 
    # adjust angle to aspect 
    dx = np.cos(angle)
    dy = np.sin(angle)
    if angle_local_aspect == True:
        rot_ang = np.arctan2( dy*ax_aspect  ,dx  )  
    else: 
        rot_ang = np.arctan2( dy  ,dx  )  

    # adjust aspect 
    X_centered2 = X_dim.copy()  
    X_centered2[:,1] = X_centered2[:,1]*ax_aspect

    # rotate all individual points 
    X_rotated = np.full( np.shape(X),np.nan)
    for j in range( np.shape(X)[0] ):
        X_rotated[j,:] = vec_rotation(X_centered2[j,:],rot_ang)  

    # scale rotated arrow 
    X_rotated2 = X_rotated.copy()
    X_rotated2[:,1] = X_rotated2[:,1]/ax_aspect   
    
    # place scaled and rotated arrow 
    X_transformed = X_rotated2+x_loc
    ax_pl.add_patch( plt.Polygon( X_transformed  ,facecolor=facecolor ,  edgecolor=edgecolor  ) )   

 
#-------------------------------------------------------------------
# homoclinic orbit plot 
#-------------------------------------------------------------------
def homoclinic_plot(ax_plot, x_low=-np.pi , x_high=np.pi,col='k' ):
    xh = np.linspace( x_low,x_high ,101 )
    yh = np.sqrt( 20*(1+np.cos(xh))  ) 
    ax_plot.plot(xh,yh,'y' ,color = col)
    ax_plot.plot(xh,-yh,'y',color = col) 
    # ax_plot.scatter( 0,0,color= 'magenta')  
    return ax_plot  

#-------------------------------------------------------------------
# axis tick formatting with pi 
#-------------------------------------------------------------------
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex)) 