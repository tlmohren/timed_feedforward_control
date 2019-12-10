import string
from itertools import cycle
from six.moves import zip 
import numpy as np 
import dynamicpendulums.phasediagram as phd
import dynamicpendulums.singlependulum as sp      
import matplotlib.pyplot as plt



def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (0.9,0.9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)
    return( ax) 


def get_ax_size(ax,fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def label_axes2(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (0.9,0.9)
    for ax, lab in zip(fig.axes, labels):
        w, h = get_ax_size(ax,fig)
         
        loc2 =  ( loc[0]/(w/100) , loc[1] )
     
        ax.annotate(lab, xy=loc2,
                    xycoords='axes fraction',
                    **kwargs)
    return( ax) 


def plot_discontinuous_lines( ax_p, y_data, color='k', linewidth = 2): 
    plot_array = []
    bool_jump = (np.abs(np.diff(y_data[:,1]) ) > 1 ) | (np.abs(np.diff(y_data[:,0]) ) > 1 )
    
    split_locs = np.where( bool_jump )[0]
    split_locs = np.insert(split_locs,0,0)
    split_locs = np.insert(split_locs,len(split_locs),len(y_data) )  + 1
    for j in range(len(split_locs)-1 ):
        plot_array.append( y_data[split_locs[j]:split_locs[j+1]  ,:] ) 

    for j in range(len(plot_array)):
        ax_p.plot( plot_array[j][:,0], plot_array[j][:,1], color=color , linewidth=linewidth )
    
    return( ax_p )  

def plot_add_contours( ax_p) :  
    y_lim = 10
    g= -10  
    xc =   np.linspace( -2*np.pi, 2*np.pi,1001)*1.4
    yc = np.linspace(- y_lim,y_lim,1001)
    Xc, Yc = np.meshgrid(xc, yc) 
    Zc = - ( 1-np.cos(Xc) ) *g + 0.5* Yc**2       
    contour_col = np.array([[1,1,1,1]])*0.9 
    
    lin = np.array([5,10,25,40,60 ]) 
    CS = ax_p.contour(Xc, Yc, Zc ,colors = contour_col  , levels = lin, zorder = 1,linewidths = 0.5)    
    
    return( ax_p) 

def plot_state_space(ax_p, x_lim = (-7.5,7.5), y_lim= (-2*np.pi,2*np.pi) ,
				contour_col=np.array([[1,1,1,1]])*0.7,
				obscure_col=np.ones((3))*0.7,
				obscure_edge=np.ones((3))*0.7,
				obscure_alpha=0.5,
				homoclinic_col=np.ones((3))*0.7):

    #-initialize axis limits 
    ax_p.set_xlim( x_lim ) 
    ax_p.set_ylim( y_lim )    

    # plot energy contours 
    y_lim = 10
    g= -10  
    xc =   np.linspace( -np.pi,np.pi,1001)*2
    yc = np.linspace(- y_lim,y_lim,1001)
    Xc, Yc = np.meshgrid(xc, yc) 
    Zc = - ( 1-np.cos(Xc) ) *g + 0.5* Yc**2      

    lin = np.array([5,10,25,40,60 ]) 
    CS = ax_p.contour(Xc, Yc, Zc ,colors = contour_col  , levels = lin, zorder = 1)    

    # homoclinic plot 
    phd.homoclinic_plot(ax_plot=ax_p,x_low = -2*np.pi, x_high = 2*np.pi, col = homoclinic_col)
  
    # plot obscure rectangles  
    L = 16
    rectangle1 = plt.Rectangle( ( -np.pi+0.3,-L/2), 2*np.pi-0.6,L, 
                                zorder=5,facecolor=obscure_col, edgecolor=obscure_edge ,alpha = obscure_alpha) 
    ax_p.add_patch(rectangle1)

    rectangle1 = plt.Rectangle( ( -3*np.pi+0.3,-L/2), 2*np.pi-0.6,L, 
                                zorder=5,facecolor=obscure_col, edgecolor=obscure_edge ,alpha = obscure_alpha) 
    ax_p.add_patch(rectangle1)

    rectangle1 = plt.Rectangle( ( np.pi+0.3,-L/2), 2*np.pi-0.6,L, 
                                zorder=5,facecolor=obscure_col, edgecolor=obscure_edge ,alpha = obscure_alpha) 
    ax_p.add_patch(rectangle1)

    # axis makeup 
    ax_p.xaxis.set_major_locator(plt.MultipleLocator(np.pi )) 
    ax_p.xaxis.set_major_formatter(plt.FuncFormatter(phd.multiple_formatter()))  
    ax_p.yaxis.set_major_locator( plt.MultipleLocator(6) ) 