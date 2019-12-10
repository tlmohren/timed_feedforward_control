import numpy as np  
from operator import sub
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def pendulum_ode(y_n,t, u_n= 0  ,time_sign = 1, b =0.0   ): 
    g = -10. 
    L = 1. 
    mp = 1.
    fy = np.zeros(y_n.shape)   
    fy[0]= time_sign*  y_n[1]   
    fy[1] = time_sign* (  g/L*np.sin( y_n[0] ) -b/(mp*L**2) *y_n[1] + 1/(mp*L**2)*u_n )
    return fy
 
def wrap2periodic(theta, period = 2*np.pi, center = 0 ):
    offset = np.pi - center
    thetaWrapped = (theta + offset ) %period - offset 
    return thetaWrapped 


def in_polygon( polygon_array, grid_array ):
    # note that the order of points is really important. The sequence of points will form the outer hull of the polygon
    
    # create list of points in polygon 
    boundary_points = []
    for j in range( polygon_array.shape[0] ):  
        boundary_points.append(  polygon_array[j,:]  )
        
    #  create polygon with points 
    boundary_polygon = Polygon(boundary_points) 
    
    # create list of points to check 
    points = [] 
    for k in range( grid_array.shape[0] ):
        points.append( Point( grid_array[k,:] )) 

    # create empty boolean list 
    point_true = np.full( grid_array.shape[0], np.nan, dtype=bool)

    # iterate over list to see if they are in the polygon 
    for k in range( grid_array.shape[0]  ): 
        point_true[k] =   boundary_polygon.contains( points[k] )   
    return point_true, boundary_polygon  
  
def create_polygon_array( t1_data, t2_data ):
    t1_data_sort = t1_data[ (-1e3-t1_data[:,4]).argsort() ]
    t2_data_sort = t2_data[ t2_data[:,4].argsort() ]  
    data_concat = np.concatenate( (t1_data_sort,t2_data_sort)) 
    return data_concat[:,:2]  
  
def compute_dE( x_array, y_array): 
    g = -10.0
    dE = ( 0.5*y_array**2 +    np.cos( x_array )*g)    - 10 
    return dE 

def calc_costToGo(x,u_j,Q=1,R=1, g = -10, E_d = 10): 
    E = 0.5*x[1]**2 + np.cos(x[0])*g 
    E_er = E-E_d 
    J_to_go = E_er*Q*E_er + u_j*R*u_j  # quadratic cost  
    return J_to_go   

def calc_costToGo_V2(x0,x1,u_j,Q=1,R=1, g = -10, E_d = 10): 
    E0 = 0.5*x0[1]**2 + np.cos(x0[0])*g 
    E1 = 0.5*x1[1]**2 + np.cos(x1[0])*g 

    E_er = (E0+E1)/2-E_d 
    J_to_go = E_er*Q*E_er + u_j*R*u_j  # quadratic cost  
    return J_to_go  

def matrix_fillnan(Mat):
    isnn = np.isnan(Mat)  
    z_max =  Mat[~isnn].max()  
    z_max_log = np.log(z_max)  
    
    Mat_clean = Mat.copy()
    Mat_clean[isnn] = z_max
    
    Mat_log = np.log(Mat_clean + 1 - Mat_clean.min())   
    return Mat_clean, Mat_log 