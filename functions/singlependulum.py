import numpy as np  
from operator import sub
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.integrate import odeint  
import time    

import scipy.interpolate as interp  

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


def compute_U(J_use,x_space,y_space, X,Y, u_opts, tU,Q,R ) :
    U = np.full( np.shape(J_use), np.nan  )
    Eff = np.full( np.shape(J_use), np.nan  )
    t_start = time.time()  

    # spline = interp.interp2d( x_space,y_space,J_use.transpose().ravel() ,kind = 'cubic' )   
    spline = interp.interp2d( x_space,y_space,J_use.transpose().ravel() ,kind = 'linear' )  # works 

    for j in range(len(x_space)):
        for k in range(len(y_space)): 
            yi = np.array( [X[k,j] , Y[k,j]] ) 
            Ji = J_use[k,j ]

            dE_0 =   compute_dE(yi[0],yi[1])
            cost_new_place = np.zeros( np.shape(u_opts) )   

            dE_n = np.zeros( np.shape(u_opts) )   
            for jj, u_c in enumerate(u_opts, start=0):  
                #----------------solve simulation 
                y_n  = odeint(   pendulum_ode, yi,  tU, args=(u_c,)  )[1]    # ode solver 
                y_n[0] =  wrap2periodic(y_n[0],2*np.pi, 0 )                  # wrap theta to range of -pi to pi 

                # find cost of the step
                J_n  =  calc_costToGo(yi,u_c, Q,R )  * tU[1] 

                # find the effective change of the step
                dE_n[jj] =   compute_dE(y_n[0],y_n[1])

                # compute cost at new step
                cost_new_place[jj] = spline( y_n[0],y_n[1]  )   + J_n

            # compare costs and find which control action is cheapest
            ind_low = np.argmin(cost_new_place)  
            U[k,j] = u_opts[ind_low]  
            Eff[k,j] = dE_n[ind_low] 
    return U 


def eval_control_cost( x0 ,y0 ) : 
    t1 = x0[0]
    tau = x0[1]  
    
    # sim parameters 
    dt = 0.001; 
    tLast = 2. 
    
    n_steps = np.int(tLast/dt) 
    t = np.array([0,dt])  
    tInt = np.arange(0,tLast+dt, dt) ;
    
    vis_limit = 0.3
    visible_x = wrap2periodic( np.array([-1,1])*vis_limit + np.pi, 2*np.pi, np.pi  ) 


    Q = 1.
    R = 100.   
    un = 0
    time_sign = 1 
    err_factor = 1e7
    y = []     # initialize lists  
    u = [] 
    
    for j in range(n_steps):
        y.append(y0)  
        u.append(un)     
        t_sim = tInt[j]
        if   (t_sim> t1) & (t_sim< (t1+np.abs(tau) )) :
            un = -3
        else: 
            un = 0 
        y1 = odeint(   pendulum_ode, y0,  t, args=(un ,time_sign,)  )[1]     # ode solver 
        y1[0] =  wrap2periodic(y1[0]) 
         
        y0 = y1  
        
        if y1[0] > visible_x[0]:              
            break 
            
    # compute cost
    E =  compute_dE( np.array(y)[:,0], np.array(y)[:,1] ) 
    U = np.array(u)  
    e_error = np.abs(E[-1]) 
    if e_error < 0.3:
        e_error = 0  
        
    J = np.sum( E**2*Q+ U**2*R)  + err_factor*e_error
    t_list = tInt[:(j+1)] 
      
    return J, t_list, np.array(y), U   

def run_sim_opt( x0, y0 ) : 
    t1 = x0[0]
    tau = x0[1]  

    vis_limit = 0.3
    visible_x = wrap2periodic( np.array([-1,1])*vis_limit + np.pi, 2*np.pi, np.pi  ) 

    
    dth_LQR_limit = np.array([ 0 , 1.88])
    
    # sim parameters 
    dt = 0.001; 
    tLast = 2. 
    n_steps = np.int(tLast/dt) 
    t = np.array([0,dt])  
    tInt = np.arange(0,tLast+dt, dt) ;
    
    Q = 1.
    R = 100.  
    un = 0   
    err_factor = 1e7
    time_sign = 1 
    y = []
    u = []   # initialize lists 
    
    for j in range(n_steps):
        y.append(y0)  
        u.append(un)     
        t_sim = tInt[j]
        if  (t_sim> t1) & (t_sim< (t1+ np.abs(tau) )) :
            un = -3
        else: 
            un = 0 
        y1 = odeint(   pendulum_ode, y0,  t, args=(un ,time_sign,)  )[1]     # ode solver 
        y1[0] =  wrap2periodic(y1[0]) 
         
        y0 = y1   
        if y1[0] > visible_x[0]:              
            break  
            
    
    # compute cost
    LQR_conv_bool = (y1[0] > dth_LQR_limit[0] ) &   (y1[1] < dth_LQR_limit[1] )
    
    E =  compute_dE( np.array(y)[:,0], np.array(y)[:,1] )
    U = np.array(u) 
    
    e_error = np.abs(E[-1])  
    LQR_conv_bool = (y1[0] > dth_LQR_limit[0] ) &   (y1[1] < dth_LQR_limit[1] )
         
    if  LQR_conv_bool:
        e_error = 0  
    J = np.sum( E**2*Q+ U**2*R)  + err_factor*e_error 
 
     
    return J  

