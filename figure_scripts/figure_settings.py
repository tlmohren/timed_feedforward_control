
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import dynamicpendulums.singlependulum as sp
import matplotlib as mpl
plt.style.use('latex_scientificPaperStyle.mplstyle')

####--------------------Visible region ---------------------------------------
vis_limit = 0.3
visible_x =  sp.wrap2periodic( np.array([-1,1])*vis_limit + np.pi, 2*np.pi, np.pi  )

####--------------------Plot colors---------------------------------------
ff_col = 'r'
lqr_col = 'm'
traj_col = 'k'

# u_trigger_col ='k'
# line_col_out = 'k'
line_col_out = 'm'
lqr_fill = 'm'
u_col = 'r'
u_trigger_col = 'm'
u_col = 'r'
pend_col = 'k'

homoclinic_col = np.ones((3))*0.5
line_col = np.array([1,1,1])*0.7
line_col_out = np.array([1,1,1])*0.8

obscure_col = np.array( [236,231,242] ) /255
obscure_alpha = 0.8
obscure_edge = 'k'

contour_col = np.array([[1,1,1,1]])*0.7
col = sns.color_palette("hls", 5)
colorsList = [col[2],np.array([255,255,200])/255,col[3]]
contr = matplotlib.colors.ListedColormap(colorsList)

annotate_font = 8# SMALL_SIZE
ANNOTATE_FONT =8
####--------------------fonts---------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rcParams['font.size'] = SMALL_SIZE
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title


plt.rcParams["font.family"] = "Times New Roman"
# matplotlib.rcParams.update({'font.size': 8})
