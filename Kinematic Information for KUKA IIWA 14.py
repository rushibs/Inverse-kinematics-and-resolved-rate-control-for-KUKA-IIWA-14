# %%
"""
# Project 1: computing all kinematic informations of a real manipulator

In this project (and the next ones), we will build all the necessary various functionalities for realistic robot manipulators. This first project aims to build the core fonctions (basic homogeneous transforms, twists, forward kinematics and Jacobians) that will be a foundation for all subsequent algorithms.

## Instructions
* Answer all questions in the notebook
* You will need to submit on Brightspace: 
    1. the code you wrote to answer the questions in a Jupyter Notebook. The code should be runnable as is.
    2. a 2-3 pages report in pdf format (pdf only) detailing the methodology you followed to answer the questions as well as answers to the questions that require a written answer. You may add the plots in the report (does not count for the page limit) or in the Jupyter notebook.


## The robot

We will use a model of the [Kuka iiwa 14 robot](https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa). This robot has 7 revolute joints and its kinematics is described in the picture below:

![](https://raw.githubusercontent.com/righetti/ROB6003/143afa17d7eb1af79c3f6ce034973a1774da5d42/Project1/kuka_kinematics.png "Kuka iiwa 14 Kinematic Model")
"""

# %%
"""
# Setup

Run the cell below only once when resetting the runtime in Colab - this will not do anything when running on a local Jupyter Notebook.
"""

# %%
"""
# Starting the visualization environment

The following code will start a visualization environment (click on the printed address to see the robot)

You need to run this only ONCE. Each time you run this cell you will get a new display environment (so you need to close the previous one!)

This should work out of the box on Google Colab and you local Jupyter Notebook (make sure you have installed the right libraries in your local computer if you do not use Colab).
"""

# %%
import numpy as np
import robot_visualizer
import time

import matplotlib.pyplot as plt

robot_visualizer.start_robot_visualizer()

# %%
"""
# Displaying an arbitrary configuration

You can use the following function to display arbitrary configurations of the robot
"""

# %%
# here we display an arbitrary configuration of the robot
q = np.random.sample([7])
print(f'we show the configuration for the angles {q}')
robot_visualizer.display_robot(q)

# %%
"""
## Question 1: basics
In this first set of questions, we aim to write the basic functions to do kinematics
* Write a function ``vec_to_skew(w)`` that transforms a 3D vector (numpy array) into a skew symmetric matrix
* Write a function ``twist_to_skew(V)`` that transforms a 6D twist into a 4x4 matrix (use ``vec_to_skew``)
* Write a function ``exp_twist_bracket(V)`` that returns the exponential of a (bracketed) twist $\mathrm{e}^{[\mathcal{V}]}$ where the input to the function is a 6D twist
* Write a function ``inverseT(T)`` that returns the inverse of a homogeneous transform T
* Write a function ``getAdjoint(T)`` that returns the adjoint of a homogeneous transform T
"""

# %%
import numpy as np
from scipy.linalg import *

################################################################

def vec_to_skew(w):
    return np.array([[0, -w[2,0], w[1,0]], 
                     [w[2,0], 0, -w[0,0]], 
                     [-w[1,0], w[0,0], 0]])
# w = np.array([[2],[3],[5]])
# print("w_ss = ", vec_to_skew(w))
################################################################

def twist_to_skew(V):
    w_ss = vec_to_skew(V[:3])
    v0 = np.concatenate((w_ss,V[3:]), axis=1)
    skew = np.concatenate((v0, np.array([[0,0,0,0]])), axis=0)
    return(skew)
V = np.array([[0],[1],[1],[0],[3],[5]])
skew1 = twist_to_skew(V)
#################################################################

def skew_to_twist(skew):
    
    mat = np.array([[skew[2][1]], [skew[0][2]], [skew[1][0]], [skew[0][3]], [skew[1][3]], [skew[2][3]]])  
    
    return(mat)
def exp_twist_bracket(V):
    ss = twist_to_skew(V)
    ss_exp = expm(ss)
    V = skew_to_twist(ss_exp)
    
    return(V)
       
# print("V_exp = ", exp_twist_bracket(V))
##################################################################

def inverseT(T):
    inv_T = inv(T)
    return(inv_T)
##################################################################

def skew(vector):
    
    return np.array([[0, -vector[0,2], vector[0,1]], 
                     [vector[0,2], 0, -vector[0,0]], 
                     [-vector[0,1], vector[0,0], 0]])
def getAdjoint(T):
    R_sb = np.array(T[0:3, 0:3])
    P_sb_ss = skew(np.array([T[0:3,3]]))
    temp1 = np.concatenate((R_sb, np.zeros((3,3))), axis=1)
    temp2 = np.concatenate((np.matmul(P_sb_ss, R_sb), R_sb), axis=1)
    adj = np.concatenate((temp1, temp2), axis=0)
    return(adj)
# print(getAdjoint(skew1))

# %%
"""
## Question 2: forward kinematics
* Write a function ``forward_kinematics(theta)`` that gets as an input an array of joint angles and computes the pose of the end-effector.

In order to test this function, you are given the following forward kinematics results (up to $10^{-4}$ precision),

$T_{SH}(0,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0) = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1.301 \end{bmatrix}$,

$T_{SH}(0.2,\ -0.2,\ 0.5,\ -0.4,\ 1.2,\ -0.8,\ 0.4) = \begin{bmatrix}
-0.4951 & -0.814 &  0.3037 & -0.0003 \\
0.6286 & -0.5769 & -0.5215 &  0.0056\\
0.5997 & -0.0673 &  0.7974 &  1.2563\\
0.  &    0.  &    0.  &    1.\end{bmatrix}$

$T_{SH}(-1.2,\ 0.7,\ 2.8,\ 0.7,\ 1.2,\ 0.2,\ 0.3) = \begin{bmatrix}
-0.9669 & -0.254 &  -0.0234 &  0.1535\\
0.0976 & -0.2835 & -0.954 &  -0.7557\\
0.2357 & -0.9247 &  0.2989 &  0.795\\
 0.  &    0.  &    0.  &    1.\end{bmatrix}$
"""

# %%
def lin_vel(w, q):
    w_ss = skew(w)
    v = np.matmul(-w_ss, q)
    return(v)

def forward_kinematics(theta):    
    exp1 = expm(S1_ss*theta[0]) 
    exp2 = expm(S2_ss*theta[1])
    exp3 = expm(S3_ss*theta[2])
    exp4 = expm(S4_ss*theta[3])
    exp5 = expm(S5_ss*theta[4])
    exp6 = expm(S6_ss*theta[5])
    exp7 = expm(S7_ss*theta[6])
    
    T_01 = np.matmul(exp1, exp2)
    T_01 = np.matmul(T_01, exp3)
    T_01 = np.matmul(T_01, exp4)    
    T_01 = np.matmul(T_01, exp5)
    T_01 = np.matmul(T_01, exp6)
    T_01 = np.matmul(T_01, exp7)    
    T_01 = np.matmul(T_01, M)
    
    return(T_01)
    
M = np.array([[1,0,0,0], [0,1,0,0], [0, 0, 1, 1.301], [0, 0, 0, 1]])   
theta = np.array([[0.2], [-0.2], [0.5], [-0.4], [1.2], [-0.8], [0.4]])

w1 = np.array([[0,0,1]])
w2 = np.array([[0,1,0]])
w3 = np.array([[0,0,1]])
w4 = np.array([[0,-1,0]])
w5 = np.array([[0,0,1]])
w6 = np.array([[0,1,0]])
w7 = np.array([[0,0,1]])

q1 = np.array([0,0,0.1575])
q2 = np.array([0,0,0.36])
q3 = np.array([0,0,0.5645])
q4 = np.array([0,0,0.78])
q5 = np.array([0,0,0.9645])
q6 = np.array([0,-0.0607,1.18])
q7 = np.array([0,0,1.261])

v1 = lin_vel(w1,q1)
v2 = lin_vel(w2,q2)
v3 = lin_vel(w3,q3)
v4 = lin_vel(w4,q4)
v5 = lin_vel(w5,q5)
v6 = lin_vel(w6,q6)
v7 = lin_vel(w7,q7)

S1 = np.array([[w1[0,0]],[w1[0,1]],[w1[0,2]],[v1[0]],[v1[1]],[v1[2]]])
S2 = np.array([[w2[0,0]],[w2[0,1]],[w2[0,2]],[v2[0]],[v2[1]],[v2[2]]])
S3 = np.array([[w3[0,0]],[w3[0,1]],[w3[0,2]],[v3[0]],[v3[1]],[v3[2]]])
S4 = np.array([[w4[0,0]],[w4[0,1]],[w4[0,2]],[v4[0]],[v4[1]],[v4[2]]])
S5 = np.array([[w5[0,0]],[w5[0,1]],[w5[0,2]],[v5[0]],[v5[1]],[v5[2]]])
S6 = np.array([[w6[0,0]],[w6[0,1]],[w6[0,2]],[v6[0]],[v6[1]],[v6[2]]])
S7 = np.array([[w7[0,0]],[w7[0,1]],[w7[0,2]],[v7[0]],[v7[1]],[v7[2]]])

S1_ss = twist_to_skew(S1)
S2_ss = twist_to_skew(S2)
S3_ss = twist_to_skew(S3)
S4_ss = twist_to_skew(S4)
S5_ss = twist_to_skew(S5)
S6_ss = twist_to_skew(S6)
S7_ss = twist_to_skew(S7)

fk = forward_kinematics(theta)
i=0
j=0

for i in range(fk.shape[0]):
    for j in range(fk.shape[1]):
        fk[i][j] = np.round(fk[i][j], 4)
    
# print(np.array2string(fk, separator= ','))

# %%
"""
## Question 3: jacobians
* Write a function ``get_space_jacobian(theta)`` that computes the space jacobian given an array of joint angles

In order to test this function, you are given the following space Jacobian results (up to $10^{-3}$ precision),
$J^S(0,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0) = \begin{bmatrix}
   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000\\
   0.000 &   1.000 &   0.000 & -1.000 &   0.000 &   1.000 &   0.000\\
   1.000 &   0.000 &   1.000 &   0.000 &   1.000 &   0.000 &   1.000\\
   0.000 & -0.360 &   0.000 &   0.780 &   0.000 & -1.180 &   0.000\\
   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000\\
   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000 &   0.000
\end{bmatrix}
$,

$J^S(0.2,\ -0.2,\ 0.5,\ -0.4,\ 1.2,\ -0.8,\ 0.4) = \begin{bmatrix}
   0.000 & -0.199 & -0.195 &   0.635 &   0.112 & -0.943 &   0.304\\
   0.000 &   0.980 & -0.039 & -0.767 &   0.213 & -0.287 & -0.522\\
   1.000 &   0.000 &   0.980 &   0.095 &   0.971 &   0.172 &   0.797\\
   0.000 & -0.353 &   0.014 &   0.590 & -0.181 &   0.344 &   0.660\\
   0.000 & -0.072 & -0.070 &   0.498 &   0.166 & -1.087 &   0.382\\
   0.000 &   0.000 &   0.000 &   0.073 & -0.016 &   0.075 & -0.002
\end{bmatrix}$

$J^S(-1.2,\ 0.7,\ 2.8,\ 0.7,\ 1.2,\ 0.2,\ 0.3) = \begin{bmatrix}
   0.000 &   0.932 &   0.233 &   0.971 &   0.146 & -0.528 & -0.023\\
   0.000 &   0.362 & -0.600 &   0.103 & -0.970 & -0.242 & -0.954\\
   1.000 &   0.000 &   0.765 & -0.216 &   0.194 & -0.814 &   0.299\\
   0.000 & -0.130 &   0.216 & -0.015 &   0.612 &   0.705 &   0.533\\
   0.000 &   0.336 &   0.084 &   0.683 &   0.080 & -0.274 & -0.065\\
   0.000 &   0.000 &   0.000 &   0.255 & -0.058 & -0.376 & -0.164
\end{bmatrix}$
"""

# %%
def get_space_jacobian(theta):
    J1 = S1
    m1 = expm(S1_ss*theta[0])
    adj2 = getAdjoint(m1) 
    J2 = np.matmul(adj2, S2)
    m2 = expm(S2_ss*theta[1])
    p2 = np.matmul(m1,m2)
    adj3 = getAdjoint(p2)
    J3 = np.matmul(adj3, S3)
    
    m3 = expm(S3_ss*theta[2])
    p3 = np.matmul(p2, m3)
    adj4 = getAdjoint(p3)
    J4 = np.matmul(adj4, S4)  
    
    m4 = expm(S4_ss*theta[3])
    p4 = np.matmul(p3, m4)
    adj5 = getAdjoint(p4)
    J5 = np.matmul(adj5, S5)
    
    m5 = expm(S5_ss*theta[4])
    p5 = np.matmul(p4, m5)
    adj6 = getAdjoint(p5)
    J6 = np.matmul(adj6, S6)
    
    m6 = expm(S6_ss*theta[5])
    p6 = np.matmul(p5, m6)
    adj7 = getAdjoint(p6)
    J7 = np.matmul(adj7, S7)
    
    J_s = np.concatenate((J1, J2, J3, J4, J5, J6, J7), axis=1)
    
    return(J_s)

J_s = get_space_jacobian(theta)

for i in range(J_s.shape[0]):
    for j in range(J_s.shape[1]):
        J_s[i][j] = np.round(J_s[i][j], 3)
   
#print(np.array2string(J_s, separator=', '))   

# %%
"""
### Hint: Q2 and Q3
Depending on which method you use to compute the quantities of Q2 and Q3, you will need to define a series of fixed homogeneous transforms, screws, etc. You may want to store these values in various variables that you can reuse (i.e. define the fixed kinematic parameters once and for all).

You may also want to store some intermediate results to later compute the Jacobians.

Feel free to design the data structure that you prefer and to also pass additional parameters or return multiple variables with these functions if it simplifies your design. You can also put these functions in a class if you wish. Any solution is ok, as long as you can compute the requested quantities.

Make sure to explain your design in the report.
"""

# %%
"""
## Question 4: displaying hand trajectories 
You are given a file ``joint_trajectory.npy`` which contains a sequence of 100 joint configurations (cf. below) corresponding to a motion of the robot over time.
* Compute the position of the hand (i.e. the origin of the frame H) in the spatial frame for all 100 joint configuration
* Plot x-y position of the hand for all the configurations (i.e. a 2D plot with x as the abscissa and y as the ordinate of the graph). What does the hand draw?
* Do the same analysis and plots for the x-z and y-z pairs.

### Hint
You may use (matplotlib)[https://matplotlib.org/] to draw plots
"""

# %%
# we open the file and put all the data in the variable joint_trajectory 
# this gives us a 7 x 200 array (each column in one set of joint configurations)
with open('joint_trajectory.npy', 'rb') as f:
    joint_trajectory = np.load(f)
    
# we display the trajectory
n_samples = joint_trajectory.shape[1]
for i in range(n_samples):
    robot_visualizer.display_robot(joint_trajectory[:,i])
    time.sleep(0.05) # we wait between two displays so we can see each configuration
    

# %%
# now we plot the joint trajectories for each joint (each cross correspond to one data point)
plt.figure(figsize=[10,15])
for i in range(7):
    plt.subplot(7,1,i+1)
    plt.plot(joint_trajectory[i,:], 'x', linewidth=4)
    plt.ylabel(f'joint {i+1}', fontsize=30)
    
pose_x = []
pose_y = []
pose_z = []

for c in range(n_samples):
    theta = joint_trajectory[:,c]
    pose = forward_kinematics(theta)
    pose_x.append(pose[0,-1])
    pose_y.append(pose[1,-1])
    pose_z.append(pose[2,-1])

plt.subplot(3,1,1)
plt.plot(pose_x, pose_y, '-', color= 'red', linewidth=4)
plt.subplot(3,1,2)
plt.plot(pose_x, pose_z, '-', linewidth=4)
plt.subplot(3,1,3)
plt.plot(pose_y, pose_z, '-', color= 'green', linewidth=4)
# plt.show()

# %%
"""
## Question 5: computing velocities
The file ``joint_velocities.npy`` contains the velocities of each joint corresponding to the sequence joint configurations seen in the previous question. 
* Use the Jacobian to compute the linear velocity of the endeffector in: 1) the spatial frame. 2) the end-effector frame and 3) in a frame with same origin as the end-effector frame but oriented like the spatial frame
* Plot these velocities in each frame (one plot per dimension x,y,z)
* Compare these plots and relate them to the plot of the positions (Question 4), is there a frame that seems most intuitive to you? Why?
"""

# %%
# we open the file and put all the data in the variable joint_trajectory 
# this gives us a 7 x 200 array (each column in one set of joint configurations)
with open('joint_velocity.npy', 'rb') as f:
    joint_velocities = np.load(f)
    
# now we plot the joint velocities for each joint (each cross correspond to one data point)
plt.figure(figsize=[10,15])
for i in range(7):
    plt.subplot(7,1,i+1)
    plt.plot(joint_velocities[i,:], 'x', linewidth=4)
    plt.ylabel(f'joint {i+1}', fontsize=30)

# %%
#####Part 1
v_x = []
v_y = []
v_z = []
for c in range(n_samples):
    theta = joint_trajectory[:,c]
    J = get_space_jacobian(theta)
    dtheta = joint_velocities[:,c]
    v = np.matmul(J, dtheta)
    v_x.append(v[3])
    v_y.append(v[4])
    v_z.append(v[5])

plt.subplot(3,1,1)
plt.plot(v_x, '-', color= 'red', linewidth=4)
plt.subplot(3,1,2)
plt.plot(v_y, '-', linewidth=4)
plt.subplot(3,1,3)
plt.plot(v_z, '-', color= 'green', linewidth=4)
# plt.show()
    

# %%
#####Part 2
v_x = []
v_y = []
v_z = []
for c in range(n_samples):
    theta = joint_trajectory[:,c]
    dtheta = joint_velocities[:,c]
    J_S = get_space_jacobian(theta)
    transform = forward_kinematics(theta)
    inv_t = inverseT(transform)
    adj_T = getAdjoint(inv_t)
    J_B = np.matmul(adj_T, J_S)   ###Body Jacobian
    v = np.matmul(J_B, dtheta)
    v_x.append(v[3])
    v_y.append(v[4])
    v_z.append(v[5])

plt.subplot(3,1,1)
plt.plot(v_x, '-', color= 'red', linewidth=4)
plt.subplot(3,1,2)
plt.plot(v_y, '-', linewidth=4)
plt.subplot(3,1,3)
plt.plot(v_z, '-', color= 'green', linewidth=4)
# plt.show()
    


# %%
######Part 3
v_x = []
v_y = []
v_z = []


for c in range(n_samples):
    theta = joint_trajectory[:,c]
    dtheta = joint_velocities[:,c]
    J_S = get_space_jacobian(theta)
    transform = forward_kinematics(theta)
    transform[:3,:3] = np.eye(3)
    inv_t = inverseT(transform)
    adj_T = getAdjoint(inv_t)
    J_B = np.matmul(adj_T, J_S)   ###Body Jacobian
    v = np.matmul(J_B, dtheta)
    v_x.append(v[3])
    v_y.append(v[4])
    v_z.append(v[5])

plt.subplot(3,1,1)
plt.plot(v_x, '-', color= 'red', linewidth=4)
plt.subplot(3,1,2)
plt.plot(v_y, '-', linewidth=4)
plt.subplot(3,1,3)
plt.plot(v_z, '-', color= 'green', linewidth=4)
# plt.show()
    