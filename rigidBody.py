# this is file for a class rigidBody
# rigidBody is unbreakable ( in our task) system of material points
# each material point has several parametres : mass coordinates velocity *other
#                                               🡗 🡖    🡗  🡖      🡗    🡖     🡓
#                                    normal mass   moment of inertia
#                                                     y and x coordinates
#                                                            forward & angular velocity
#                                                                             usualy as other parametre will consist only of friction coefficient

# About problem we are solving:
# whole this "library" will and is written to simulate the rigid body dinamics, esp. with evolving frinction coefficient
# one of the purpose of this task after former -- simulation-visualisation is to check how the height of the posistion of the center of mass below curface affects body's movement
#																										     --------🠒    ----------🠒	 🠒																																									
# to solve this problem physically we need to calculate two definite integrals: friction force =       (ε * forward_unit + [angular_unit, r])  2
#                                                                                               -μ∫ σ*  -----------------------------------  dr
#                                                                                                             --------🠒    ----------🠒	 🠒
#                                                                                                      |(ε * forward_unit + [angular_unit, r])|  
#
#                                                                                                  🠒   --------------🠒
#                                                                 and moment of friction force = ∫[r, d(Friction force)]
#
# the problem is followiing : the integral can be calculated only numerically due to elliptical integrals in it's origid
#                                                                                                                 🠒      🠒 🠒    🠒
# That's why to simulate we can use eulers method, which is probably upgradded newton's law to torque system : I*dω/dt + [ω,L] = ΣM 
#												  whick becames these equasions in the cartesian system:
#																											  🠒               🠒 🠒    🠒
#																								        I1 * dω1/dt + (I3 - I1)ω2ω3 = ΣM1
#                                                                                                             🠒               🠒 🠒    🠒            
#																										I2 * dω2/dt + (I1 - I3)ω3ω1 = ΣM2
#                                                                                                             🠒               🠒 🠒    🠒   
#																										I3 * dω3/dt + (I2 - I1)ω1ω2 = ΣM3

import math

from termcolor import colored
#from Point import *


class RigidBody:

    imageExtension = ['.jpg', '.jpeg', '.png']
    textExtension  = ['.txt']

    def __init__(self, loadPath : str):
        if loadPath[loadPath.index('.'):] in RigidBody.imageExtension:
                ...
        elif loadPath[loadPath.index('.'):] in RigidBody.textExtension:
            print(f'''Note, that {loadPath} file should contain legend with key : value, pair in it 
            (, is the separator {colored('DO NOT FORGET', 'red')} to use it)                  {colored('↖ this is the needed', 'blue')}
                                    {colored('~~ ~~~ ~~~~~~', 'red')}                               {colored('and the only available format', 'blue')}
                Where former is a {colored('single-cell', 'blue')} symbol and latter stays for the mass-value that this symbol represents respectively
                Otherwise will be load stadart parametres:
                                    (* = 0, 
                                        # = 1,
                                        mu = 0.5) 
                Also, note, that if you want to change friction coefficient - 
                add frictionDistributionMap.txt to the {loadPath} folder
                ({colored('DO NOT SPECIFY', 'red')} friction coefficient in that case) 
                    {colored('~~ ~~~ ~~~~~~~', 'red')}''')

            with open(loadPath, 'r') as profile:
                profileSection = [section.rstrip('\n') for section in profile]
                
            height, width = len(profileSection[1:]), len(profileSection[1:][0])
            for row in profileSection[1:]:
                print(row)
            if not(all([len(row) == width for row in profileSection[1:]])):
                raise ValueError("rows in file should be same width")
                    
            if ', ' in profileSection[0]:
                section = profileSection[0].split(', ')
            elif ',' in profileSection[0]:
                section = profileSection[0].split(',')

            legend = {}
            for marker in section:
                if ' :' in marker:
                    legend[marker[:marker.index(' :')]] = float(marker[-marker.index(' :'):])
                elif ':' in marker:
                    legend[marker[:marker.index(':')]] = float(marker[-marker.index(':'):])
                else:
                    raise ValueError('Check the required legend format and try again')

            profileSection = profileSection[1:]
            bodyMap = [[[] for i in range(width)] for j in range(height)]
            for yIndex, row in enumerate(profileSection):
                for xIndex, el in enumerate(row):
                    try:
                        bodyMap[yIndex][xIndex] = legend[el]
                    except KeyError:
                        raise KeyError(f"Incorrect marker type in position {yIndex, xIndex}, check legend and try again")              

            if not(any([marker == 'mu' for marker in legend.keys()])):
                print("You haven't added mu, make sure there is frictionDistributionMap.txt otherwise mu = .5")
                if input("Do you want to use standart friction coefficient? >>> (y/n) ") == 'y':
                    legend['mu'] = .5
                    frictionMap = [[legend['mu'] for i in range(width)] for j in range(height)]

                else:
                    try:
                        print("frictionDistributionMap should have legend, the format remains (except the mu parameter obviously) \
                                \nHowever, there are no standart values")

                        with open('frictionDistributionMap') as distributionMap:
                            mapSection = [section.rstrip('\n') for section in distributionMap]

                        if ', ' in mapSection[0]:
                            section = mapSection[0].split(', ')

                        elif ',' in profileSection[0]:
                            section = mapSection[0].split(',')

                        else:
                            raise ValueError("Incorrect legend")
                                
                        mapLegend = {}
                        for marker in section:
                            if ' :' in marker:
                                mapLegend[marker[:marker.index(' :')]] = float(marker[-marker.index(' :'):])
                            elif ':' in marker:
                                mapLegend[marker[:marker.index(':')]] = float(marker[-marker.index(':'):])
                            else:
                                raise ValueError('Check the required legend format and try again')

                        mapSection = mapSection[1:]
                        frictionMap = [[[] for i in range(width)] for j in range(height)]
                        for yIndex, row in enumerate(mapSection):
                            for xIndex, el in enumerate(row):
                                try:
                                    frictionMap[yIndex][xIndex] = mapLegend[el]
                                except KeyError:
                                    raise KeyError(f"Incorrect marker type in position {yIndex, xIndex}, check legend and try again")

                    except FileNotFoundError:
                        raise FileNotFoundError("Create frictionDistributionMap.txt and try again")
            else:
                frictionMap = [[legend['mu'] for i in range(width)] for j in range(height)]
                        
        else:
            raise ValueError(f"Unavailable file format : {loadPath[loadPath.index('.'):]} \ncheck path and try again")
        self.bodyMap     = bodyMap
        self.frictionMap = frictionMap


import math
class RigidBody:
    def __init__(self, position, velocity, mass, inertia_tensor):
        """
        A class to represent a rigid body.
        :param position: a tuple of the x, y and z coordinates of the center of mass.
        :param velocity: a tuple of the x, y and z components of the velocity.
        :param mass: the mass of the rigid body.
        :param inertia_tensor: the 3x3 inertia tensor of the rigid body.
        """
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.force = (0, 0, 0)
        self.torque = (0, 0, 0)
        self.orientation = (1, 0, 0, 0)
        self.angular_velocity = (0, 0, 0)
    def integrate(self, dt):
        """
        Integrate the state of the rigid body over a time interval of dt using the Euler method.
        """
        # Compute the acceleration of the rigid body
        acceleration = tuple(f / self.mass for f in self.force)
        # Update the position and velocity of the rigid body
        for i in range(3):
            self.velocity[i] += dt * acceleration[i]
            self.position[i] += dt * self.velocity[i]
        # Compute the angular acceleration of the rigid body
        angular_acceleration = tuple(
            sum(
                self.inertia_tensor[i][j] * self.torque[j]
                for j in range(3)
            )
            for i in range(3)
        )
        # Update the orientation and angular velocity of the rigid body
        quat_dt = (
            0.5 * dt * angular_acceleration[0],
            0.5 * dt * angular_acceleration[1],
            0.5 * dt * angular_acceleration[2],
        )
        self.orientation = tuple(
            quat_dt[i] * self.orientation[i+1] + self.orientation[i]
            for i in range(3)
        )
        self.angular_velocity = tuple(
            quat_dt[i] * self.orientation[i+1] + self.angular_velocity[i]
            for i in range(3)
        )
import numpy as np

def simulate_rigid_body_plane_dynamics(mass, inertia, initial_pos, initial_vel, initial_angle, initial_ang_vel, time_step, total_time):
    """
    This function simulates the rigid body plane dynamics using the Euler method.
    
    Parameters:
    mass (float): The mass of the rigid body
    inertia (float): The moment of inertia of the rigid body
    initial_pos (numpy.ndarray): A 2D numpy array representing the initial position of the rigid body in the x-y plane
    initial_vel (numpy.ndarray): A 2D numpy array representing the initial velocity of the rigid body in the x-y plane
    initial_angle (float): The initial angle of the rigid body with respect to the x-axis
    initial_ang_vel (float): The initial angular velocity of the rigid body
    time_step (float): The time step for the simulation
    total_time (float): The total time for the simulation
    
    Returns:
    numpy.ndarray: A 2D numpy array representing the position of the rigid body at each time step
    """
    try:
        # Check if the initial position and velocity are 2D numpy arrays
        if not isinstance(initial_pos, np.ndarray) or not isinstance(initial_vel, np.ndarray) or initial_pos.shape != (2,) or initial_vel.shape != (2,):
            raise TypeError("Initial position and velocity must be 2D numpy arrays")
        
        # Initialize the position and velocity arrays
        pos = np.zeros((int(total_time/time_step)+1, 2))
        vel = np.zeros((int(total_time/time_step)+1, 2))
        pos[0] = initial_pos
        vel[0] = initial_vel
        
        # Initialize the angle and angular velocity arrays
        angle = np.zeros(int(total_time/time_step)+1)
        ang_vel = np.zeros(int(total_time/time_step)+1)
        angle[0] = initial_angle
        ang_vel[0] = initial_ang_vel
        
        # Calculate the acceleration and angular acceleration arrays
        acc = np.zeros((int(total_time/time_step)+1, 2))
        ang_acc = np.zeros(int(total_time/time_step)+1)
        
        for i in range(int(total_time/time_step)):
            # Calculate the acceleration and angular acceleration at the current time step
            acc[i] = np.array([0, -9.81]) + np.array([np.cos(angle[i]), np.sin(angle[i])]) * np.sum(np.array([0, -mass*9.81]))
            ang_acc[i] = np.sum(np.array([0, -1]) * np.cross(np.array([0, 0, ang_vel[i]]), np.array([0, 0, inertia])))
            
            # Update the position, velocity, angle, and angular velocity arrays using the Euler method
            pos[i+1] = pos[i] + vel[i] * time_step
            vel[i+1] = vel[i] + acc[i] * time_step
            angle[i+1] = angle[i] + ang_vel[i] * time_step
            ang_vel[i+1] = ang_vel[i] + ang_acc[i] * time_step
        
        return pos
    
    except TypeError as e:
        # Log the error
        print(f"Error: {e}")
        return np.zeros((1, 2))
