#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pdb
from carla import Transform, Location, Rotation

import random
import time
import pdb

# import for MPC  
from scipy.optimize import minimize
from test_mpc import ModelPredictiveControl
import time
 
        
def run_mpc(mpc_controller, ref_seq, state_seq, u_frame, bounds):
    # u_frame: input sequence of each frame after mpc computation
    u_frame = np.delete(u_frame,0)
    u_frame = np.delete(u_frame,0)
    u_frame = np.append(u_frame, u_frame[-2])
    u_frame = np.append(u_frame, u_frame[-2])
    
    u_solution = minimize(mpc_controller.cost_function, u_frame, (state_seq[-1], ref_seq[-1]), 
                          method='SLSQP', bounds=bounds, tol = 1e-5)
    u_frame = u_solution.x 
    return u_frame[0], u_frame[1]
    
        
        

def visualize_image(image):
	data = np.array(image.raw_data) # shape is (image.height * image.width * 4,) 
	data_reshaped = np.reshape(data, (image.height, image.width,4))
	rgb_3channels = data_reshaped[:,:,:3] # first 3 channels
	
	cv2.imshow("image",rgb_3channels )
	cv2.waitKey(10)

def main():
    actor_list = [] 
    try: 
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        # pdb.set_trace()
        
        world = client.load_world("/Game/Carla/Maps/Town04") 
        blueprint_library = world.get_blueprint_library()
        #bp = random.choice(blueprint_library.filter('vehicle'))
        bp1 = blueprint_library.filter('vehicle')[0]
        bp2 = blueprint_library.filter('vehicle')[1]
 
        #transform = random.choice(world.get_map().get_spawn_points())
        
        # Always fix the starting position 
        #transform1 = world.get_map().get_spawn_points()[0] 
        transform1 = Transform(Location(x=230.252029, y=-364.109375, z=0.281942), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        print(transform1)
        vehicle_lead = world.spawn_actor(bp1, transform1) 
        print(vehicle_lead.get_transform().rotation)
        
        transform2 = Transform(Location(x=225.252029, y=-364.109375, z=0.281942), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        print(transform2)
        vehicle_follow = world.spawn_actor(bp2, transform2)
        # pdb.set_trace()

        # So let's tell the world to spawn the vehicle.  
        actor_list.append(vehicle_lead)
        actor_list.append(vehicle_follow)
        print('created leading vahicle %s' % vehicle_lead.type_id)
        print('created following vahicle %s' % vehicle_follow.type_id)

        # Let's put the vehicle to drive around. 
        vehicle_lead.set_autopilot(True)
        #vehicle_follow.set_autopilot(True)
        
        
        frame = 0
        rgb_list = list()
        
        # Let's add now an "RGB" camera attached to the vehicle.
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb.set_attribute('image_size_x',  str(640))
        camera_bp_rgb.set_attribute('image_size_y',  str(320))
        camera_bp_rgb.set_attribute('fov',  str(100))
        camera_transform_rgb = carla.Transform(carla.Location(x=-7.0, z=2.4))
        camera_rgb = world.spawn_actor(camera_bp_rgb, camera_transform_rgb, attach_to=vehicle_follow)
        actor_list.append(camera_rgb)
        print('created %s' % camera_rgb.type_id)
        camera_rgb.listen(lambda image: rgb_list.append(image) if frame > 10 else None	)
        
        # create mpc controller
        mpc = ModelPredictiveControl()
        num_inputs = 2
        u = np.zeros(mpc.horizon*num_inputs)
        input_bounds = []
        for i in range(mpc.horizon):
            input_bounds += [[0, 1]] # throttle bound
            input_bounds += [[-0.8, 0.8]]
        
        state_i = np.array([[0,0,0,0]])
        ref_i = np.array([[0,0,0,0]]) 
        u_i = np.array([[0,0]])   
        #predict_info = [state_i]
         
        for frame in range(100):
            # Do tick
            world.tick()
            if frame>10:
                visualize_image(rgb_list[-1]) 
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        
            print('frame %s' % frame)
            print("Throttle: {}, Steering: {}, Brake: {}".format(vehicle_lead.get_control().throttle, vehicle_lead.get_control().steer, vehicle_lead.get_control().brake))
            print("Vehicle location: (x,y,z): ({},{},{})".format(vehicle_lead.get_location().x,vehicle_lead.get_location().y, vehicle_lead.get_location().z ))
            print("Vehicle velocity: {} ".format(vehicle_lead.get_velocity().x))
            print("Vehicle yaw_angle: {} ".format(vehicle_lead.get_transform().rotation.yaw))
            
            #pdb.set_trace() 
            
            v_lead_vec = np.array([vehicle_lead.get_velocity().x, vehicle_lead.get_velocity().y,
                                    vehicle_lead.get_velocity().z])
            v_t_lead = np.linalg.norm(v_lead_vec)
            psi_t_lead = vehicle_lead.get_transform().rotation.yaw
            new_ref_lead = [vehicle_lead.get_location().x, vehicle_lead.get_location().y, psi_t_lead, v_t_lead]
            print('new_ref_lead', new_ref_lead)
            ref_i = np.append(ref_i, np.array([new_ref_lead]), axis = 0)
            #print(ref_i)
            
            v_fol_vec = np.array([vehicle_follow.get_velocity().x, vehicle_follow.get_velocity().y,
                                    vehicle_follow.get_velocity().z])
            v_t_fol = np.linalg.norm(v_fol_vec)
            psi_t_fol = vehicle_follow.get_transform().rotation.yaw * np.pi / 180
            new_ref_fol = [vehicle_follow.get_location().x, vehicle_follow.get_location().y, psi_t_fol, v_t_fol]
            print('new_ref_fol', new_ref_fol)
            state_i = np.append(state_i, np.array([new_ref_fol]), axis = 0)
            #print(state_i)
            
            throttle, str_angle = run_mpc(mpc, ref_i, state_i, u, input_bounds)
            print('throttle and steering angle of following car: ({},{})'.format(throttle, str_angle))
            u_i = np.append(u_i, np.array([(throttle, str_angle)]), axis=0) 
            vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
            #vehicle_follow.apply_control(carla.VehicleControl(throttle=vehicle_lead.get_control().throttle, steer=vehicle_lead.get_control().steer))
            
        

    finally:

        print('destroying actors')
        camera_rgb.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
