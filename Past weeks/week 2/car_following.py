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
# import for visualization
import io
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from output.town04.steer import all_steer
from output.town04.throttle import all_throttle
from output.town04.location import all_location
from output.town04.all_u import control_u

IMG_SIZE = [800,560]
CAR_WIDTH = 1.0
CAR_LENGTH = 3.7
FIG_SIZE = [8,8]
        
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
    
def norm_vector(vector=carla.Vector3D):
    length = (vector.x**2 + vector.y**2 + vector.z**2)**(1/2)
    return length        
        

def visualize_image(image, plt):
    data = np.array(image.raw_data) # shape is (image.height * image.width * 4,) 
    data_reshaped = np.reshape(data, (image.height, image.width,4))
    rgb_3channels = data_reshaped[:,:,:3] # first 3 channels 
    flipped = cv2.flip(rgb_3channels, 1)
    image_sum = np.concatenate((plt, flipped), axis = 1) 
    cv2.imshow("plot location and camera view",image_sum)
    cv2.waitKey(10)
    
    
def car_patch_pos(x, y, psi, car_width = 1.0):
    # Shift xy, centered on rear of car to rear left corner of car. 
    x_new = x - np.sin(psi)*(car_width/2)
    y_new = y + np.cos(psi)*(car_width/2)
    return [x_new, y_new]

def update_plot(num, p_car, p_goal, car_lead, car_follow, text_pt, text_vlead, text_vfollow, text_dis, frame): 
    # vehicle_follow 
    p_car.set_xy(car_patch_pos(car_follow.get_location().x,car_follow.get_location().y, 
                               car_follow.get_transform().rotation.yaw))
    psi_follow = car_follow.get_transform().rotation.yaw
    if psi_follow < 0:
        psi_follow += 360   
    p_car.angle = psi_follow - 90 

    # vehicle_lead
    p_goal.set_xy(car_patch_pos(car_lead.get_location().x,car_lead.get_location().y, car_lead.get_transform().rotation.yaw))
    psi_lead = car_lead.get_transform().rotation.yaw 
    if psi_lead < 0:
        psi_lead += 360 
    p_goal.angle = psi_lead - 90  
    text_pt.set_text("frame = {}".format(frame))
    text_vlead.set_text('velocity of lead car: % .2f m/s' % norm_vector(car_lead.get_velocity()))
    text_vfollow.set_text('velocity of follow car: % .2f m/s' % norm_vector(car_follow.get_velocity()))
    car_dis = ((car_lead.get_location().x-car_follow.get_location().x) **2 + 
               (car_lead.get_location().y-car_follow.get_location().y) **2)**0.5
    text_dis.set_text('car distance: % .2f m' % car_dis)
    
    
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.resize(img,(IMG_SIZE[0], IMG_SIZE[1])) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img
    

def main():
    actor_list = [] 
    try: 
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        # pdb.set_trace()
        
        world = client.load_world("/Game/Carla/Maps/Town04") 
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        #bp = random.choice(blueprint_library.filter('vehicle'))
        bp1 = blueprint_library.filter('vehicle')[0]
        bp2 = blueprint_library.filter('vehicle')[0]
        color1 = bp1.get_attribute('color').recommended_values[0]
        color2 = bp2.get_attribute('color').recommended_values[1]
        bp2.set_attribute('color', color2)
 
        #transform = random.choice(world.get_map().get_spawn_points())
        
        # Always fix the starting position 
        #transform1 = world.get_map().get_spawn_points()[0] 
        transform1 = Transform(Location(x=230.252029, y=-364.109375, z=0.0), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        #transform1 = Transform(Location(x=-5.44616985e+00, y=-7.90550232e+01, z=2.21428841e-01), Rotation(pitch=0.000000, yaw=92.00418854, roll=0.000000))
        print(transform1)
        vehicle_lead = world.spawn_actor(bp1, transform1) 
        print(vehicle_lead.get_transform().rotation)
        
        transform2 = Transform(Location(x=222.252029, y=-364.109375, z=0.0), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        #transform2 = Transform(Location(x=-5.44616985e+00, y=-8.70550232e+01, z=2.21428841e-01), Rotation(pitch=0.000000, yaw=92.00418854, roll=0.000000))
        print(transform2)
        vehicle_follow = world.spawn_actor(bp2, transform2)
        vehicle_lead.apply_control(carla.VehicleControl(throttle=all_throttle[0], steer=all_steer[0]))
        vehicle_follow.apply_control(carla.VehicleControl(throttle=control_u[1][0], steer=control_u[1][1]))
        # pdb.set_trace()

        # So let's tell the world to spawn the vehicle.  
        actor_list.append(vehicle_lead)
        actor_list.append(vehicle_follow)
        print('created leading vahicle %s' % vehicle_lead.type_id)
        print('created following vahicle %s' % vehicle_follow.type_id)

        # Let's put the vehicle to drive around. 
        #vehicle_lead.set_autopilot(True)
        #vehicle_follow.set_autopilot(True)
         
        frame = 0
        rgb_list = list()
        
        # Let's add now an "RGB" camera attached to the vehicle.
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb.set_attribute('image_size_x',  str(IMG_SIZE[0]))
        camera_bp_rgb.set_attribute('image_size_y',  str(IMG_SIZE[1]))
        camera_bp_rgb.set_attribute('fov',  str(100))
        camera_transform_rgb = carla.Transform(carla.Location(x=-7.0, z=2.4))
        camera_rgb = world.spawn_actor(camera_bp_rgb, camera_transform_rgb, attach_to=vehicle_follow)
        actor_list.append(camera_rgb)
        print('created %s' % camera_rgb.type_id)
        camera_rgb.listen(lambda image: rgb_list.append(image) if frame > 5 else None)
        '''
        # create mpc controller
        mpc = ModelPredictiveControl()
        num_inputs = 2
        u = np.zeros(mpc.horizon*num_inputs)
        input_bounds = []
        for i in range(mpc.horizon):
            input_bounds += [[0, 1]] # throttle bound
            input_bounds += [[-0.8, 0.8]]
        
        state_i = np.array([[225.252029, -364.109375,-0.295319,0]])
        ref_i = np.array([[0,0,0]]) 
        u_i = np.array([[0,0]])   
        #predict_info = [state_i]'''
        
         ##### SIMULATOR DISPLAY ######### 

        # Total Figure
        fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs = gridspec.GridSpec(8,8)

        # Elevator plot settings.
        ax = fig.add_subplot(gs[:8, :8]) 
        plt.xlim(200, 400)
        ax.set_ylim([-400, -150])
        plt.xticks(np.arange(200, 401, step=10))
        plt.yticks(np.arange(-400, -149, step=10))
        plt.title('MPC 2D')
        ref_loc = all_location[:,:2]
        plt.plot(ref_loc[:,0], ref_loc[:,1],linewidth = 1) 
        
        # Main plot info. 
        patch_car = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', fill=False)
        patch_goal = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='b', ls='dashdot', fill=False)

        ax.add_patch(patch_car)
        ax.add_patch(patch_goal)
        predict, = ax.plot([], [], 'r--', linewidth = 1) 
        print('Create vehicle_follow and vehicle_lead in the figure.')
        
        # plot text
        text_pt = plt.text(210, -175, '', fontsize=8)
        text_vlead = plt.text(210, -185, '', fontsize=8)
        text_vfollow = plt.text(210, -195, '', fontsize=8)
        text_dis = plt.text(210, -205, '', fontsize = 8)
        
        gear_list = []
        
        for frame in range(684):
            # Do tick
            world.tick() 
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        
            print('frame %s' % frame)
            print("Throttle: {}, Steering: {}, Brake: {}".format(vehicle_follow.get_control().throttle, vehicle_follow.get_control().steer, vehicle_follow.get_control().brake))
            print("Vehicle location: (x,y,z): ({},{},{})".format(vehicle_follow.get_location().x,vehicle_follow.get_location().y, vehicle_follow.get_location().z ))
            #print("Vehicle velocity: {} ".format(vehicle_lead.get_velocity().x))
            print("Vehicle yaw_angle: {} ".format(vehicle_follow.get_transform().rotation.yaw))
            print('Gear', vehicle_follow.get_control().gear)
            gear_list.append(vehicle_follow.get_control().gear)
            print('Vehicle velocity', norm_vector(vehicle_follow.get_velocity()))
            print('ref_velocity', norm_vector(vehicle_lead.get_velocity()))
            
            #pdb.set_trace() 
            '''
            v_lead_vec = np.array([vehicle_lead.get_velocity().x, vehicle_lead.get_velocity().y,
                                    vehicle_lead.get_velocity().z])
            v_t_lead = np.linalg.norm(v_lead_vec)
            psi_t_lead = vehicle_lead.get_transform().rotation.yaw * np.pi / 180
            new_ref_lead = [vehicle_lead.get_location().x, vehicle_lead.get_location().y, psi_t_lead]
            print('new_ref_lead', new_ref_lead)
            #ref_i = np.append(ref_i, np.array([new_ref_lead]), axis = 0)
            #print(ref_i)
            
            v_fol_vec = np.array([vehicle_follow.get_velocity().x, vehicle_follow.get_velocity().y,
                                    vehicle_follow.get_velocity().z])
            v_t_fol = np.linalg.norm(v_fol_vec)
            psi_t_fol = vehicle_follow.get_transform().rotation.yaw * np.pi / 180
            new_ref_fol = [vehicle_follow.get_location().x, vehicle_follow.get_location().y, psi_t_fol, v_t_fol]
            print('new_ref_fol', new_ref_fol) 
            #state_i = np.append(state_i, np.array([new_ref_fol]), axis = 0)
            #print(state_i)
            
            print('ref_yaw', ref_i[-1][2])
            print('state_yaw', state_i[-1][2])
            
            throttle, str_angle = run_mpc(mpc, ref_i, state_i, u, input_bounds)
            y = mpc.plant_model(state_i[-1], mpc.dt, throttle, str_angle)
            state_i = np.append(state_i, np.array([y]), axis=0) 
            print('throttle and steering angle of following car: ({},{})'.format(throttle, str_angle))
            u_i = np.append(u_i, np.array([(throttle, str_angle)]), axis=0) '''
            vehicle_lead.apply_control(carla.VehicleControl(throttle=all_throttle[frame+1], steer=all_steer[frame+1]))
            vehicle_follow.apply_control(carla.VehicleControl(throttle=control_u[frame+2][0], steer=control_u[frame+2][1]))
            
            ###### Visualization ######
            update_plot(frame, patch_car, patch_goal, vehicle_lead, vehicle_follow, 
                        text_pt, text_vlead, text_vfollow, text_dis, frame) 
            fig.canvas.draw()  
            img_loc = get_img_from_fig(fig, dpi=180) 
            if frame > 5:
                visualize_image(rgb_list[-1], img_loc) 
             
        f1 = open("./output/town04/gear.py","a")
        f1.write(str(repr(gear_list)))
    
    finally:

        print('destroying actors')
        camera_rgb.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
