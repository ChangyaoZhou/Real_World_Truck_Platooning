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
CAR_WIDTH = 2.0
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
    plt.plot(car_follow.get_location().x,car_follow.get_location().y, 'b', marker=".", markersize=1)
    p_car.set_xy(car_patch_pos(car_follow.get_location().x,car_follow.get_location().y, 
                               car_follow.get_transform().rotation.yaw))
    psi_follow = car_follow.get_transform().rotation.yaw
    if psi_follow < 0:
        psi_follow += 360   
    p_car.angle = psi_follow - 90 

    # vehicle_lead
    plt.plot(car_lead.get_location().x,car_lead.get_location().y, 'r', marker=".", markersize=1)
    p_goal.set_xy(car_patch_pos(car_lead.get_location().x,car_lead.get_location().y, car_lead.get_transform().rotation.yaw))
    psi_lead = car_lead.get_transform().rotation.yaw 
    if psi_lead < 0:
        psi_lead += 360 
    p_goal.angle = psi_lead - 90  
    text_pt.set_text("frame : {}".format(frame))
    text_vlead.set_text('velocity of the target vehicle: % .2f m/s' % norm_vector(car_lead.get_velocity()))
    text_vfollow.set_text('velocity of the ego vehicle: % .2f m/s' % norm_vector(car_follow.get_velocity()))
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
        transform1 = world.get_map().get_spawn_points()[5] 
        #transform1.location.x -= 40 
        #transform1 = Transform(Location(x=230.252029, y=-364.109375, z=0.0), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        #transform1 = Transform(Location(x=223.05120849609375,y=-388.6428527832031,z=-0.007714137900620699), Rotation(pitch=0.000000, yaw=-179.5698699951172, roll=0.000000))
        vehicle_lead = world.spawn_actor(bp1, transform1) 
        
        #transform2 = world.get_map().get_spawn_points()[5] 
        transform2 = transform1
        transform2.location.x += 8
        #transform2 = Transform(Location(x=222.252029, y=-364.109375, z=0.0), Rotation(pitch=0.000000, yaw=-0.295319, roll=0.000000))
        #transform2 = Transform(Location(x=231.05120849609375,y=-388.6428527832031,z=-0.007714137900620699), Rotation(pitch=0.000000, yaw=-179.5698699951172, roll=0.000000))
        vehicle_follow = world.spawn_actor(bp2, transform2)
        #vehicle_lead.apply_control(carla.VehicleControl(throttle=all_throttle[0], steer=all_steer[0]))
        #vehicle_follow.apply_control(carla.VehicleControl(throttle=control_u[1][0], steer=control_u[1][1]))
        # pdb.set_trace()

        # So let's tell the world to spawn the vehicle.  
        actor_list.append(vehicle_lead)
        actor_list.append(vehicle_follow)
        print('created target vehicle %s' % vehicle_lead.type_id)
        print('created ego vehicle %s' % vehicle_follow.type_id)

        # Let's put the vehicle to drive around. 
        vehicle_lead.set_autopilot(True)
        #vehicle_follow.set_autopilot(True)
        physics_vehicle = vehicle_follow.get_physics_control()
        car_mass = physics_vehicle.mass
         
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
        
        # create mpc controller
        mpc = ModelPredictiveControl()
        num_inputs = 2
        u = np.zeros(mpc.horizon*num_inputs)
        input_bounds = []
        for i in range(mpc.horizon):
            input_bounds += [[0, 1]] # throttle bound
            input_bounds += [[-1, 1]]
        
        #state_i = np.array([[222.252029, -364.109375,-0.295319,0]])
        state_i = np.array([[223.1039276123047,-395.6426086425781,-3.1340854687929687,0]])
        #ref_i = np.array([[230.252029,-364.109375,-0.295319,0]]) 
        ref_i = np.array([[231.1039276123047,-395.6426086425781,-3.1340854687929687,0]]) 
        u_i = np.array([[0,0]])   
        
         ##### SIMULATOR DISPLAY ######### 

        # Total Figure
        fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs = gridspec.GridSpec(8,8)

        # Elevator plot settings.
        ax = fig.add_subplot(gs[:8, :8]) 
        plt.xlim(-200, 300)
        ax.set_ylim([-400, -100])
        plt.xticks(np.arange(-200, 301, step=50))
        plt.yticks(np.arange(-400, -99, step=10))
        plt.title('MPC 2D')

        
        # Main plot info. 
        patch_car = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', fill=True)
        patch_goal = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', ls='dashdot', fill=True)

        ax.add_patch(patch_car)
        ax.add_patch(patch_goal)
        predict, = ax.plot([], [], 'r--', linewidth = 1) 
        print('Create vehicle_follow and vehicle_lead in the figure.')
        
        # plot text
        text_pt = plt.text(100, -125, '', fontsize=8)
        text_vlead = plt.text(100, -135, '', fontsize=8)
        text_vfollow = plt.text(100, -145, '', fontsize=8)
        text_dis = plt.text(100, -155, '', fontsize = 8)
        
        gear_list = []
        
        for frame in range(800):
            print('frame %s' % frame)
            # Do tick
            world.tick()
            
            if frame == 40:
                impulse = 15.0 *car_mass
                vehicle_follow.add_impulse(carla.Vector3D(0, impulse, 0)) 
                
            if (frame % 40) == 0 and frame != 0 and frame != 40:                
                impulse = random.uniform(4.0,7.0) *car_mass
                minus_list = [-1,1]
                impulse_minus = random.choice(minus_list)
                impulse = impulse_minus *impulse
                impulse_axis = random.randint(0,1)
                if impulse_axis == 0:
                    vehicle_follow.add_impulse(carla.Vector3D(impulse, 0, 0))
                elif impulse_axis == 1:
                    vehicle_follow.add_impulse(carla.Vector3D(0, impulse, 0))                
                print('impulse:{}, axis:{}'.format(impulse,impulse_axis))
            
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        
                        
            print("Vehicle location: (x,y,z): ({},{},{})".format(vehicle_follow.get_location().x,vehicle_follow.get_location().y, vehicle_follow.get_location().z ))
            #print("Vehicle velocity: {} ".format(vehicle_lead.get_velocity().x))
            print('Gear', vehicle_follow.get_control().gear)
            print("Throttle: {}, Steering: {}, Brake: {}".format(vehicle_follow.get_control().throttle, vehicle_follow.get_control().steer, vehicle_follow.get_control().brake))
            print('Vehicle velocity', norm_vector(vehicle_follow.get_velocity()))
            print('ref_velocity', norm_vector(vehicle_lead.get_velocity()))
            print('Vehicle yaw', vehicle_lead.get_transform().rotation.yaw)
            
            #pdb.set_trace() 
            vehicle_lead.set_autopilot(True)
            #vehicle_follow.set_autopilot(True)
            v_lead_vec = np.array([vehicle_lead.get_velocity().x, vehicle_lead.get_velocity().y,
                                    vehicle_lead.get_velocity().z])
            v_t_lead = np.linalg.norm(v_lead_vec)
            psi_t_lead = vehicle_lead.get_transform().rotation.yaw * np.pi / 180
            new_ref_lead = [vehicle_lead.get_location().x, vehicle_lead.get_location().y, psi_t_lead, v_t_lead]
            #print('new_ref_lead', new_ref_lead)
            ref_i = np.append(ref_i, np.array([new_ref_lead]), axis = 0)
            #print(ref_i)
            
            v_fol_vec = np.array([vehicle_follow.get_velocity().x, vehicle_follow.get_velocity().y,
                                    vehicle_follow.get_velocity().z])
            v_t_fol = np.linalg.norm(v_fol_vec)
            psi_t_fol = vehicle_follow.get_transform().rotation.yaw * np.pi / 180
            new_ref_fol = [vehicle_follow.get_location().x, vehicle_follow.get_location().y, psi_t_fol, v_t_fol]
            #print('new_ref_fol', new_ref_fol) 
            state_i = np.append(state_i, np.array([new_ref_fol]), axis = 0)
            #print(state_i)
            
            print('ref_yaw', ref_i[-1][2])
            print('state_yaw', state_i[-1][2])
            
            
            throttle, str_angle = run_mpc(mpc, ref_i, state_i, u, input_bounds)
            #y = mpc.plant_model(state_i[-1], mpc.dt, throttle, str_angle)
            #state_i = np.append(state_i, np.array([y]), axis=0) 
            #print('throttle and steering angle of following car: ({},{})'.format(throttle, str_angle))
            u_i = np.append(u_i, np.array([(throttle, str_angle)]), axis=0) 
            #vehicle_lead.apply_control(carla.VehicleControl(throttle=all_throttle[frame+1], steer=all_steer[frame+1]))
            #vehicle_follow.apply_control(carla.VehicleControl(throttle=control_u[frame+2][0], steer=control_u[frame+2][1]))
            vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
            
            ###### Visualization ######
            update_plot(frame, patch_car, patch_goal, vehicle_lead, vehicle_follow, 
                        text_pt, text_vlead, text_vfollow, text_dis, frame) 
            fig.canvas.draw()  
            img_loc = get_img_from_fig(fig, dpi=180) 
            if frame > 11:
                visualize_image(rgb_list[-1], img_loc)               
        
    
    finally:
        state_list = np.append(ref_i, state_i,axis=1)
        np.set_printoptions(threshold=sys.maxsize)
        f1 = open("./output/test01/state_list.py","a")
        f1.write(str(repr(state_list)))
        f2 = open("./output/test01/control_list.py","a")
        f2.write(str(repr(u_i)))
        
        print('destroying actors')
        camera_rgb.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
