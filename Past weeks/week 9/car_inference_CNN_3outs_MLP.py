#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
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

# import for model
from models.my_models_alex2 import MyModel_CNN
from models.my_models_MLP import MyModel1
#from model.my_models_CNN import MyModel_CNN 
from torchvision import transforms
# import for visualization
import io
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

IMG_SIZE = [400,800,700] # location plot 800x700, two images 400x400
CAR_WIDTH = 2.0
CAR_LENGTH = 3.7
FIG_SIZE = [8,8]
INPUT_SIZE = [800,300]
MODEL_PATH_CNN = './models/mynet_alex_3outs_5.pth'
MODEL_PATH_MLP = './models/mynet_transform.pth'
    
def norm_vector(vector=carla.Vector3D):
    length = (vector.x**2 + vector.y**2 + vector.z**2)**(1/2)
    return length        
        

def visualize_image(image1, image2, plt):
    data = np.array(image1.raw_data) # shape is (image.height * image.width * 4,) 
    data_reshaped = np.reshape(data, (image1.height, image1.width,4))
    rgb1_3channels = data_reshaped[:,:,:3] # first 3 channels 
    flipped1 = cv2.flip(rgb1_3channels, 1)
    #print(flipped1.shape)
    
    #data = np.array(image2.raw_data) # shape is (image.height * image.width * 4,) 
    #data_reshaped = np.reshape(data, (image2.height, image2.width,4))
    rgb2_3channels = image2 # first 3 channels 
    #print(rgb2_3channels.shape)
    #rgb2_3channels = cv2.resize(rgb2_3channels, (IMG_SIZE[0],0.5 * IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    flipped2 = cv2.flip(rgb2_3channels, 1)
    #print(flipped2.shape)
    #print(plt.shape)
    
    image_cat = np.concatenate((flipped1, flipped2), axis = 0)
    image_sum = np.concatenate((plt, image_cat), axis = 1) 
    cv2.imshow("plot location and camera view",image_sum)
    cv2.waitKey(10)
    
    
def car_patch_pos(x, y, psi, car_width = 1.0):
    # Shift xy, centered on rear of car to rear left corner of car. 
    x_new = x - np.sin(psi)*(car_width/2)
    y_new = y + np.cos(psi)*(car_width/2)
    return [x_new, y_new]

def update_plot(num, p_car, p_goal, car_lead, car_follow, text_list, frame): 
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
    text_list[0].set_text("frame : {}".format(frame))
    text_list[1].set_text('velocity of the target vehicle: % .2f m/s' % norm_vector(car_lead.get_velocity()))
    text_list[2].set_text('velocity of the ego vehicle: % .2f m/s' % norm_vector(car_follow.get_velocity()))
    car_dis = ((car_lead.get_location().x-car_follow.get_location().x) **2 + 
               (car_lead.get_location().y-car_follow.get_location().y) **2)**0.5
    text_list[3].set_text('car distance: % .2f m' % car_dis)
    text_list[4].set_text('yaw angle of the target vehicle: % .4f' % (car_lead.get_transform().rotation.yaw * np.pi / 180))
    text_list[5].set_text('yaw angle of the ego vehicle: % .4f' % (car_follow.get_transform().rotation.yaw * np.pi / 180))
    text_list[6].set_text('dis/v_ref:% .2f ' % (car_dis/(norm_vector(car_lead.get_velocity())+0.01)))
    
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.resize(img,(IMG_SIZE[2], IMG_SIZE[1])) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img
    
def compute_rel_tran(ref_state, ego_state):
    x1, y1, theta1, v1 = ref_state
    x2, y2, theta2, v2 = ego_state
    theta1 = (theta1+2*np.pi)%(2*np.pi)
    theta2 = (theta2+2*np.pi)%(2*np.pi)
    G1 = np.array([[np.cos(theta1), -np.sin(theta1), x1], 
                   [np.sin(theta1), np.cos(theta1), y1],
                   [0, 0, 1]])
    G2 = np.array([[np.cos(theta2), -np.sin(theta2), x2], 
                   [np.sin(theta2), np.cos(theta2), y2],
                   [0, 0, 1]])
    T12 = np.linalg.inv(G2).dot(G1)
    delta_x = T12[0][2]
    delta_y = T12[1][2] 
    delta_theta = np.arcsin(T12[1][0])
    return np.array([delta_x, delta_y, delta_theta, v1, v2])[None, :]

def center_resize(img, w, h):
    image = np.array(img.raw_data)
    img_reshaped = np.reshape(image, (img.height, img.width, 4)) 
    img_input = img_reshaped[:,:,:3].reshape((img.height, img.width, 3))
    #x = img_input.shape[0]/2 - w/2
    y = img_input.shape[1]/2 - h/2
    #print(x,y)
    crop_img = img_input[:int(w), int(y):int(y+h),:]
    crop_img = np.array(crop_img, dtype='uint8')
    resize_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_AREA)
    return resize_img

def main():
    actor_list = [] 
    try: 
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        # pdb.set_trace()
        
        world = client.load_world("/Game/Carla/Maps/Town03") 
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        #bp = random.choice(blueprint_library.filter('vehicle'))
        bp1 = blueprint_library.filter('vehicle')[0]
        bp2 = blueprint_library.filter('vehicle')[0]
        color1 = random.choice(bp1.get_attribute('color').recommended_values)
        color2 = bp2.get_attribute('color').recommended_values[1]
        bp1.set_attribute('color', color1)
        bp2.set_attribute('color', color2)
   
        # Always fix the starting position 
        transform1 = world.get_map().get_spawn_points()[150]
        #transform1 = Transform(Location(x=14.631523132324219, y=-320.760498046875, z=0.0), Rotation(pitch=0.000000, yaw=126.48568725585938 , roll=0.000000))
        vehicle_lead = world.spawn_actor(bp1, transform1)
        
        transform2 = transform1
        transform2.location.y -= 8 
        #transform2 = Transform(Location(x=20.405834197998047, y=-328.1571044921875, z=0.0), Rotation(pitch=0.000000, yaw=129.0533905029297 , roll=0.000000))

        vehicle_follow = world.spawn_actor(bp2, transform2) 
        # pdb.set_trace()

        # So let's tell the world to spawn the vehicle.  
        actor_list.append(vehicle_lead)
        actor_list.append(vehicle_follow)
        print('created target vehicle %s' % vehicle_lead.type_id)
        print('created ego vehicle %s' % vehicle_follow.type_id)
        
        physics_vehicle = vehicle_follow.get_physics_control()
        car_mass = physics_vehicle.mass
        
        # Let's put the vehicle to drive around. 
        vehicle_lead.set_autopilot(True) 
        
        # add spectator view
        spectator = world.get_actors().filter('spectator')[0]
        spectator.set_transform(carla.Transform(transform1.location +carla.Location(z=600)+carla.Location(x=-80)+carla.Location(y=100),carla.Rotation(pitch=-90)))
         
        frame = 0
        rgb1_list = []
        rgb2_list = []
        
        # RGB camera 1 for general view
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb.set_attribute('image_size_x',  str(IMG_SIZE[0]))
        camera_bp_rgb.set_attribute('image_size_y',  str(0.5 * IMG_SIZE[1]))
        camera_bp_rgb.set_attribute('fov',  str(100))
        camera_transform_rgb = carla.Transform(carla.Location(x=-7.0, z=2.5))
        camera_rgb = world.spawn_actor(camera_bp_rgb, camera_transform_rgb, attach_to=vehicle_follow)
        actor_list.append(camera_rgb)
        print('created %s' % camera_rgb.type_id)
        camera_rgb.listen(lambda image: rgb1_list.append(image) if frame > 5 else None)
        
        # RGB camera 2 for data collection
        camera_bp_rgb2 = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb2.set_attribute('image_size_x',  str(INPUT_SIZE[0]))
        camera_bp_rgb2.set_attribute('image_size_y',  str(INPUT_SIZE[1]))
        camera_bp_rgb2.set_attribute('fov',  str(130))
        camera_transform_rgb2 = carla.Transform(carla.Location(x=1.8, z=1.3))
        camera_rgb2 = world.spawn_actor(camera_bp_rgb2, camera_transform_rgb2, attach_to=vehicle_follow)
        actor_list.append(camera_rgb2)
        print('created %s' % camera_rgb2.type_id)        
        camera_rgb2.listen(lambda image: rgb2_list.append(center_resize(image, 250, 400)))
         
        
        #state_i = np.array([[222.252029, -364.109375,-0.295319,0]])
        state_i = np.array([[231.025146484375,-385.14300537109375, -2.9595362983108866, 0]])
        #ref_i = np.array([[230.252029,-364.109375,-0.295319,0]]) 
        ref_i = np.array([[223.025146484375,-385.14300537109375,-3.1340854687929687,0]]) 
        u_i = np.array([[0,0]])   
        
         ##### SIMULATOR DISPLAY ######### 

        # Total Figure
        fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs = gridspec.GridSpec(8,8)

        # Elevator plot settings.
        ax = fig.add_subplot(gs[:8, :8]) 
        plt.xlim(-500, 500)
        ax.set_ylim([-500, 500])
        plt.xticks(np.arange(-500, 501, step=30))
        plt.yticks(np.arange(-500, 501, step=30))
        plt.title('MPC 2D')

        
        # Main plot info. 
        patch_car = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', fill=True)
        patch_goal = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', ls='dashdot', fill=True)

        ax.add_patch(patch_car)
        ax.add_patch(patch_goal)
        predict, = ax.plot([], [], 'r--', linewidth = 1) 
        print('Create vehicle_follow and vehicle_lead in the figure.')
        
        # plot text
        text_pt = plt.text(-450, 450, '', fontsize=8)
        text_vlead = plt.text(-450, 420, '', fontsize=8)
        text_vfollow = plt.text(-450, 390, '', fontsize=8)
        text_dis = plt.text(-450, 360, '', fontsize = 8)
        text_yawlead = plt.text(-450, 330, '', fontsize = 8)
        text_yawfol = plt.text(-450, 300, '', fontsize = 8)
        text_ratio = plt.text(-450, 270, '', fontsize=8) 
        text_list = [text_pt, text_vlead, text_vfollow, text_dis, text_yawlead, text_yawfol, text_ratio]
         
        control_values = []
        channel = [30, 30, 30, 3]
        kernel = 5
        neuron = [64, 32]
        stride = 2 
        mynet_CNN= MyModel_CNN() 
        mynet_MLP = MyModel1(neurons = [256, 1024, 256]) 
        #mynet= MyModel_CNNalex()
        #mynet= MyModel_CNN() 
        
        mynet_CNN.load_state_dict(torch.load(MODEL_PATH_CNN, map_location='cpu'))
        mynet_MLP.load_state_dict(torch.load(MODEL_PATH_MLP, map_location='cpu'))
        
        for frame in range(1000):
            print('frame %s' % frame)
            # Do tick
            world.tick()
            vehicle_lead.set_autopilot(True) 
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
            
            '''
            #### test by adding random impulses
            if (frame % 40) == 0 and frame != 0:          
                impulse = random.uniform(4.0,8.0) *car_mass
                minus_list = [-1,1]
                impulse_minus = random.choice(minus_list)
                impulse = impulse_minus *impulse
                impulse_axis = random.randint(0,1)
                if impulse_axis == 0:
                    vehicle_follow.add_impulse(carla.Vector3D(impulse, 0, 0))
                elif impulse_axis == 1:
                    vehicle_follow.add_impulse(carla.Vector3D(0, impulse, 0))                
                print('impulse:{}, axis:{}'.format(impulse,impulse_axis))
            '''     
            print('TARGET Vehicle')
            print("Location: (x,y,z): ({},{},{})".format(vehicle_lead.get_location().x,vehicle_lead.get_location().y, vehicle_lead.get_location().z))
            print("Throttle: {}, Steering Angle: {}, Brake: {}".format(vehicle_lead.get_control().throttle, vehicle_lead.get_control().steer, vehicle_lead.get_control().brake))
            print('EGO Vehicle')
            print("Location: (x,y,z): ({},{},{})".format(vehicle_follow.get_location().x,vehicle_follow.get_location().y, vehicle_follow.get_location().z))
            print("Throttle: {}, Steering Angle: {}, Brake: {}".format(vehicle_follow.get_control().throttle, vehicle_follow.get_control().steer, vehicle_follow.get_control().brake))
             
            #pdb.set_trace()  
            v_lead_vec = np.array([vehicle_lead.get_velocity().x, vehicle_lead.get_velocity().y,
                                    vehicle_lead.get_velocity().z])
            v_t_lead = np.linalg.norm(v_lead_vec)
            psi_t_lead = vehicle_lead.get_transform().rotation.yaw * np.pi / 180
            new_ref_lead = [vehicle_lead.get_location().x, vehicle_lead.get_location().y, psi_t_lead, v_t_lead] 
              
            v_fol_vec = np.array([vehicle_follow.get_velocity().x, vehicle_follow.get_velocity().y,
                                  vehicle_follow.get_velocity().z])
            v_t_fol = np.linalg.norm(v_fol_vec)
            psi_t_fol = vehicle_follow.get_transform().rotation.yaw * np.pi / 180
            new_ref_fol = [vehicle_follow.get_location().x, vehicle_follow.get_location().y, psi_t_fol, v_t_fol] 
            
            if frame == 0:
                ref_i = np.expand_dims(np.array(new_ref_lead), axis=0)
                state_i = np.expand_dims(np.array(new_ref_fol), axis=0)
            else:
                ref_i = np.append(ref_i, np.array([new_ref_lead]), axis = 0)
                state_i = np.append(state_i, np.array([new_ref_fol]), axis = 0) 
            
            # input preparation
            #img_input = cv2.resize(rgb2_list[-1], (128,128), interpolation = cv2.INTER_AREA)
            img_input = rgb2_list[-1]
            print('image shape', img_input.shape)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_input = transform(img_input/255) 
            
            model_input = [torch.tensor(img_input), torch.tensor(v_t_lead), torch.tensor(v_t_fol)] 
            
            gt_input = compute_rel_tran(ref_i[-1], state_i[-1])
            # model inference for CNN 
            #mynet.eval() 
            output_CNN = mynet_CNN(model_input) 
            input_MLP = torch.cat((output_CNN[0], torch.tensor([v_t_lead,v_t_fol])), axis = 0)
            # model inference for MLP 
            mynet_MLP.eval() 
            #print(input_MLP.shape)
            print('gt output',gt_input)
            print('CNN output',input_MLP)
            input_MLP = torch.unsqueeze(input_MLP, 0)
            output_MLP = mynet_MLP(input_MLP.to(torch.float32))
            #print(output_pred)
            throttle, str_angle = output_MLP[0].detach().numpy()
            throttle = float(throttle)  
            str_angle = float(str_angle) 
            control_values.append([throttle, str_angle])   
            u_i = np.append(u_i, np.array([(throttle, str_angle)]), axis=0)  
            print('throttle', throttle)
            print('str_angle', str_angle)
            vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
            
            ###### Visualization ######
            update_plot(frame, patch_car, patch_goal, vehicle_lead, vehicle_follow, text_list, frame) 
            fig.canvas.draw()  
            img_loc = get_img_from_fig(fig, dpi=180) 
            if frame > 11:
                #img_fol = cv2.resize(rgb2_list[-1], (IMG_SIZE[0],0.5 * IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
                rgb2 = cv2.resize(rgb2_list[-1], (400, 400), interpolation = cv2.INTER_AREA)
                visualize_image(rgb1_list[-1], rgb2, img_loc)   
    
    finally:

        print('destroying actors')
        camera_rgb.destroy()
        camera_rgb2.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
