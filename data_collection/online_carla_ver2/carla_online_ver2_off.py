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


# basic settings
IMG_SIZE = [1000,750,750]
CAR_WIDTH = 1.0
CAR_LENGTH = 3.7
FIG_SIZE = [8,8]
N_sample = 20
N_frame = 20


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

def update_plot(p_car, p_goal, car_lead, car_follow, text_list, num, num_sample, frame, control):  
    # vehicle_follow 
    state_follow = get_current_state(car_follow)
    plt.plot(state_follow[0],state_follow[1], 'b', marker=".", markersize=1)
    p_car.set_xy(car_patch_pos(state_follow[0], state_follow[1], state_follow[2]))
    psi_follow = state_follow[2]*180/np.pi 
    if psi_follow < 0:
        psi_follow += 360   
    p_car.angle = psi_follow - 90 
    
    # vehicle_lead 
    state_lead = get_current_state(car_lead)
    p_goal.set_xy(car_patch_pos(state_lead[0], state_lead[1], state_lead[2]))
    psi_lead = state_lead[2]*180/np.pi
    if psi_lead < 0:
        psi_lead += 360 
    p_goal.angle = psi_lead - 90 
    # update text
    car_dis = ((car_lead.get_location().x-car_follow.get_location().x) **2 + 
               (car_lead.get_location().y-car_follow.get_location().y) **2)**0.5
    text_list[0].set_text("frame in trajectory : {}".format(num))
    text_list[1].set_text("num of sampled points : {}/20".format(num_sample+1))
    text_list[2].set_text("frame : {}/20".format(frame+1)) 
    text_list[3].set_text("throttle of ego vehicle: % .3f" % control[0])
    text_list[4].set_text("steering angle of ego vehicle: % .3f" % control[1])
    text_list[5].set_text('velocity of ego vehicle: % .2f m/s' % norm_vector(car_follow.get_velocity()))
    text_list[6].set_text('dis/v_ref:% .2f ' % (car_dis/(norm_vector(car_lead.get_velocity())+0.01)))

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

def set_random_transfer(vehicle, transform, v_follow_vec):   
    x_bias = np.random.uniform(-3, 3)
    y_bias = np.random.uniform(-3, 3)
    yaw_bias = np.random.uniform(-10, 10)
    v_bias = np.random.uniform(-2, 2)

    yaw = transform.rotation.yaw + yaw_bias
    x = transform.location.x + x_bias
    y = transform.location.y + y_bias
    
    v_fol = np.linalg.norm(v_follow_vec)
    new_v = v_fol + v_bias
    vx_follow = new_v * np.cos(transform.rotation.yaw * np.pi / 180) 
    vy_follow = new_v * np.sin(transform.rotation.yaw * np.pi / 180)
 
    vehicle.set_transform(Transform(Location(x=x,y=y,z=transform.location.z), Rotation(pitch=0.000000,yaw=yaw,roll=0.000000)))
    vehicle.set_target_velocity(carla.Vector3D(x=vx_follow, y=vy_follow, z=vehicle.get_velocity().z)) 
    print('Transform', vehicle.get_transform())
    print('Transform of ego vehicle changed.')

def get_current_state(vehicle):
    transform = vehicle.get_transform() 
    v_fol_vec = np.array([vehicle.get_velocity().x, vehicle.get_velocity().y, vehicle.get_velocity().z])
    v_fol = np.linalg.norm(v_fol_vec)
    return [transform.location.x, transform.location.y, transform.rotation.yaw * np.pi / 180, v_fol]

def visualize_image(image, plt):
    data = np.array(image.raw_data) # shape is (image.height * image.width * 4,) 
    data_reshaped = np.reshape(data, (image.height, image.width,4))
    rgb_3channels = data_reshaped[:,:,:3] # first 3 channels 
    flipped = cv2.flip(rgb_3channels, 1)
    image_sum = np.concatenate((plt, flipped), axis = 1) 
    cv2.imshow("off-trajectory data collection",image_sum)
    cv2.waitKey(10)

def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0) 

        ############### settings in CARLA simulator ########################
        world = client.load_world("/Game/Carla/Maps/Town04") 
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library() 
        bp1 = blueprint_library.filter('vehicle')[0]
        bp2 = blueprint_library.filter('vehicle')[0]
        color1 = bp1.get_attribute('color').recommended_values[0]
        color2 = bp2.get_attribute('color').recommended_values[1]
        bp2.set_attribute('color', color2)

        # set the starting position
        transform1 = world.get_map().get_spawn_points()[5] 
        vehicle_lead = world.spawn_actor(bp1, transform1)
        transform2 = transform1
        transform2.location.x += 8
        vehicle_follow = world.spawn_actor(bp2, transform2)

        actor_list.append(vehicle_lead)
        actor_list.append(vehicle_follow)
        print('created target vehicle %s' % vehicle_lead.type_id)
        print('created ego vehicle %s' % vehicle_follow.type_id)

        # set target vehicle in AUTOPILOT mode
        vehicle_lead.set_autopilot(True)
       
        
        # add a rgb-camera on ego evhicle
        rgb_list = list()
        
        # RGB camera 1 for general view
        camera_bp_rgb1 = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb1.set_attribute('image_size_x',  str(750))
        camera_bp_rgb1.set_attribute('image_size_y',  str(750))
        camera_bp_rgb1.set_attribute('fov',  str(100))
        camera_transform_rgb1 = carla.Transform(carla.Location(x=-7.0, z=2.5))
        camera_rgb1 = world.spawn_actor(camera_bp_rgb1, camera_transform_rgb1, attach_to=vehicle_follow)
        actor_list.append(camera_rgb1)
        print('created %s' % camera_rgb1.type_id) 
        camera_rgb1.listen(lambda image: rgb_list.append(image) if frame > 10 else None)
        
        # RGB camera 2 for data collection
        camera_bp_rgb2 = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb2.set_attribute('image_size_x',  str(800))
        camera_bp_rgb2.set_attribute('image_size_y',  str(600))
        camera_bp_rgb2.set_attribute('fov',  str(100))
        camera_transform_rgb2 = carla.Transform(carla.Location(x=0.0, z=2.0))
        camera_rgb2 = world.spawn_actor(camera_bp_rgb2, camera_transform_rgb2, attach_to=vehicle_follow)
        actor_list.append(camera_rgb2)
        print('created %s' % camera_rgb2.type_id)
        camera_rgb2.listen(lambda image: image.save_to_disk('./data_collection/online_carla_ver2/output/images_off/%06d_%02d_%02d.png' % (frame,n+1,f+1)) if n >= 0 else None)
        
         
            
        #################### create mpc controller #####################
        mpc = ModelPredictiveControl()
        num_inputs = 2 
        u = np.zeros(mpc.horizon*num_inputs) 
        input_bounds = []
        for i in range(mpc.horizon):
            input_bounds += [[0, 1]] # throttle bound
            input_bounds += [[-1, 1]] # steering angle bound
            
        # build state list
        state_i = np.array([[231.025146484375,-385.14300537109375, -3.1340854687929687, 0]]) 
        ref_i = np.array([[223.025146484375,-385.14300537109375,-3.1340854687929687,0]]) 
        u_i = np.array([[0,0]])
        state_collect = []
        u_collect = []

        ################## plot visualization #########################
        # Total Figure
        fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs = gridspec.GridSpec(8,8)

        # Elevator plot settings.
        ax = fig.add_subplot(gs[:8, :8]) 
        plt.xlim(-200, 240)
        ax.set_ylim([-410, -100])
        plt.xticks(np.arange(-200, 239, step=20))
        plt.yticks(np.arange(-410, -99, step=20))
        plt.title('off-trajectory data collection')

        # add text on plot
        text_num = plt.text(-190, -120, '', fontsize=10)
        text_sample = plt.text(-190, -130, '', fontsize=10)
        text_frame = plt.text(-190, -140, '', fontsize=10)
        text_pedal = plt.text(0, -120, '', fontsize=10)
        text_str = plt.text(0, -130, '', fontsize=10)  
        text_v = plt.text(0, -140, '', fontsize=10) 
        text_ratio = plt.text(0, -150, '', fontsize=10) 
        text_list = [text_num, text_sample, text_frame, text_pedal, text_str, text_v, text_ratio]

        # Main plot info. 
        patch_car = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', fill=True)
        patch_goal = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', ls='dashdot', fill=False)

        ax.add_patch(patch_car)
        ax.add_patch(patch_goal)
        predict, = ax.plot([], [], 'r--', linewidth = 1)  
        print('Create vehicle_follow and vehicle_lead in the figure.')

    ##################### data collection ################################
        
        for frame in range(801): 
            print('frame {}'.format(frame)) 
            n = -1
            world.tick()
            vehicle_lead.set_autopilot(True)
            
            
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)
                    
            print('TARGET Vehicle')
            print("Location: (x,y,z): ({},{},{})".format(vehicle_lead.get_location().x,vehicle_lead.get_location().y, vehicle_lead.get_location().z))
            print("Throttle: {}, Steering Angle: {}, Brake: {}".format(vehicle_lead.get_control().throttle, vehicle_lead.get_control().steer, vehicle_lead.get_control().brake))
            print('EGO Vehicle')
            print("Location: (x,y,z): ({},{},{})".format(vehicle_follow.get_location().x,vehicle_follow.get_location().y, vehicle_follow.get_location().z))
            print("Throttle: {}, Steering Angle: {}, Brake: {}".format(vehicle_follow.get_control().throttle, vehicle_follow.get_control().steer, vehicle_follow.get_control().brake))
             
            if frame > 39 and frame % 10 == 0:
                transform_lead = vehicle_lead.get_transform()
                transform_follow = vehicle_follow.get_transform()
                v_fol_vec = np.array([vehicle_follow.get_velocity().x, 
                                      vehicle_follow.get_velocity().y, vehicle_follow.get_velocity().z])
                v_fol_velocity = vehicle_follow.get_velocity()
                v_lead_vec = vehicle_lead.get_velocity()
                print('here')
                print('v_lead_vec', v_lead_vec)
                state_collect = []
                u_collect = [] 
                
                for n in range(N_sample):                    
                    set_random_transfer(vehicle_follow, transform_follow, v_fol_vec)
                    vehicle_lead.set_transform(transform_lead)
                    print('initial target yaw', vehicle_lead.get_transform().rotation.yaw)
                    vehicle_lead.set_target_velocity(v_lead_vec)
                    print('initial target vehicle velocity', norm_vector(vehicle_lead.get_velocity()))
                    print('random sample point {} at frame {}'.format(n+1, frame))
                   
                    for f in range(N_frame):   
                        #world.apply_settings(settings)
                        world.tick() 
                        vehicle_lead.set_autopilot(True)
                        new_ref_state = get_current_state(vehicle_lead)
                        #print('new ref state:',new_ref_state)
                        new_ego_state = get_current_state(vehicle_follow) 
                        #print('new ego state:',new_ego_state)
                        ref_i = np.append(ref_i,np.array([new_ref_state]), axis = 0)  
                        state_i = np.append(state_i, np.array([new_ego_state]), axis = 0) 
                        
                        throttle, str_angle = run_mpc(mpc, ref_i, state_i, u, input_bounds) 
                        u_i = np.append(u_i, np.array([[throttle, str_angle]]), axis=0) 
                        vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
                        
                        state_collect_i = []
                        state_collect_i.extend(new_ref_state)
                        state_collect_i.extend(new_ego_state) 
                        state_collect.append(state_collect_i)
                        u_collect.append([throttle, str_angle])
                        #print('target vehicle velocity', norm_vector(vehicle_lead.get_velocity()))
                        
                        # save the image path, v and outputs into .txt file
                        f_data = open('./data_collection/online_carla_ver2/output/offdata_image.txt','a+')
                        print('images_off/%06d_%02d_%02d.png' % (frame,n+1,f+1),
                              ref_i[-1][3], state_i[-1][3], throttle, str_angle, file = f_data)
                        f_data.close()
                                                
                        ############### Visualization #######################
                        update_plot(patch_car, patch_goal, vehicle_lead, vehicle_follow, text_list, frame, n, f, u_i[-1]) 
                        fig.canvas.draw()  
                        img_vis = get_img_from_fig(fig, dpi=180) 
                        visualize_image(rgb_list[-1], img_vis)
                
                #n = -1
                vehicle_lead.set_transform(transform_lead)                
                vehicle_lead.set_target_velocity(v_lead_vec)
                vehicle_follow.set_transform(transform_follow)
                vehicle_follow.set_target_velocity(v_fol_velocity) 
                
            else:   
                ref_state = get_current_state(vehicle_lead)
                ego_state = get_current_state(vehicle_follow) 
                ref_i = np.append(ref_i,np.array([ref_state]), axis = 0)  
                state_i = np.append(state_i, np.array([ego_state]), axis = 0)

                throttle, str_angle = run_mpc(mpc, ref_i, state_i, u, input_bounds) 
                u_i = np.append(u_i, np.array([[throttle, str_angle]]), axis=0) 

                vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
                '''
                # save the image path and steering angle into .txt file
                f_data = open('./data_collection/online_carla_ver2/output/ondata_image.txt','a+')
                print('images_on/%06d.png' % frame,
                      ref_i[-1][3], state_i[-1][3], throttle, str_angle, file = f_data)
                f_data.close()
                '''
            
            ############### Visualization #######################
            update_plot(patch_car, patch_goal, vehicle_lead, vehicle_follow, text_list, frame, -1, -1, u_i[-1])  
            fig.canvas.draw()  
            img_vis = get_img_from_fig(fig, dpi=180)
            if frame > 11:
                visualize_image(rgb_list[-1], img_vis)
                
                
    finally:
        print('destroying actors')
        camera_rgb1.destroy()
        camera_rgb2.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')
        

    
if __name__ == '__main__':

    main()
