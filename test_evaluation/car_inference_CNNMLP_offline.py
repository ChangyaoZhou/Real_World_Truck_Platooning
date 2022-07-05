import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import argparse
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
from torchvision import transforms

# import for model
from model.my_models import MyModel_MLP_transform, MyModel_CNN2


# import for visualization
import io
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

IMG_SIZE = [400,800,800]  
CAR_WIDTH = 2.0
CAR_LENGTH = 3.7
FIG_SIZE = [8,8]
INPUT_SIZE = [800,300]
MODEL_PATH_STE = './model/trained_models/mynet_CNN_stereo.pth'
MODEL_PATH_DEP = './model/trained_models/mynet_CNN_depth.pth'
MODEL_PATH_MLP = './model/trained_models/mynet_MLP_transform.pth'  
SPAWN_POINTS = {(1,0):['y', -8], (1, 100): ['y', 8], (1, 150): ['x', -8],
                (3, 112):['y', 8], (3, 200):['y', 8], (3, 0): ['y', -8], 
                (4, 5): ['x', 8], (4, 368): ['xy', 5.5, -6], 
                (5, 2): ['y', -8], (5, 100):['x', 8], (5, 150):['x', -8]}    

def visualize_image(image1, image2, plt):
    data = np.array(image1.raw_data)  
    data_reshaped = np.reshape(data, (image1.height, image1.width,4))
    rgb1_3channels = data_reshaped[:,:,:3]  
    flipped1 = cv2.flip(rgb1_3channels, 1) 
    flipped2 = cv2.flip(image2, 1) 
    
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
    
    v_lead_vec = np.array([car_lead.get_velocity().x, car_lead.get_velocity().y, car_lead.get_velocity().z])
    v_follow_vec = np.array([car_follow.get_velocity().x, car_follow.get_velocity().y, car_follow.get_velocity().z])
    
    text_list[0].set_text("frame : {}".format(frame))
    text_list[1].set_text('velocity of the target vehicle: % .2f m/s' % np.linalg.norm(v_lead_vec))
    text_list[2].set_text('velocity of the ego vehicle: % .2f m/s' % np.linalg.norm(v_follow_vec))
    car_dis = ((car_lead.get_location().x-car_follow.get_location().x) **2 + 
               (car_lead.get_location().y-car_follow.get_location().y) **2)**0.5
    text_list[3].set_text('car distance: % .2f m' % car_dis) 
    text_list[4].set_text('dis/v_ref:% .2f ' % (car_dis/(np.linalg.norm(v_lead_vec)+0.01)))
    
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

def center_resize(img, w, h):
    x = img.shape[0]/2 - w/2
    y = img.shape[1]/2 - h/2 
    crop_img = img[int(x):int(x+w), int(y):int(y+h),:]
    crop_img = np.array(crop_img, dtype='uint8')
    resize_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_AREA)
    return crop_img, resize_img

def set_spawn_point(town, spawn_num, transform_lead, SPAWN_POINTS):  
    spawn_trans = SPAWN_POINTS[(town, spawn_num)]
    if spawn_trans[0] == 'x':
        transform_follow = transform_lead 
        transform_follow.location.x += spawn_trans[1] 
    elif spawn_trans[0] == 'y':
        transform_follow = transform_lead
        transform_follow.location.y += spawn_trans[1]
    elif spawn_trans[0] == 'xy':
        transform_follow = transform_lead
        transform_follow.location.x += spawn_trans[1]
        transform_follow.location.y += spawn_trans[2]
    return transform_follow
 

def main():
    parser = argparse.ArgumentParser(description='Input the starting position of both vehicles')
    parser.add_argument('-model', type=str, help='town number')
    parser.add_argument('-town', type=int, help='town number')
    parser.add_argument('-spawn_point', type=int, help='starting point of target vehicle') 
    args = parser.parse_args()
    
    actor_list = [] 
    try: 
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0) 
        
        print("Map loaded from /Game/Carla/Maps/Town%02d" % args.town)
        world = client.load_world("/Game/Carla/Maps/Town%02d" % args.town) 
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library() 
        bp1 = blueprint_library.filter('vehicle')[0]
        bp2 = blueprint_library.filter('vehicle')[0]
        color1 = random.choice(bp1.get_attribute('color').recommended_values)
        color2 = bp2.get_attribute('color').recommended_values[1]
        bp1.set_attribute('color', color1)
        bp2.set_attribute('color', color2)
   
        # Set the starting position 
        transform1 = world.get_map().get_spawn_points()[args.spawn_point]
        if args.town == 3 and args.spawn_point == 0:
            transform1.location.x += 0.8
        vehicle_lead = world.spawn_actor(bp1, transform1)
        
        transform2= set_spawn_point(args.town, args.spawn_point, transform1, SPAWN_POINTS)   
        vehicle_follow = world.spawn_actor(bp2, transform2)

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
        spectator.set_transform(carla.Transform(transform1.location +carla.Location(z=400)+carla.Location(x=-80)+carla.Location(y=100),carla.Rotation(pitch=-90)))
         
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
        camera_rgb2.listen(lambda image: rgb2_list.append(image))
         
        
        state_i = np.array([[transform1.location.x,transform1.location.y, transform1.rotation.yaw, 0]]) 
        ref_i = np.array([[transform2.location.x,transform2.location.y, transform2.rotation.yaw,0]])  
        u_i = np.array([[0,0]]) 
        
         ##### SIMULATOR DISPLAY ######### 

        # Total Figure
        fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
        gs = gridspec.GridSpec(8,8)

        # Elevator plot settings.
        ax = fig.add_subplot(gs[:8, :8]) 
        plt.xlim(-500, 500)
        ax.set_ylim([-500, 500])
        plt.xticks(np.arange(-500, 501, step=50))
        plt.yticks(np.arange(-500, 501, step=50))
        plt.title('Trajectory Overview')

        
        # Main plot info. 
        patch_car = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', fill=True)
        patch_goal = mpatches.Rectangle((0, 0), CAR_WIDTH, CAR_LENGTH, fc='k', ls='dashdot', fill=True)

        ax.add_patch(patch_car)
        ax.add_patch(patch_goal)
        predict, = ax.plot([], [], 'r--', linewidth = 1) 
        print('Create vehicle_follow and vehicle_lead in the figure.')
        
        # plot text
        text_pt = plt.text(-450,450, '', fontsize=8)
        text_vlead = plt.text(-450,420, '', fontsize=8)
        text_vfollow = plt.text(-450,390, '', fontsize=8)
        text_dis = plt.text(-450,360, '', fontsize = 8) 
        text_ratio = plt.text(-450,330, '', fontsize=8) 
        text_list = [text_pt, text_vlead, text_vfollow, text_dis, text_ratio]
         
        # define the models 
        mynet_CNN= MyModel_CNN2()
        if args.model == 'depth': 
            mynet_CNN.load_state_dict(torch.load(MODEL_PATH_DEP, map_location='cpu'))
        if args.model == 'stereo': 
            mynet_CNN.load_state_dict(torch.load(MODEL_PATH_STE, map_location='cpu'))
        print('CNN model:', mynet_CNN)
        
        mynet_MLP = MyModel_MLP_transform(neurons = [256, 1024, 256])  
        mynet_MLP.load_state_dict(torch.load(MODEL_PATH_MLP, map_location='cpu')) 
        print('MLP model:', mynet_MLP)
        
        for frame in range(800):
            print('frame %s' % frame)
            # Do tick
            world.tick()
            vehicle_lead.set_autopilot(True) 
            # Always have the traffic light on green
            if vehicle_lead.is_at_traffic_light():
                traffic_light = vehicle_lead.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
            
            # add ramdom impulse as disturbances
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
            print('Target Vehicle')
            print("Location: (x,y): ({},{})".format(vehicle_lead.get_location().x,vehicle_lead.get_location().y ))
            print("Throttle: {}, Steering Angle: {}".format(vehicle_lead.get_control().throttle, vehicle_lead.get_control().steer))
            print('Ego Vehicle')
            print("Location: (x,y): ({},{})".format(vehicle_follow.get_location().x,vehicle_follow.get_location().y))
            print("Throttle: {}, Steering Angle: {}".format(vehicle_follow.get_control().throttle, vehicle_follow.get_control().steer))
             
            
            v_lead_vec = np.array([vehicle_lead.get_velocity().x, vehicle_lead.get_velocity().y,
                                    vehicle_lead.get_velocity().z])
            v_t_lead = np.linalg.norm(v_lead_vec)
            psi_t_lead = vehicle_lead.get_transform().rotation.yaw * np.pi / 180

            v_fol_vec = np.array([vehicle_follow.get_velocity().x, vehicle_follow.get_velocity().y, vehicle_follow.get_velocity().z])
            v_t_fol = np.linalg.norm(v_fol_vec)
            psi_t_fol = vehicle_follow.get_transform().rotation.yaw * np.pi / 180
             
            # input preparation
            img = np.array(rgb2_list[-1].raw_data)
            img_reshaped = np.reshape(img, (rgb2_list[-1].height, rgb2_list[-1].width, 4)) 
            raw_img = img_reshaped[:,:,:3].reshape((rgb2_list[-1].height, rgb2_list[-1].width, 3)) 
            
            crop_img, resize_img = center_resize(raw_img, 250, 400)
            view_img = cv2.resize(crop_img, (400,400), interpolation = cv2.INTER_AREA) 
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_input = transform(resize_img/255) 
            
            model_input = [torch.tensor(img_input), torch.tensor(v_t_lead), torch.tensor(v_t_fol)] 
             
            
            # model inference for CNN 
            mynet_CNN.eval()
            output_CNN = mynet_CNN(model_input) 
            input_MLP = torch.cat((output_CNN[0], torch.tensor([v_t_lead,v_t_fol])), axis = 0) 
            # model inference for MLP 
            mynet_MLP.eval()   
            input_MLP = torch.unsqueeze(input_MLP, 0)
            output_MLP = mynet_MLP(input_MLP.to(torch.float32)) 
            throttle, str_angle = output_MLP[0].detach().numpy()
            throttle = float(throttle)  
            str_angle = float(str_angle)    
            u_i = np.append(u_i, np.array([(throttle, str_angle)]), axis=0)   
            vehicle_follow.apply_control(carla.VehicleControl(throttle=throttle, steer=str_angle))
            
            ###### Visualization ######
            update_plot(frame, patch_car, patch_goal, vehicle_lead, vehicle_follow, text_list, frame) 
            fig.canvas.draw()  
            img_loc = get_img_from_fig(fig, dpi=180) 
            if frame > 11: 
                visualize_image(rgb1_list[-1], view_img, img_loc)   
    
    finally:

        print('destroying actors')
        camera_rgb.destroy()
        camera_rgb2.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
