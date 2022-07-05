import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time

import sys
sys.path.append('./carla/reference2/')

from location import all_location
from steer import all_steer
from throttle import all_throttle
from yaw import all_yaw
from velocity import all_velocity
#from reference.velocity import all_velocity


def sim_run(options, MPC):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = MPC()

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)
    bounds = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[0, 1]] # pedal bound
        bounds += [[-0.8, 0.8]] # steering angle bound 
    
    # initial state  
    state_i = np.array([[222.252029, -364.109375, -0.005154280934995745, 0.0]])
    all_yaw_rad = all_yaw *np.pi / 180.0 
    #print(all_yaw_rad[0])
    all_yaw_rad = (all_yaw_rad + 2*np.pi)%(np.pi*2)
    #print(all_yaw_rad[0])
    ref_i = np.hstack((all_location[:, :2], all_yaw_rad[:, None], all_velocity[:, None]))
    #ref_i = np.hstack((all_location[:, :2], all_yaw_rad[:, None]))
    control_value = np.hstack((all_throttle[:, None], all_steer[:, None]))
    u_i = np.array([[0,0]])
    sim_total = 685
    predict_info = [state_i]
    ref = mpc.reference1
    ref_2 = mpc.reference2


    for i in range(0,sim_total):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        
        start_time = time.time()
         
         
        # Non-linear optimization.
        
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref_i[i:i+15], i),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        print('Step ' + str(i) + ' of ' + str(sim_total) + ' Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        #u[0] = all_throttle[i-1]
        #u[1] = all_steer[i-1]
        #y = mpc.plant_model(state_i[-1], mpc.dt, control_value[i-1][0], control_value[i-1][1], i) 
        if i <= 25 and i>=3:
            u[0] = 0.699
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1], i) 
        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1], i) 
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0) 
        
        u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)
        #u_i = np.append(u_i, np.array([(control_value[i-1][0], control_value[i-1][1])]), axis=0)
        print('pedal: {}, steering angle: {}'.format(u[0], u[1]))
        print('yaw', state_i[-1][2])
        print('ref_yaw', ref_i[i][2])
        print('velocity', state_i[-2][3])
        print('ref_velocity', ref_i[i][3])
        
    np.set_printoptions(threshold=np.inf)
    f1 = open("./carla/output/all_u.py",'a')
    f1.write(str(repr(u_i)))
    f1.close()
    f2 = open("./carla/output/ref_yaw.py",'a')
    f2.write(str(repr(all_yaw_rad)))
    f2.close()


    ###################
    # SIMULATOR DISPLAY

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(215, 400)
    plt.ylim(-370, -160)
    plt.xticks(np.arange(215, 400, step=10))
    plt.yticks(np.arange(-370, -160, step=10))
    plt.title('MPC 2D')
    plt.plot(ref_i[:,0],ref_i[:,1])

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)

    # Main plot info.
    car_width = 1.0
    patch_car = mpatches.Rectangle((0, 0), car_width, 3.7, fc='k', fill=False)
    patch_goal = mpatches.Rectangle((0, 0), car_width, 3.7, fc='b',
                                    ls='dashdot', fill=False)

    ax.add_patch(patch_car)
    ax.add_patch(patch_goal)
    predict, = ax.plot([], [], 'r--', linewidth = 1)

    # Car steering and throttle position.
    telem = [40,40]
    patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
    ax.add_patch(patch_wheel)
    wheel_1, = ax.plot([], [], 'k', linewidth = 3)
    wheel_2, = ax.plot([], [], 'k', linewidth = 3)
    wheel_3, = ax.plot([], [], 'k', linewidth = 3)
    throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],
                                'b', linewidth = 20, alpha = 0.4)
    throttle, = ax.plot([], [], 'k', linewidth = 20)
    brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],
                            'b', linewidth = 20, alpha = 0.2)
    brake, = ax.plot([], [], 'k', linewidth = 20)
    throttle_text = ax.text(telem[0], telem[1]-8, 'Forward', fontsize = 5,
                        horizontalalignment='center')
    brake_text = ax.text(telem[0]+3, telem[1]-8, 'Reverse', fontsize = 5,
                        horizontalalignment='center')

    # Obstacles
    if OBSTACLES:
        patch_obs = mpatches.Circle((mpc.x_obs, mpc.y_obs),0.5)
        ax.add_patch(patch_obs)

    # Shift xy, centered on rear of car to rear left corner of car.
    def car_patch_pos(x, y, psi):
        #return [x,y]
        x_new = x - np.sin(psi)*(car_width/2)
        y_new = y + np.cos(psi)*(car_width/2)
        return [x_new, y_new]

    def steering_wheel(wheel_angle):
        wheel_1.set_data([telem[0]-3, telem[0]-3+np.cos(wheel_angle)*2],
                         [telem[1], telem[1]+np.sin(wheel_angle)*2])
        wheel_2.set_data([telem[0]-3, telem[0]-3-np.cos(wheel_angle)*2],
                         [telem[1], telem[1]-np.sin(wheel_angle)*2])
        wheel_3.set_data([telem[0]-3, telem[0]-3+np.sin(wheel_angle)*2],
                         [telem[1], telem[1]-np.cos(wheel_angle)*2])

    def update_plot(num):
        # Car.
        patch_car.set_xy(car_patch_pos(state_i[num,0], state_i[num,1], state_i[num,2]))
        patch_car.angle = np.rad2deg(state_i[num,2])-90
        # Car wheels
        np.rad2deg(state_i[num,2])
        steering_wheel(u_i[num,1]*2)
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/5*4)])
        brake.set_data([telem[0]+3, telem[0]+3],
                        [telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/5*4)])

        # Goal.
        if (num <= 130 or ref_2 == None):
            patch_goal.set_xy(car_patch_pos(ref_i[num,0],ref_i[num,1],ref_i[num,2]))
            patch_goal.angle = np.rad2deg(ref_i[num,2])-90
        else:
            patch_goal.set_xy(car_patch_pos(ref_2[0],ref_2[1],ref_2[2]))
            patch_goal.angle = np.rad2deg(ref_2[2])-90
        
        

        #print(str(state_i[num,3]))
        predict.set_data(predict_info[num][:,0],predict_info[num][:,1])
        # Timer.
        #time_text.set_text(str(100-t[num]))
        '''if (state_i[num,0] > 5):
            plt.xlim(state_i[num,0]-10, state_i[num,0]+15)
            telem[0] = state_i[num,0]-9
        if (state_i[num,0] < 5 and state_i[num,0] > 1):
            plt.xlim(state_i[num,0]-15, state_i[num,0]+5)
            telem[0] = state_i[num,0]-14
        if (state_i[num,1] > 5):
            plt.ylim(state_i[num,1]-10, state_i[num,1]+15)
            telem[1] = state_i[num,1]+10
        if (state_i[num,1] < 5 and state_i[num,1] > 1):
            plt.ylim(state_i[num,1]-15, state_i[num,1]+5)
            telem[1] = state_i[num,1]'''

        return patch_car, time_text


    print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.
    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(state_i)), interval=100, repeat=True, blit=False)
    #car_ani.save('mpc-video.mp4')

    plt.show()
