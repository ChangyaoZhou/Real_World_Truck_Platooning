import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time

import sys
sys.path.append('./carla/reference/')

from location import all_location
from steer import all_steer
from throttle import all_throttle
from yaw import all_yaw


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
        bounds += [[-10, 10]]
        bounds += [[-0.8, 0.8]]
    
    #total steps
    sim_total = 180
    
    #reference state
    all_yaw_pi = all_yaw /180.0 *np.pi
    ref = mpc.reference1
    ref_2 = mpc.reference2
    #ref = ref_1
    ref_state_i = np.zeros((200,3))    
    ref_state_i[:,0:2] = all_location[:,0:2]
    ref_state_i[:,2] = all_yaw_pi
    
    #car state
    state_i = np.zeros((sim_total, 4))
    state_i[0] = np.array([[-5.44615602e+00, -8.40549850e+01, 1.6057757493113909, 0.0]])
    u_i = np.zeros((sim_total, 2))
    u_i[0] = np.array([[0,0]])
    
    predict_info = [state_i]
    
    print(ref_state_i.shape)
    for i in range(1,sim_total):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        start_time = time.time()        
                
        # Non-linear optimization.tolist()
        u_solution = minimize(mpc.cost_function, u, (state_i[i-1], ref_state_i[i-1:i+14], ref_state_i[i-2]),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        
        #update the reference position  
        print('x_location = ', ref_state_i[i,0])
        print('y_location = ', ref_state_i[i,1])
        print('p_current = ', ref_state_i[i,2])
       
        print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[i-1], mpc.dt, u[0], u[1])
        if (i > 130 and ref_2 != None):
            ref = ref_2
        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i[i] = np.array([y])
        u_i[i] = np.array([(u[0], u[1])])
        #u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)


    f1 = open("./carla/output/carla_state.py","a")
    f1.write(str(repr(state_i)))
    f2 = open("./carla/output/carla_u.py","a")
    f2.write(str(repr(u_i)))

    ###################
    # SIMULATOR DISPLAY

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-5, 15)
    ax.set_ylim([-5, 15])
    plt.xticks(np.arange(-100,100, step=2))
    plt.yticks(np.arange(-100,100, step=2))
    #plt.xlim(0, 20)
    #plt.ylim(0, 20)
    plt.title('MPC 2D')

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)

    # Main plot info.
    car_width = 1.8
    patch_car = mpatches.Rectangle((0, 0), car_width, 3.705369, fc='k', fill=False)
    patch_goal = mpatches.Rectangle((0, 0), car_width, 3.705369, fc='b',
                                    ls='dashdot', fill=False)

    ax.add_patch(patch_car)
    ax.add_patch(patch_goal)
    predict, = ax.plot([], [], 'r--', linewidth = 1)
    
    """
    # Car steering and throttle position.
    telem = [6,16]
    patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
    ax.add_patch(patch_wheel)
    wheel_1, = ax.plot([], [], 'k', linewidth = 3)
    wheel_2, = ax.plot([], [], 'k', linewidth = 3)
    wheel_3, = ax.plot([], [], 'k', linewidth = 3)
    #throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],
    #                            'b', linewidth = 20, alpha = 0.4)
    throttle, = ax.plot([], [], 'k', linewidth = 20)
    #brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],
    #                        'b', linewidth = 20, alpha = 0.2)
    brake, = ax.plot([], [], 'k', linewidth = 20)
    throttle_text = ax.text(telem[0], telem[1]-3, 'Forward', fontsize = 10,
                        horizontalalignment='center')
    brake_text = ax.text(telem[0]+3, telem[1]-3, 'Reverse', fontsize = 10,
                        horizontalalignment='center')
    """
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
        brake_text.set_x(telem[0]+3)
        brake_text.set_y(telem[1]-3)
        throttle_text.set_x(telem[0])
        throttle_text.set_y(telem[1]-3)
        patch_wheel.center = telem[0]-3, telem[1]



    def update_plot(num):
        # Car.
        patch_car.set_xy(car_patch_pos(state_i[num,0], state_i[num,1], state_i[num,2]))
        patch_car.angle = np.rad2deg(state_i[num,2])-90
        """
        # Car wheels
        np.rad2deg(state_i[num,2])
        steering_wheel(u_i[num,1]*2)
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/1*4)])
        brake.set_data([telem[0]+3, telem[0]+3],
                        [telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/1*4)])
        """
        # Goal.
        if (num <= 130 or ref_2 == None):
            patch_goal.set_xy(car_patch_pos(ref_state_i[num,0],ref_state_i[num,1],ref_state_i[num,2]))
            patch_goal.angle = np.rad2deg(ref_state_i[num,2])-90
        else:
            patch_goal.set_xy(car_patch_pos(ref_2[0],ref_2[1],ref_2[2]))
            patch_goal.angle = np.rad2deg(ref_2[2])-90

        #print(str(state_i[num,3]))
        predict.set_data(predict_info[num][:,0],predict_info[num][:,1])
        # Timer.
        #time_text.set_text(str(100-t[num]))
        #if (state_i[num,0] > 5):
        #plt.xlim(state_i[num,0]-5, state_i[num,0]+15)
        #plt.ylim(state_i[num,1]-5, state_i[num,1]+15)
        #telem[0] = state_i[num,0]+1
        #telem[1] = state_i[num,1]+11

        return patch_car, time_text


    print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.
    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(state_i)), interval=100, repeat=True, blit=False)
    #car_ani.save('mpc-video.mp4')
    
    print('ref_state_i', ref_state_i.shape)
    print('state_i', state_i.shape)
    print('u_i', u_i.shape)

    plt.show()
