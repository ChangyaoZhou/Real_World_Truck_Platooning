import numpy as np
from sim.sim2d_carla import sim_run


# Simulator options.
options = {}
options['FIG_SIZE'] = [16,10]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.1

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0, 2]
        self.reference2 = None #[10, 2, 3.14/2]
        self.car_len = 3.705

    def plant_model(self, prev_state, dt, pedal, steering, t):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3] 
        
        beta = steering * 1.62
        
        if t<=23:
            a_t = 0.0
            v_t_1 = v_t 
        elif t<= 40:
            a_t = pedal * 11.9
            v_t_1 = v_t + a_t * dt - v_t/11.65
        elif t>=41 and t<=44:
            v_t_1 = v_t 
        else: 
            a_t = pedal * 3.75
            v_t_1 = v_t + a_t * dt - v_t/36.95
        #print('v_t',v_t)
        #print('a_t', a_t)
        #print('v_t_1',v_t_1)
        x_dot = v_t * np.cos(psi_t)
        y_dot = v_t * np.sin(psi_t)
        psi_dot = v_t * np.tan(beta)/self.car_len
        
        x_t += x_dot * dt
        y_t += y_dot * dt
        
        psi_t += psi_dot * dt 
        
        
        return [x_t, y_t, psi_t, v_t_1]  

    def cost_function(self, u, *args):
        # state & ref: [x_t, y_t, psi_t, v_t]
        state = args[0]
        ref = args[1]
        t = args[2]
        cost = 0.0       
         
        #print(pos_ref.shape)
        
        for k in range(0, self.horizon):
            #if k > 0:
                #ref = self.plant_model(ref, self.dt, u[k*2], u[k*2+1],t)
                
            x_ref = ref[k][0] 
            y_ref = ref[k][1]
            psi_ref = ref[k][2]
            velocity_ref = ref[k][3] 
            psi_ego = state[2] 
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1],t+k)
             
            
            car_dis = (velocity_ref / 5 + 1) *self.car_len
            #car_dis = 2* self.car_len
            #print(car_dis)
            
            dis_x = car_dis * (np.cos(psi_ego)*0.7 + np.cos(psi_ref)*0.3) 
            dis_y = car_dis * (np.sin(psi_ego)*0.7 + np.sin(psi_ref)*0.3) 

            cost += abs(x_ref - dis_x - state[0]) 
            cost += abs(y_ref - dis_y - state[1]) 
          
            
            # angle cost
            current_psi = (state[2] + 2*np.pi)%(np.pi*2)
            current_psi_to_pi = abs(current_psi - np.pi)
            current_psi_ref_to_pi = abs(psi_ref - np.pi)
            cost += abs(current_psi_ref_to_pi - current_psi_to_pi)**2
            #cost += abs(u[k*2+1])
            
            #cost += abs(ref[2] - np.arcsin(pos_head_y / pos_diff_norm)) 
            # acceleration cost
            # cost += (ref[3] - state[3])**2
            # steering angle cost
            #cost += u[k*2+1]**2*self.dt
             
        return cost

sim_run(options, ModelPredictiveControl)
