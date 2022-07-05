import numpy as np
#from sim.sim2d import sim_run


class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.2

        
    def plant_ref(self, reference, u_ref):  
        a_ref = u_ref[0]
        beta_ref = u_ref[1]
        
        x_ref = reference[0]  
        y_ref = reference[1]
        psi_ref = reference[2]
        v_ref = reference[3]
        
        L = 2.5
        v_ref_t1 = v_ref + a_ref * self.dt - v_ref/25.0
        
        x_ref += self.dt * v_ref * np.cos(psi_ref)  
        y_ref += self.dt * v_ref * np.sin(psi_ref) 
        psi_ref += self.dt * v_ref * np.tan(beta_ref)/L
      
        return [x_ref, y_ref, psi_ref, v_ref_t1]
    

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3] 
        
        beta = steering
        a_t = pedal
        L = 2.5
        v_t_1 = v_t + a_t * dt - v_t/25.0
        
        x_dot = v_t * np.cos(psi_t)
        y_dot = v_t * np.sin(psi_t)
        psi_dot = v_t * np.tan(beta)/L
        
        x_t += x_dot * dt
        y_t += y_dot * dt
        psi_t += psi_dot * dt 
         
        
        return [x_t, y_t, psi_t, v_t_1]  

    def cost_function(self, u, *args):
        # state & ref: [x_t, y_t, psi_t, v_t]
        state = args[0]
        ref = args[1] 
        cost = 0.0 
        L = 3        
        pos_ref = np.array([ref[0], ref[1]])[None,:]
        #print(pos_ref.shape)
        
        for k in range(0, self.horizon):
            ts = [0,1]
            psi_ref = ref[3]
            psi_ego = state[3]
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])
            pos_current = np.array([state[0], state[1]])[None, :]
            
            car_dis = L
            
            dis_x = car_dis * (np.cos(psi_ego)*0.3 + np.cos(psi_ref)*0.8)
            dis_y = car_dis * (np.sin(psi_ego)*0.3 + np.sin(psi_ref)*0.8)

            cost += abs(ref[0] - dis_x - state[0])
            cost += abs(ref[1] - dis_y - state[1])
            
            # angle cost
            cost += abs(ref[2] - state[2])**2 *1000
            #cost += abs(ref[2] - np.arcsin(pos_head_y / pos_diff_norm)) 
            # acceleration cost
            # cost += (ref[3] - state[3])**2
            # steering angle cost
            #cost += u[k*2+1]**2*self.dt
             
        return cost

#sim_run(options, ModelPredictiveControl)
