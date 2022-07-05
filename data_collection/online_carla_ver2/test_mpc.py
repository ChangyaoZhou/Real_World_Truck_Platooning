import numpy as np


class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 10
        self.dt = 0.1
        self.car_len = 3.705369

        
    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3] 
        
        beta = steering * 3.0
        a_t = pedal * 7.1
        
        x_dot = v_t * np.cos(psi_t)
        y_dot = v_t * np.sin(psi_t)
        psi_dot = v_t * beta/self.car_len
            
        x_t += x_dot * dt
        y_t += y_dot * dt
        psi_t += psi_dot * dt
        v_t_1 = v_t + a_t * dt - v_t/19.8
        
        return [x_t, y_t, psi_t, v_t_1] 
    
    def cost_function(self,u, *args):
        state = args[0]       #the current state of the ego vehicle        
        ref = args[1]         #the current state of the target vehicle
        ref_x = ref[0]
        ref_y = ref[1]
        psi_target_current = ref[2]
        ref_v = ref[3]
        
        cost = 0.0
                
        for i in range(self.horizon):              
            x_current = state[0]
            y_current = state[1]
            psi_current = state[2]
            v_current = state[3]
            ######cost for the position#####
            #get the total distance
            distance = (ref_v / 5 + 1) *self.car_len
                
            #get the x-/y-distance
            distance_x = distance * (np.cos(psi_current)*0.5 + np.cos(psi_target_current)*0.5)
            distance_y = distance * (np.sin(psi_current)*0.5 + np.sin(psi_target_current)*0.5)
            
            #compute the distance cost
            cost += (abs((ref_x - distance_x) - x_current)) **2
            cost += (abs((ref_y - distance_y) - y_current)) **2
            
            distance_norm = ((ref_x - x_current)**2 + (ref_y - y_current)**2)**0.5
            cost += abs(distance_norm - distance)**2

            ######cost for the psi angle#####    

            # angle cost
            current_psi = (psi_current + 2*np.pi)%(np.pi*2)
            current_psi_to_pi = abs(current_psi - np.pi)
            ref_psi = (psi_target_current + 2*np.pi)%(np.pi*2)
            current_psi_ref_to_pi = abs(ref_psi - np.pi)
            cost += abs(current_psi_to_pi - current_psi_ref_to_pi) #**2
            
            ######cost for velocity difference######
            cost+= abs(ref_v - v_current)*2
                        
            #####extra cost for extreme acceleration#####
            #if u[i*2] < 0.1 or u[i*2] > 0.9:
            #    cost += abs(u[i*2] - 0.5)
            
            #predict the position of the ego vehicle
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1])
            
            #predict the position of the target vehicle
            ref_x += ref_v * np.cos(psi_target_current)* self.dt
            ref_y += ref_v * np.sin(psi_target_current)* self.dt
                        
        return cost
    


