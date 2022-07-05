import numpy as np
from sim.sim2d_carla import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [16,10]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [-5.44615602e+00, -7.90549850e+01, 1.6057757493113909]  
        self.reference2 = None #[11, 2, 3* 3.14/2]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]       
        a_t = pedal
        car_length = 3.705369
        
        x_t_1 = x_t + v_t * dt * np.cos(psi_t)
        y_t_1 = y_t + v_t * dt * np.sin(psi_t)
        v_t_1 = v_t + a_t *dt #- v_t/25.0
        #psi_t_1 = psi_t + v_t * dt * np.tan(steering) / car_length
        psi_t_1 = psi_t + v_t * dt * steering / car_length

        return [x_t_1, y_t_1, psi_t_1, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]         #the current state of the target car
        last_ref = args[2]    #the last state of the target car
        psi_target_current = ref[2]
        car_length = 3.705369
        cost = 0.0
                
        for i in range(self.horizon):
            ref_x = ref[i][0]
            ref_y = ref[i][1]
            psi_target_current = ref[i][2]
            
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1])
            x_current = state[0]
            y_current = state[1]
            psi_current = state[2]
            v_current = state[3]
            
            ######cost for the position#####
            #get the total distance
            #if np.array_equal(ref, last_ref):   #if the target car stops, reduce the distance
                #print('stop')
             #   distance = 0.0 * car_length
            #else:
                #print('go')
            distance = 2.0 * car_length
                
            #get the x-/y-distance
            distance_x = distance * (np.cos(psi_current)*0.3 + np.cos(psi_target_current)*0.8)
            distance_y = distance * (np.sin(psi_current)*0.3 + np.sin(psi_target_current)*0.8)
            
            #compute the distance cost
            cost += (abs((ref_x - distance_x) - x_current)) #**2
            cost += (abs((ref_y - distance_y) - y_current)) #**2
            """
            #cost += (abs((ref[0] - distance * np.cos(psi_current)) - x_current)**2) 
            #cost += (abs((ref[1] - distance * np.sin(psi_current)) - y_current)**2) 
            #cost += (abs((ref[0] - distance * np.cos(psi_target_current)) - x_current)**2) 
            #cost += (abs((ref[1] - distance * np.sin(psi_target_current)) - y_current)**2) 
            """
            ######cost for the psi angle#####    

            if psi_target_current > np.pi:
                cost += ((psi_target_current - 2*np.pi) - psi_current)**2
            elif psi_target_current < -np.pi:
                cost += ((psi_target_current + 2*np.pi) - psi_current)**2
            else:
                cost += abs(psi_current - psi_target_current)**2 
                
            #####extra cost for reverse#####
            #if v_current < 0:
            #    cost+= abs(v_current) *100
            #print('cost', cost)
                    
        return cost

sim_run(options, ModelPredictiveControl)
