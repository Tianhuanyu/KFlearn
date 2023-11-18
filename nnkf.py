
import torch
from SystemModel import SystemModel
import torch.nn as nn
import torch.nn.functional as func


class ESKF_Torch(torch.nn.Module):
    def __init__(self, 
                 system_model:SystemModel,
                 initial_state:torch.tensor,
                 initial_covariance: torch.tensor,
                 args):
        self.system_model = system_model
        self.state, self.error_state, self.covariance = self.system_model.initsetup(initial_state, initial_covariance)
        super().__init__()
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.args = args

    
    def predict(self,control_vector):
        F = self.system_model._compute_state_transition_matrix(self._dt)
        B = self.system_model._compute_control_matrix(self._dt)

        Q = self.init_Q

        # print("control_vector = ",control_vector)
        # raise ValueError("123")
        # self.error_state = self.system_model.reset_error_state(self.state, self._dt,control_vector)

        self.error_state_prior = F @ self.error_state + B @ control_vector    #self._propagate_state(F, gyro_measurement)
        self.predict_state = self.system_model._state_injection(self._dt,self.state, self.error_state_prior).to(self.device)
        self.covariance = self.system_model._propagate_covariance(F, Q, self.covariance).to(self.device)

    def update(self, measurement):
        Hx = self.system_model.Hx_fun()
        Xdx = self.system_model.Xdx_fun(self.state)

        H = Hx @ Xdx
        R = self.init_S
        # print("H = ",H.size())
        # print("self.covariance = ",self.covariance.size())

        cov = H @ self.covariance @H.transpose(1,2) +R

        # print("torch.inverse(cov)= ",torch.inverse(cov).size())
        # cov_inv = torch.zeros_like(cov)

        # for i in range(cov.shape[0]):
        #     cov_inv[i,:,:] = torch.inverse(cov[i,:,:])
        K = self.covariance @ H.transpose(1,2) @ torch.inverse(cov)
        # print("self.error_state",self.error_state.size())

        t = measurement-self.predict_state

        min_mag = torch.zeros_like(min_mag).to(self.device)

        max_mag = torch.tensor([
                0.005, 0.005, 0.005, 0.1, 0.1, 0.1, 0.1
            ]).unsqueeze(0).unsqueeze(2).repeat(self.args.n_batch,1,1).to(self.device)

        sign = t.sign()
        t = t.abs_().clamp_(min_mag, max_mag)
        t *= sign

        self.error_state = K@ t   # This is defination of obs_diff
        # print("self.error_state",self.error_state.size())
        eYe = torch.eye(self.error_state.shape[1]).unsqueeze(0).repeat(self.system_model.n_batch,1,1).to(self.device)

        # print("eYe",eYe.size())
        # print("H",H.size())
        self.covariance = (eYe - K @ H) @ self.covariance

        self.state = self.system_model._state_injection(self._dt,self.predict_state, self.error_state)
        
        #reset
        self.prvious_error_state = self.error_state
        self.error_state = torch.zeros_like(self.error_state)
        return self.state
    

    def get_state(self):
        return self.state
    
    def reset_state(self, init_state):
        # self.state = init_state
        self.covariance = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)
        
        diag_matrix_P = torch.diag(torch.tensor([0.01]*3+
                            [0.002]*3, requires_grad=True)).unsqueeze(0).repeat(self.args.n_batch,1,1).to(self.device)
        diag_matrix_R = torch.diag(torch.tensor(
                [1000.0]*3+
                            [1000.0]*4, requires_grad=True)
            ).unsqueeze(0).repeat(self.args.n_batch,1,1).to(self.device)
        T_fitler = torch.tensor(0.01)
        self.reset_init_state(init_state, 
                                diag_matrix_P, 
                                diag_matrix_R, 
                                T_fitler,
                                self.args)

    def reset_init_state(self, state, system_noise_covariance, measurement_noise_covariance,dt,args):
        self.state = state
        self.NNBuild(system_noise_covariance, measurement_noise_covariance,dt,args)

    def NNBuild(self, system_noise_covariance, measurement_noise_covariance,dt,args):
        self.prior_Sigma = self.covariance
        self.init_Q = system_noise_covariance
        self.init_S = measurement_noise_covariance


        self._dt = dt
        self.batch_size = self.system_model.n_batch
        self.seq_len_input = 1
        self.predict_state = self.state

        self.prvious_error_state = self.error_state
        
        self.m = self.prior_Sigma.shape[1]
        self.n = self.init_S.shape[1]

        self.Output_dim_of_fnn = args.in_mult_KNet

    def forward(self, x):
        # x= x.to(self.device)
        twist = x[8:14].unsqueeze(0).permute(2, 1, 0)
        # twist = twist

        measurement = x[0:7].unsqueeze(0).permute(2,1,0)

        self.predict(
            control_vector = twist
        )

        state = self.update(measurement)

        return state





class KalmanNet(ESKF_Torch):
    def __init__(self, 
                 system_model:SystemModel,
                 initial_state:torch.tensor,
                 initial_covariance: torch.tensor,
                 args,
                 dt):
        super().__init__(system_model, initial_state, initial_covariance,args)


        
        diag_matrix_P = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)
        diag_matrix_R = torch.diag(torch.tensor(
            [0.0001]*3+
                        [0.01,0.01,0.01,0.01], requires_grad=True)).to(self.device)
        


        self.covariance = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)

        self.NNBuild(diag_matrix_P,
                     diag_matrix_R,
                     dt,
                     args
        )

        # self.init_hidden_KNet()
        


    def NNBuild(self, system_noise_covariance, measurement_noise_covariance,dt,args):
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.prior_Sigma = self.covariance
        self.init_Q = dt * system_noise_covariance

        # print("self.init_Q = ",self.init_Q.size())
        self.init_S = measurement_noise_covariance
        self._dt = dt
        self.batch_size = args.n_batch
        self.seq_len_input = 1

        self.last_measurement = self.state


        self.prvious_error_state = torch.zeros_like(self.error_state)
        
        self.m = self.prior_Sigma.shape[1]
        self.n = measurement_noise_covariance.shape[0]
        self.pos_n = 3

        self.Output_dim_of_fnn = args.in_mult_KNet

        self.InitKGainNet()
        # self.init_hidden_KNet()

    # def NNreset(self, system_noise_covariance, measurement_noise_covariance)


    def InitKGainNet(self):
        # print("self.m = ",self.m)
        # print("self.Output_dim_of_fnn = ",self.Output_dim_of_fnn)
        # GRU to track Q
        self.d_input_Q = self.m * self.Output_dim_of_fnn
        self.d_hidden_Q = self.m **2
        self.GRU_Q = nn.GRU(self.d_input_Q, 
                            self.d_hidden_Q
                            ).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * self.Output_dim_of_fnn
        self.d_hidden_Sigma = self.m **2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, 
                                self.d_hidden_Sigma
                                ).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * self.Output_dim_of_fnn
        self.d_hidden_S = self.n **2
        self.GRU_S = nn.GRU(self.d_input_S, 
                            self.d_hidden_S
                            ).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.pos_n * self.pos_n
        self.d_hidden_FC2 = self.d_input_FC2 * self.Output_dim_of_fnn
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
                ).to(self.device)

        self.d_input_FC21 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC21 = (self.n - self.pos_n) * (self.m- self.pos_n)
        self.d_hidden_FC21 = self.d_input_FC2 * self.Output_dim_of_fnn
        self.FC21 = nn.Sequential(
                nn.Linear(self.d_input_FC21, self.d_hidden_FC21),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC21, self.d_output_FC21)
                ).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2 + self.d_output_FC21
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()
                ).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * self.Output_dim_of_fnn
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * self.Output_dim_of_fnn
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * self.Output_dim_of_fnn
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)
        
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        # def expand_dim(x):
        #     expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
        #     expanded[0, :, :] = x
        #     return expanded

        # obs_diff = expand_dim(obs_diff)
        # obs_innov_diff = expand_dim(obs_innov_diff)
        # fw_evol_diff = expand_dim(fw_evol_diff)
        # fw_update_diff = expand_dim(fw_update_diff)
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        # print("fw_evol_diff = ",fw_evol_diff.size())
        in_FC5 = fw_evol_diff

        # print("self.m = ",self.m)
        # print("in_FC5 = ",in_FC5.size())
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5

        # print("in_Q = ",in_Q.size())
        # print("self.h_Q = ",self.h_Q.size())
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)
        # out_Q = torch.diag_embed(out_Q.squeeze(-1))


        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)

        # print("in_Sigma = ",in_Sigma.size())
        # print("self.h_Sigma = ",self.h_Sigma.size())
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
        # out_Sigma = torch.diag_embed(out_Sigma.squeeze(-1))

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        # out_S = torch.diag_embed(out_Sigma.squeeze(-1))


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC21 = self.FC21(in_FC2)

        # out_FC2 = torch.cat((_FC2, _FC21), dim = 2)
        # out_FC21 = torch.cat((torch.zeros_like(_FC2).to(self.device),  _FC21), dim = 2)

        # out_FC2 = torch.cat((out_FC2, out_FC21), dim = 1)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2, out_FC21), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4
        # raise ValueError("Run to  here")

        return out_FC2, out_FC21
    ###############
    ### Forward ###
    ###############
    def forward(self, x):
        x = x.to(self.device)
        return self.KNet_step(x)
    

    def KNet_step(self, x):

        # Compute Priors
        twist = x[8:14].unsqueeze(0).permute(2, 1, 0)
        # self.predict(
        #     control_vector = twist
        # )


        F = self.system_model._compute_state_transition_matrix(self._dt)
        B = self.system_model._compute_control_matrix(self._dt)

        # Q = self.init_Q
        # Compute Kalman Gain
        measurement = x[0:7].unsqueeze(0).permute(2,1,0)

        self.error_state_prior = F @ self.error_state + B @ twist    #self._propagate_state(F, gyro_measurement)
        # self.predict_state = self.system_model._state_injection(self._dt,self.state, self.error_state_prior)
        # print("B @ twist = ",B @ twist)
        self.predict_state = self.system_model._state_injection(self._dt,self.state, self.error_state_prior)

        # print("measurement = ", measurement)
        
        self.step_KGain_est(measurement)
        # Innovation
        # dy = measurement - self.predict_state # [batch_size, n, 1]

        dy = measurement-self.predict_state


           # This is defination of obs_diff

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)

        # min_mag = torch.zeros_like(INOV).to(self.device)

        # max_mag = torch.tensor([
        #         0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        #     ]).unsqueeze(0).unsqueeze(2).repeat(self.args.n_batch,1,1).to(self.device)*100.0

        # sign = INOV.sign()
        # INOV = INOV.abs_().clamp_(min_mag, max_mag)
        # INOV =INOV* sign


        # print("INOV = ",INOV)
        # raise ValueError("Run to here")

        self.state = self.system_model._state_injection(self._dt,self.state, INOV)
        

        # print("INOV = ",INOV)
        # print("self.state = ",self.state)
        # raise ValueError("Run to here")
        #reset
        self.last_measurement = measurement
        self.prvious_error_state = self.error_state
        self.error_state = torch.zeros_like(self.error_state)
        
        return self.state


    
    def step_KGain_est(self, state):
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(state,2) - torch.squeeze(self.state,2) 
        obs_innov_diff = torch.squeeze(state,2) - torch.squeeze(self.predict_state,2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.error_state,2) - torch.squeeze(self.error_state_prior,2)
        fw_update_diff = torch.squeeze(self.error_state,2) - torch.squeeze(self.prvious_error_state,2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # print("obs_diff size = ",obs_diff.shape)

        # Kalman Gain Network Step
        KG1, KG2 = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        KG1 = torch.reshape(KG1, (self.batch_size, self.pos_n, self.pos_n))
        KG2 = torch.reshape(KG2, (self.batch_size, self.m-self.pos_n, self.n - self.pos_n))

        # print("KG1 size =", KG1.shape)
        # print("KG2 size =", KG2.shape)

        _KG1 = torch.cat([KG1,  torch.zeros_like(KG2).to(self.device)], dim=2)
        _KG2 = torch.cat([torch.zeros_like(KG1).to(self.device), KG2], dim=2)
        self.KGain = torch.cat([_KG1, _KG1], dim=1) #torch.reshape(KG, (self.batch_size, self.m, self.n))


    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.init_S.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.init_Q.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion


    def reset_state(self, init_state):
        self.state = init_state
        self.covariance = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)
        self.init_hidden_KNet()


    # def reset_state_hidden_(self, inistate)

        # print("size hQ= ",self.init_Q.reshape(1,1, -1).shape)
        


    
# error state = dp_rt, dth_rw, dth_rt, dp_wrt, dw_tt
# state = p_rt, q_rw, q_rt, p_wrt, w_tt
# state = p_rt, q_rt
# 相当于从robot to robot_t+1/  robot to trocar;
# 此时robot t+1 to trocar 就是 p_rt t+1;
# 在这一过程中 Omega t and v t 由机器人给出，相对坐标系是机器人EE。
# 除此之外， reprojection error 可以作为卡尔曼滤波器方差输入， 考虑神经网络（VAE+LSTM), 结合卡尔曼滤波器联合训练。
# Input 输入已知， 关键的问题是ground truth。 是否可以考虑机器人轨迹，类似重投影误差。

# 1. ICP 方法，进行配准（数据对齐）
# 2. 将机器人被对准后的数据作为Ground Truth
# 3. 假设Trocar一直没有移动
# 4. 通过卡尔曼滤波器生成实际轨迹，进行联合训练。




class KalmanNetOrigin(ESKF_Torch):
    def __init__(self, 
                 system_model:SystemModel,
                 initial_state:torch.tensor,
                 initial_covariance: torch.tensor,
                 args,
                 dt):
        super().__init__(system_model, initial_state, initial_covariance,args)


        
        diag_matrix_P = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)
        diag_matrix_R = torch.diag(torch.tensor(
            [0.0001]*3+
                        [0.01,0.01,0.01,0.01], requires_grad=True)).to(self.device)
        


        self.covariance = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)

        self.NNBuild(diag_matrix_P,
                     diag_matrix_R,
                     dt,
                     args
        )

        # self.init_hidden_KNet()
        


    def NNBuild(self, system_noise_covariance, measurement_noise_covariance,dt,args):
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.prior_Sigma = self.covariance
        self.init_Q = dt * system_noise_covariance

        # print("self.init_Q = ",self.init_Q.size())
        self.init_S = measurement_noise_covariance
        self._dt = dt
        self.batch_size = args.n_batch
        self.seq_len_input = 1

        self.last_measurement = self.state


        self.prvious_error_state = torch.zeros_like(self.error_state)
        
        self.m = self.prior_Sigma.shape[1]
        self.n = measurement_noise_covariance.shape[0]

        self.Output_dim_of_fnn = args.in_mult_KNet

        self.InitKGainNet()
        # self.init_hidden_KNet()

    # def NNreset(self, system_noise_covariance, measurement_noise_covariance)


    def InitKGainNet(self):
        # print("self.m = ",self.m)
        # print("self.Output_dim_of_fnn = ",self.Output_dim_of_fnn)
        # GRU to track Q
        self.d_input_Q = self.m * self.Output_dim_of_fnn
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, 
                            self.d_hidden_Q
                            ).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * self.Output_dim_of_fnn
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, 
                                self.d_hidden_Sigma
                                ).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * self.Output_dim_of_fnn
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, 
                            self.d_hidden_S
                            ).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * self.Output_dim_of_fnn
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
                nn.Dropout(p=0.5)
                ).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()
                ).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * self.Output_dim_of_fnn
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * self.Output_dim_of_fnn
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * self.Output_dim_of_fnn
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.Dropout(p=0.5),
                nn.ReLU()).to(self.device)
        
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        # def expand_dim(x):
        #     expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
        #     expanded[0, :, :] = x
        #     return expanded

        # obs_diff = expand_dim(obs_diff)
        # obs_innov_diff = expand_dim(obs_innov_diff)
        # fw_evol_diff = expand_dim(fw_evol_diff)
        # fw_update_diff = expand_dim(fw_update_diff)
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        # print("fw_evol_diff = ",fw_evol_diff.size())
        in_FC5 = fw_evol_diff

        # print("self.m = ",self.m)
        # print("in_FC5 = ",in_FC5.size())
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5

        # print("in_Q = ",in_Q.size())
        # print("self.h_Q = ",self.h_Q.size())
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)


        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)

        # print("in_Sigma = ",in_Sigma.size())
        # print("self.h_Sigma = ",self.h_Sigma.size())
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4
        # raise ValueError("Run to  here")

        return out_FC2
    ###############
    ### Forward ###
    ###############
    def forward(self, x):
        x = x.to(self.device)
        return self.KNet_step(x)
    

    def KNet_step(self, x):

        # Compute Priors
        twist = x[8:14].unsqueeze(0).permute(2, 1, 0)
        # self.predict(
        #     control_vector = twist
        # )


        F = self.system_model._compute_state_transition_matrix(self._dt)
        B = self.system_model._compute_control_matrix(self._dt)

        # Q = self.init_Q
        # Compute Kalman Gain
        measurement = x[0:7].unsqueeze(0).permute(2,1,0)

        self.error_state_prior = F @ self.error_state + B @ twist    #self._propagate_state(F, gyro_measurement)
        # self.predict_state = self.system_model._state_injection(self._dt,self.state, self.error_state_prior)
        # print("B @ twist = ",B @ twist)
        self.predict_state = self.system_model._state_injection(self._dt,self.state, self.error_state_prior)

        # print("measurement = ", measurement)
        
        self.step_KGain_est(measurement)
        # Innovation
        # dy = measurement - self.predict_state # [batch_size, n, 1]

        dy = measurement-self.predict_state


           # This is defination of obs_diff

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)

        min_mag = torch.zeros_like(INOV).to(self.device)

        max_mag = torch.tensor([
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ]).unsqueeze(0).unsqueeze(2).repeat(self.args.n_batch,1,1).to(self.device)*100.0

        sign = INOV.sign()
        INOV = INOV.abs_().clamp_(min_mag, max_mag)
        INOV =INOV* sign
        # print("INOV = ",INOV)
        # raise ValueError("Run to here")

        self.state = self.system_model._state_injection(self._dt,self.state, INOV)
        

        # print("INOV = ",INOV)
        # print("self.state = ",self.state)
        # raise ValueError("Run to here")
        #reset
        self.last_measurement = measurement
        self.prvious_error_state = self.error_state
        self.error_state = torch.zeros_like(self.error_state)
        
        return self.state


    
    def step_KGain_est(self, state):
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(state,2) - torch.squeeze(self.state,2) 
        obs_innov_diff = torch.squeeze(state,2) - torch.squeeze(self.predict_state,2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.error_state,2) - torch.squeeze(self.error_state_prior,2)
        fw_update_diff = torch.squeeze(self.error_state,2) - torch.squeeze(self.prvious_error_state,2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # print("obs_diff size = ",obs_diff.shape)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))


    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.init_S.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.init_Q.reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1).to(self.device) # batch size expansion


    def reset_state(self, init_state):
        self.state = init_state
        self.covariance = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).to(self.device)
        self.init_hidden_KNet()