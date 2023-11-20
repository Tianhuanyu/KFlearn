import torch
from abc import abstractmethod


class SystemModel:
    def __init__(self,state_size:int, 
                 error_state_size:int,
                 args):
        self.state_size = state_size
        self.covariance_size = state_size
        self.error_state_size = error_state_size
        self.n_batch = args.n_batch

    def initsetup(self, state=None, cov=None):
        _error_state = torch.zeros(self.n_batch, self.error_state_size,1
                                   )
        _state = torch.zeros(self.n_batch, self.state_size,1 
                             )
        _covariance = torch.eye(self.error_state_size)
        _covariance = _covariance.unsqueeze(0)
        _covariance = _covariance.repeat(self.n_batch,1,1)
        
        
        # .repeat(
        #      self.n_batch,2
        # )
                                
        

        if(state is not None and cov is not None):
            print("state.size()  {0}==_state.size() {1}".format(state.size(),_state.size()))
            print("cov.size()  {0}==_covariance.size() {1}".format(cov.size(),_covariance.size()))
            if(state.size()==_state.size() and 
               cov.size()==_covariance.size()):
                _state = state
                _covariance = cov
            else:
                raise ValueError("Wrong tensor size in state and covariance")
        
        return _state, _error_state, _covariance
    
    @abstractmethod
    def _compute_state_transition_matrix(self, dt):
        pass

    @abstractmethod
    def _propagate_covariance(self, F, Q):
        pass

    @abstractmethod
    def Hx_fun(self):
        pass

    @abstractmethod
    def Xdx_fun(self):
        pass

    @abstractmethod
    def _state_injection(self, dt, state, error_state):
        pass

    @abstractmethod
    def _compute_control_matrix(self,dt):
        pass





class RobotSensorFusion(SystemModel):
    def __init__(self, state_size:int, 
                 error_state_size:int,
                 args):
        super().__init__(state_size, error_state_size,args)  # 调用父类的初始化函数
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def quaternion_normalize(self,qs):
        o = torch.zeros_like(qs)
        # print("qs = ",qs.size())
        for i,q in enumerate(qs):
            w = q[0][0]
            x = q[1][0]
            y = q[2][0]
            z = q[3][0]
            s = w*w + x*x+y*y+z*z
            o[i,:,:] = torch.tensor([w/s, x/s, y/s, z/s], requires_grad=True).unsqueeze(1)
        return o.to(self.device)
    
    
    def quaternion_to_left_matrix(self, qs):
        Q_th = torch.zeros([qs.shape[0], qs.shape[1], 3])
        for i,q in enumerate(qs):
            w = q[0]
            x = q[1]
            y = q[2]
            z = q[3] #, x, y, z = q
            # print(w.size())
            # print(x.size())
            # print(y.size())
            # print(z.size())
            Q_th[i,:,:] = torch.tensor(0.5) * torch.tensor([
                [-x, -y, -z],
                [w, -z, y],
                [z, w, -x],
                [-y, x, w]
            ],  requires_grad=True)
        return Q_th.to(self.device)
    
    def initsetup(self, 
                  initial_state:torch.tensor, 
                  initial_covariance:torch.tensor):

        initial_state = initial_state.to(self.device)
        initial_covariance = initial_covariance.to(self.device)

        state, error_state, covariance = super().initsetup(initial_state, initial_covariance)

        # print("state",state.size())

        state[:,self.state_size-4:self.state_size,:] = self.quaternion_normalize(
            state[:,self.state_size-4:self.state_size,:]
            )

        return state.to(self.device), error_state.to(self.device), covariance.to(self.device)
    
    def _compute_state_transition_matrix(self, dt):
        # compute matrix F for error state
        l = int(self.error_state_size/2)
        eye_3 = torch.eye(l).to(self.device)
        zeros_3 = torch.zeros(l, l).to(self.device)

        # 使用torch.cat进行水平和垂直的连接
        F_1 = torch.cat([eye_3, zeros_3], dim=1)
        F_2 = torch.cat([zeros_3, eye_3], dim=1)

        F = torch.cat([F_1, F_2], dim=0).unsqueeze(0)

        # print("F = ",F.size())
        F = F.repeat(self.n_batch,1,1 )
        return F.to(self.device)

    def _propagate_covariance(self, F, Q, covariance):
        # print(" F.transpose(1, 2) ={0} ", F.transpose(1, 2))
        covariance = F@covariance @F.transpose(1,2) + Q
        
        # torch.mm(
        #     F, torch.mm(covariance, F.transpose(1, 2))
        #     ) + Q
        return covariance.to(self.device)
    
    def _compute_control_matrix(self, dt, state):

        R = self.quaternion_to_rotation_matrix(state[:,3:,:]).to(self.device)

        S = torch.eye(6).to(self.device)
        S = S.repeat(self.n_batch, 1,1)

        S[:,:3,:3] = R


        return S.to(self.device)*dt
    
    def Hx_fun(self):
        Hx = torch.eye(
            self.state_size
            ).unsqueeze(0) # directly measurement for all states
        Hx = Hx.repeat(self.n_batch,1,1)
        return Hx.to(self.device)
    
    

    def Xdx_fun(self, state):
        q_rt = state[:,3:7,:]
        # print("q_rt = ", q_rt.size())
        Q_th_rt = self.quaternion_to_left_matrix(q_rt)

        Xdx = torch.zeros(self.n_batch, 7, 6)
        for i in range(self.n_batch):
            Xdx_1 = torch.cat([torch.eye(3), 
                            torch.zeros((4,3))], dim=0).to(self.device)
            Xdx_2 = torch.cat([torch.zeros((3,3)).to(self.device), 
                            Q_th_rt[i,:,:]], dim=0)
            Xdx[i,:,:]= torch.cat([Xdx_1, Xdx_2], dim=1)

        return Xdx.to(self.device)
    
    def quaternion_multiply(self, q1, q2):
        # 假设 q1 和 q2 的形状为 [n_batch, 4, 1]
        # 其中四元数以 [w, x, y, z] 的形式存储

        # 分别获取 w, x, y, z
        w1, x1, y1, z1 = q1[:, 0, :], q1[:, 1, :], q1[:, 2, :], q1[:, 3, :]
        w2, x2, y2, z2 = q2[:, 0, :], q2[:, 1, :], q2[:, 2, :], q2[:, 3, :]

        # 执行四元数乘法
        # w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        # x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        # y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        # z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # 将结果组合回四元数的形式
        q = torch.cat([w, x, y, z], dim=1).unsqueeze(2)

        return q.to(self.device)
    
    # def quaternion_multiply(self,q1, q2):
    #     w1 = q1[0]
    #     x1 = q1[1]
    #     y1 = q1[2]
    #     z1 = q1[3]

    #     # print("w1", w1.size())

    #     w2 = q2[0]
    #     x2 = q2[1]
    #     y2 = q2[2]
    #     z2 = q2[3]

    #     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    #     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    #     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    #     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    #     # print("torch.tensor([w, x, y, z])", torch.tensor([w, x, y, z]).size())


    #     return torch.tensor([w, x, y, z], requires_grad=True).unsqueeze(1).to(self.device)
    
    def _state_injection(self, dt, state, error_state):
        dt = dt.to(self.device)
        state = state.to(self.device)
        error_state = error_state.to(self.device)
        true_state = torch.zeros_like(state)
        try:
            true_state[:,0:3,:] = state[:,0:3,:] + error_state[:,0:3,:]
        except RuntimeError as e:
            # 处理RuntimeError
            print(f"Caught an exception: {e}")
            print("true_state =",true_state.size())
            print("state =",state.size())
            print("error_state =",error_state.size())

        # output is not a quaterion vector
        # for i in range(self.n_batch):
        #     dx1,dy1,dz1 = 0.5 * error_state[i,3:6,:]
        #     dq1 = torch.tensor([0.0, dx1, dy1, dz1], requires_grad=True).unsqueeze(1).to(self.device)
        #     # print("dq1 = ",dq1.size())
        #     q1 = self.quaternion_multiply(state[i,3:7,:], dq1)*dt + state[i,3:7,:]
        #     # print("state[i,3:7,:] = ",state[i,3:7,:].size())
        #     # print("q1 = ",q1.size())

        #     true_state[i,3:7,:] = q1
        # print("error_state[:, 3, :] = ",error_state)
        # raise ValueError("123")
        dx = 0.5 * error_state[:, 3, :].unsqueeze(1)
        dy = 0.5 * error_state[:, 4, :].unsqueeze(1)
        dz = 0.5 * error_state[:, 5, :].unsqueeze(1)
        dq = torch.cat([torch.zeros(self.n_batch, 1, 1, device=self.device), dx, dy, dz], dim=1)
        # dq.requires_grad = True
        q1 = self.quaternion_multiply(dq,state[:, 3:7, :])*dt + state[:, 3:7, :]

        true_state[:, 3:7, :] = q1

        return true_state.to(self.device)
    
    def quaternion_to_rotation_matrix(self, q):
        """
        将四元数转换为旋转矩阵

        参数：
        q: 四元数，表示为 [w, x, y, z]

        返回：
        R: 旋转矩阵,3x3的NumPy数组
        """
        w, x, y, z = q[:,0,:],q[:,1,:],q[:,2,:],q[:,3,:]

        R_1 = torch.cat([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=1).unsqueeze(2)
        R_2 = torch.cat([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=1).unsqueeze(2)
        R_3 = torch.cat([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=1).unsqueeze(2)


        R = torch.cat([R_1, R_2, R_3], dim=2)
        # print("R_1", R_1.shape)

        # R = R.repeat(self.n_batch,1,1)

        return R.to(self.device)
    

      
    def _skew_symmetric_matrix(self, v):

        sk1 = torch.cat([torch.zeros_like(v[:,2,:]), -v[:,2,:], v[:,1,:]], dim=1)
        sk2 = torch.cat([v[:,2,:], torch.zeros_like(v[:,2,:]), -v[:,0,:]], dim=1)
        sk3 = torch.cat([-v[:,1,:], v[:,0,:], torch.zeros_like(v[:,2,:])], dim=1)

        sk = torch.cat([sk1, sk2, sk3], dim=0).unsqueeze(0)

        sk = sk.repeat(self.n_batch,1,1)
        # skew symmetric matrix for Forward kinematics
        return sk.to(self.device)
    
    def reset_error_state(self, state, dt,contro_vector):
        t_t = state[:,0:3,:]
        q_t = state[:,3:7,:]
        
        R_t = self.quaternion_to_rotation_matrix(q_t)
        W33 = self._skew_symmetric_matrix(contro_vector[:,3:6,:])*dt

        # print("W33 = ",W33)

        I33 = torch.eye(3).repeat(self.n_batch,1,1).to(self.device)
        Rt1 = (I33 + W33)@R_t

        # print("Rt1 = ",Rt1.shape)

        sp =  Rt1.transpose(1,2) @ t_t -t_t
        agp = torch.cat([sp, torch.zeros_like(t_t).to(self.device)], dim=1)

        # print("agp = ", agp)

        agp = agp.repeat(self.n_batch,1,1)

        return agp.to(self.device)


