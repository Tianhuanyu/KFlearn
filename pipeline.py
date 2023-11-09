import torch
from nnkf import ESKF_Torch,KalmanNet
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings

from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


class Pipeline:
    def __init__(self, 
                 number_list_train:list = list(range(0,11)), 
                 number_list_valid:list = list(range(11,14)), 
                 number_list_test:list = list(range(14,15)), 
                 modelName:str= "saved_state.csv",
                 args=None):
        self.sysModel = None
        self.model = None

        instance_train = RegistrationData(
            number_list=number_list_train,
            names=[modelName])
        self.data_loader_train = instance_train.generateDataLoader(window_size= args.n_seq)

        instance_valid = RegistrationData(
            number_list=number_list_valid,
            names=[modelName])
        self.data_loader_valid = instance_valid.generateDataLoader(window_size= args.n_seq)

        instance_test = RegistrationData(
            number_list=number_list_test,
            names=[modelName])
        self.data_loader_test = instance_test.generateDataLoader(window_size= args.n_seq)

    # TODO
    def setssModel(self, sysModel):
        self.sysModel = sysModel

    def setNNModel(self, model):
        self.model = model

    def setargs(self, args):
        self.args = args

    def buildSetup(self, sysModel, args):
        self.setssModel(sysModel)
        self.setargs(args)
        

    def trainFilter(self):
        # print("sefl.mddel = ",self.model)
        if(self.model is None or self.sysModel is None):
            raise ValueError("You should initial the model using setssModel and setModel first!")


        data_test = DataLoader(self.data_loader_train,
                               batch_size=self.args.n_batch, 
                               shuffle=True, 
                               num_workers=4)
        
        for _x_traj, _y_traj in data_test:

            x_traj = _x_traj.permute(1, 2, 0)
            y_traj = _y_traj.permute(1, 2, 0)

            init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)

            i=0
            diag_matrix_P = torch.diag(torch.tensor([0.001]*3+
                            [0.002]*3, requires_grad=True)).unsqueeze(0).repeat(self.args.n_batch,1,1)
            diag_matrix_R = torch.diag(torch.tensor(
                [0.0001]*3+
                            [0.01,0.01,0.01,0.01], requires_grad=True)
            ).unsqueeze(0).repeat(self.args.n_batch,1,1)

            T_fitler = torch.tensor(0.01)
            self.model.reset_init_state(init_state, 
                                        diag_matrix_P, 
                                        diag_matrix_R, 
                                        T_fitler,
                                        self.args)
            for x, y in zip(x_traj,y_traj):
                i+=1
 
                state = self.model(x)


                gt = y[0:7].unsqueeze(0)
                gt = gt.permute(2,1,0)
                print("error = {0}".format(torch.norm(gt[:,0:3,:]-state[:,0:3,:])))

    def trainNetwork(self):
        self.learningRate = self.args.lr
        self.weightDecay = self.args.wd
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        data_test = DataLoader(self.data_loader_train,
                        batch_size=self.args.n_batch, 
                        shuffle=True, 
                        num_workers=4)
        criterion = torch.nn.MSELoss(reduction='mean')
        
        writer = SummaryWriter('runs/experiment_1')

        def loss_with_acc(out,out_p,out_p2, y):
            return 0.7*(criterion(out[:,0:3], y[:,0:3]) + 
                             0.1* criterion(out[:,3:7], y[:,3:7])) + 0.02*(
                                 criterion(out[:,0:3]+out_p2[:,0:3], 2.0*out_p[:,0:3]) + 
                             0.1* criterion(out[:,3:7]+out_p2[:,3:7], 2.0*out_p[:,3:7]))+ 0.1*(
                                 criterion(x[:,0:3], out[:,0:3]) + 0.1* criterion(x[:,3:7], out[:,3:7])
                             )



        def quaternion_loss(q_pred, q_true):
            # 确保四元数是归一化的
            # q_pred = q_pred / q_pred.norm(p=2, dim=1, keepdim=True)
            # q_true = q_true / q_true.norm(p=2, dim=1, keepdim=True)

            # 计算四元数之间的点积
            dot_product = (q_pred * q_true).sum(dim=1)

            # 解决符号二义性
            dot_product = torch.abs(dot_product)

            # 计算角度损失
            loss = 1 - dot_product**2
            return loss.mean()  # 返回批次的平均损失
        # criterion_quat = quaternion_loss()

        # i=0

        for epoch in range(10):
            for tj_id, (_x_traj, _y_traj) in enumerate(data_test):
                self.optimizer.zero_grad()
                x_traj = _x_traj.permute(1, 2, 0)
                y_traj = _y_traj.permute(1, 2, 0)
                loss = torch.tensor(0.0)
                loss_c = torch.tensor(0.0)

                init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)

                if(init_state.size()[0] != self.args.n_batch):
                    break

                
                self.model.reset_state(init_state)
                out_p = torch.zeros_like(init_state).squeeze(2)
                out_p2 = torch.zeros_like(init_state).squeeze(2)

                x_p = torch.zeros_like(init_state).squeeze(2)
                x_p2 = torch.zeros_like(init_state).squeeze(2)
                for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
                    # print("i =",i)
                    # print("x = ",x)
                    out = self.model(x).squeeze(2)

                    #TODO
                    # untimeError: The expanded size of the tensor (1) must match the existing size (4) at non-singleton dimension 0.  Target sizes: [1, 3, 1].  Tensor sizes: [4, 3, 1]

                    # print("out = ",out.size())
                    # print("init_state = ",init_state.size())
                    # print("y = ",y)
                    # print("x = ",x)
                    y = y.permute(1,0)
                    x = x.permute(1,0)

                    """
                    Loss 1.0
                    """
                    # loss += 0.7*(criterion(out[:,0:3], y[:,0:3]) + 
                    #          0.1* criterion(out[:,3:7], y[:,3:7])) + 0.3*(
                    #              criterion(x[:,0:3], out[:,0:3]) + 
                    #          0.1* criterion(x[:,3:7], out[:,3:7]))
                             
                    
                    loss_c += loss_with_acc(x[:,0:7],x_p, x_p2, y)
                    
                    """
                    Loss with min acc
                    """
                    loss += loss_with_acc(out,out_p, out_p2, y)
                    
                    out_p2 = out_p
                    out_p = out

                    x_p2 = x_p
                    x_p = x[:,0:7]
                    
                    
                    # print(f'Epoch {epoch+1}, Traj id {tj_id},Loss: {loss.item()}, LossC {loss_c.item()}')
                    # if(ptid == 20):
                    #     raise ValueError("Num ERR")

                loss.backward(retain_graph=True)
                self.optimizer.step()

                # Print statistics
                # if (epoch+1) % 10 == 0:
                writer.add_scalar('Loss/train', loss, epoch * len(data_test) + tj_id)
                print(f'Epoch {epoch+1}, Traj id {tj_id},Loss: {loss.item()}, LossC {loss_c.item()}')
                # if(loss.item() >1.0):
                #     print("y = ",y)
                #     print("x = ",x)
                #     print("out = ",out)
                    
                #     raise ValueError("Data break")


                    










def main():

    args = general_settings()
    instance = Pipeline(args=args)

    task_model = RobotSensorFusion(state_size=7,
                                   error_state_size=6,
                                   args=args)
    instance.buildSetup(task_model,args)



    ini_state = torch.tensor([
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ]).unsqueeze(0).unsqueeze(2).repeat(args.n_batch,1,1)
    ini_covariance = torch.eye(
        6).unsqueeze(0).repeat(args.n_batch,1,1)

    KF_model = ESKF_Torch(system_model= task_model,
                          initial_state=ini_state,
                          initial_covariance=ini_covariance)

    instance.setNNModel(KF_model)

    instance.trainFilter()


def mainKFNet():
    args = general_settings()
    instance = Pipeline(args=args)


    task_model = RobotSensorFusion(state_size=7,
                                   error_state_size=6,
                                   args=args)
    instance.buildSetup(task_model,args)



    ini_state = torch.tensor([
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ]).unsqueeze(0).unsqueeze(2).repeat(args.n_batch,1,1)
    ini_covariance = torch.eye(
        6).unsqueeze(0).repeat(args.n_batch,1,1)
    
    T_fitler = torch.tensor(0.01)

    KF_model = KalmanNet(system_model= task_model,
                          initial_state=ini_state,
                          initial_covariance=ini_covariance,
                          args=args,
                          dt=T_fitler)

    instance.setNNModel(KF_model)

    instance.trainNetwork()

if  __name__ == "__main__":
    mainKFNet()
