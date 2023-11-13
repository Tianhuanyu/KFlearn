import torch
from nnkf import ESKF_Torch,KalmanNet
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings

from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import copy


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
        self.data_loader_test = instance_test.generateDataLoader(window_size= args.n_seq, is_test=True)

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
        self.criterion = torch.nn.MSELoss(reduction='mean')
        

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

    def loss_with_acc(self,out,out_p,out_p2, y,x):
        return 100.0*(self.criterion(out[:,0:3], y[:,0:3]) + 
                            0.2* self.criterion(out[:,3:7], y[:,3:7])) + 1.0*(
                                self.criterion(out[:,0:3]+out_p2[:,0:3], 2.0*out_p[:,0:3]) + 
                            0.2* self.criterion(out[:,3:7]+out_p2[:,3:7], 2.0*out_p[:,3:7]))+ 1.0*(
                                self.criterion(x[:,0:3], out[:,0:3]) + 0.2* self.criterion(x[:,3:7], out[:,3:7])
                            )

    def lossinTraj(self, init_state, x_traj, y_traj):
        loss = torch.tensor(0.0).to(self.model.device)
        loss_c = torch.tensor(0.0).to(self.model.device)
        self.model.reset_state(init_state)
        out_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        out_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)

        x_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
            # print("i =",i)
            # print("x = ",x)
            out = self.model(x).squeeze(2)

            y = y.permute(1,0)
            x = x.permute(1,0)

            """
            Loss 1.0
            """

            loss_c += self.loss_with_acc(x[:,0:7],x_p, x_p2, y,x)
            
            """
            Loss with min acc
            """
            loss += self.loss_with_acc(out,out_p, out_p2, y,x)
            
            out_p2 = out_p
            out_p = out

            x_p2 = x_p
            x_p = x[:,0:7]
            # print("RUn to here ptid = ",ptid)

        return loss, loss_c

    def trainNetwork(self, pth=None):
        if pth:
            self.model.load_state_dict(torch.load(pth))

        self.learningRate = self.args.lr
        self.weightDecay = self.args.wd
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        data_train = DataLoader(self.data_loader_train,
                        batch_size=self.args.n_batch, 
                        shuffle=True, 
                        num_workers=self.args.num_workers,
                        pin_memory=True,
                        prefetch_factor=self.args.prefetch_factor)
        

        data_valid = DataLoader(self.data_loader_valid,
                        batch_size=self.args.n_batch, 
                        shuffle=True, 
                        num_workers=self.args.num_workers,
                        pin_memory=True,
                        prefetch_factor=self.args.prefetch_factor)
        
        print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        
        writer = SummaryWriter('runs/experiment_1')


        # criterion_quat = quaternion_loss()

        # i=0

        best_loss = float('inf')

        for epoch in range(10):

            self.model.train()
            for tj_id, (_x_traj, _y_traj) in enumerate(data_train):
                self.optimizer.zero_grad()
                x_traj = _x_traj.permute(1, 2, 0).to(self.model.device)
                y_traj = _y_traj.permute(1, 2, 0).to(self.model.device)


                init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)
                if(init_state.size()[0] != self.args.n_batch):
                    break

                loss, loss_c = self.lossinTraj(init_state, x_traj, y_traj)


                loss.backward(retain_graph=True)
                self.optimizer.step()

                # Print statistics
                # if (epoch+1) % 10 == 0:
                writer.add_scalar('Loss/train', loss, epoch * len(data_train) + tj_id)
                # if(tj_id % 50):
                print(f'Epoch {epoch+1}, 111 Traj id {tj_id},Loss: {loss.item()}, LossC {loss_c.item()}', flush=True)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_c = 0.0
                for tj_id, (_x_traj, _y_traj) in enumerate(data_valid):
                    self.optimizer.zero_grad()
                    x_traj = _x_traj.permute(1, 2, 0).to(self.model.device)
                    y_traj = _y_traj.permute(1, 2, 0).to(self.model.device)


                    init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)
                    if(init_state.size()[0] != self.args.n_batch):
                        break

                    loss, loss_c = self.lossinTraj(init_state, x_traj, y_traj)
                    val_loss += loss
                    val_loss_c += loss_c
                    print(f'Epoch {epoch+1}, 111 Traj id {tj_id},Loss: {val_loss}, LossC {val_loss_c}', flush=True)
                val_loss /= len(data_valid)
                writer.add_scalar('Loss/valid', val_loss, epoch * len(data_valid) + tj_id)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                # 保存模型
                torch.save(self.model.state_dict(), 'best_model_{0}_{1}_{2}_epoch{3}.pth'.format(self.args.lr, self.args.n_batch, self.args.wd, epoch))


    def testModelwithpth(self, pth):
        self.model.load_state_dict(torch.load(pth))
        self.model.eval()

        data_test = DataLoader(self.data_loader_test,
                batch_size=1, 
                shuffle=True, 
                num_workers=self.args.num_workers)

        print(" len(data_test) = ",len(data_test))
        with torch.no_grad():
            val_loss = 0.0
            val_loss_c = 0.0
            xs_list = []
            ys_list = []
            os_list = []
            for tj_id, (_x_traj, _y_traj) in enumerate(data_test):
                x_traj = _x_traj.permute(1, 2, 0).to(self.model.device)
                y_traj = _y_traj.permute(1, 2, 0).to(self.model.device)


                init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)

                loss, loss_c,xs, ys, os = self.outputinTraj(init_state, x_traj, y_traj)
                val_loss += loss
                val_loss_c += loss_c
                xs_list.append(xs)
                ys_list.append(ys)
                os_list.append(os)

                print(f' Traj id {tj_id},Loss: {val_loss}, LossC {val_loss_c}  x_traj {x_traj.shape}')
            val_loss /= len(data_test)


        print(os_list)
        print("len = ", len(os_list[0]))

        return xs_list, ys_list, os_list

    def outputinTraj(self, init_state, x_traj, y_traj):
        loss = torch.tensor(0.0).to(self.model.device)
        loss_c = torch.tensor(0.0).to(self.model.device)
        self.model.reset_state(init_state)
        out_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        out_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)

        x_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        output_record_x = []
        output_record_y = []
        output_record_o = []
        for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
            # print("i =",i)
            # print("x = ",x)
            out = self.model(x).squeeze(2)

            y = y.permute(1,0)
            x = x.permute(1,0)

            """
            Loss 1.0
            """

            loss_c += self.loss_with_acc(x[:,0:7],x_p, x_p2, y,x)
            
            """
            Loss with min acc
            """
            loss += self.loss_with_acc(out,out_p, out_p2, y,x)
            
            out_p2 = out_p
            out_p = out

            x_p2 = x_p
            x_p = x[:,0:7]

            output_record_x.append(x[:,0:3].squeeze().cpu().detach().numpy().tolist())
            output_record_y.append(y[:,0:3].squeeze().cpu().detach().numpy().tolist())
            output_record_o.append(out[:,0:3].squeeze().cpu().detach().numpy().tolist())
            # print("RUn to here ptid = ",ptid)

        return loss, loss_c, output_record_x, output_record_y, output_record_o

            

                


                
            
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
    ]).unsqueeze(0).unsqueeze(2).repeat(args.n_batch,1,1).to(task_model.device)
    ini_covariance = torch.eye(
        6).unsqueeze(0).repeat(args.n_batch,1,1)
    
    T_fitler = torch.tensor(0.01).to(task_model.device)

    KF_model = KalmanNet(system_model= task_model,
                          initial_state=ini_state,
                          initial_covariance=ini_covariance,
                          args=args,
                          dt=T_fitler).cuda()

    instance.setNNModel(KF_model)

    instance.trainNetwork()

if  __name__ == "__main__":
    mainKFNet()
