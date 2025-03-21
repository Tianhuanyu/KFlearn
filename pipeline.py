import torch
from nnkf import ESKF_Torch,KalmanNet
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings

from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import copy
import time


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
                # print("error = {0}".format(torch.norm(gt[:,0:3,:]-state[:,0:3,:])))

    def loss_with_acc(self,out,out_p,out_p2, y,x):
        return (1.0 *self.criterion(out[:,0:3], y[:,0:3]) + 0.2*self.criterion(out[:,3:7], y[:,3:7]) 
        + 0.03*self.criterion(out[:,0:3]+ out_p2[:,0:3], 2.0* out_p[:,0:3]) +
         0.01*self.criterion(out[:,3:7] + out_p2[:,3:7], 2.0* out_p[:,3:7]))

    def lossinTraj(self, init_state, x_traj, y_traj,r_error):
        loss = torch.tensor(0.0).to(self.model.device)
        loss_c = torch.tensor(0.0).to(self.model.device)
        self.model.reset_state(init_state,r_error)
        out_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        out_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)

        x_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
            # print("i =",i)
            # print("x = ",x)

            # if(torch.norm(x[4,:])<0.1):
            #     init_state = x_traj[ptid,0:7,:].unsqueeze(0).permute(2, 1, 0)
            #     repro_error = x_traj[ptid,7,:].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)
            #     self.model.reset_state(init_state,repro_error)
            # else:
            #     pass

            out = self.model(x).squeeze(2)


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
            self.model.load_state_dict(torch.load(pth), strict=False)

        self.learningRate = self.args.lr
        self.weightDecay = self.args.wd
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        data_train = DataLoader(self.data_loader_train,
                        batch_size=self.args.n_batch, 
                        shuffle=False, 
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

        best_loss = 0.2
        max_norm =0.1

        for epoch in range(100):

            self.model.train()
            for tj_id, (_x_traj, _y_traj) in enumerate(data_train):
                self.optimizer.zero_grad()
                x_traj = _x_traj.permute(1, 2, 0).to(self.model.device)
                y_traj = _y_traj.permute(1, 2, 0).to(self.model.device)


                init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)
                # print("x_traj[0,0:7,:] = {0},  x_traj[0,7,:] = {1}".format(x_traj[0,0:7,:].shape, x_traj[0,7,:].unsqueeze(0).unsqueeze(0).shape))
                re_error = x_traj[0,7,:].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)
                if(init_state.size()[0] != self.args.n_batch):
                    break

                loss, loss_c = self.lossinTraj(init_state, x_traj, y_traj,re_error)

                loss = loss/torch.tensor(x_traj.size()[0]).to(self.model.device)
                loss_c = loss_c/torch.tensor(x_traj.size()[0]).to(self.model.device)


                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
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
                    re_error = x_traj[0,7,:].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)
                    if(init_state.size()[0] != self.args.n_batch):
                        break

                    loss, loss_c = self.lossinTraj(init_state, x_traj, y_traj,re_error)
                    val_loss += loss
                    val_loss_c += loss_c
                    print(f'Epoch {epoch+1}, 111 Traj id {tj_id},Loss: {loss}, LossC {loss_c}', flush=True)
                val_loss /= len(data_valid)
                writer.add_scalar('Loss/valid', val_loss, epoch * len(data_valid) + tj_id)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                # 保存模型
                torch.save(self.model.state_dict(), 'Dbest_model_{0}_{1}_{2}_seq{3}_KFNET.pth'.format(self.args.lr, self.args.n_batch, self.args.wd, self.args.n_seq))


    def testModelwithpth(self, pth, update_period=None):
        if(pth):
            self.model.load_state_dict(torch.load(pth))
            self.model.eval()

        data_test = DataLoader(self.data_loader_test,
                batch_size=1, 
                shuffle=True, 
                num_workers=self.args.num_workers)

        # print(" len(data_test) = ",len(data_test))
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
                re_error = x_traj[0,7,:].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)

                loss, loss_c,xs, ys, os = self.outputinTraj(init_state, x_traj, y_traj,update_period,re_error)
                val_loss += loss
                val_loss_c += loss_c
                xs_list.append(xs)
                ys_list.append(ys)
                os_list.append(os)

                print(f' Traj id {tj_id},Loss: {loss}, LossC {loss_c}  x_traj {x_traj.shape}')
            val_loss /= len(data_test)

        return xs_list, ys_list, os_list


    def ValidModelwithpth(self, pth, update_period=None, max_iter=None):
        if(pth):
            self.model.load_state_dict(torch.load(pth))
            self.model.eval()

        data_test = DataLoader(self.data_loader_valid,
                batch_size=1, 
                shuffle=True, 
                num_workers=self.args.num_workers)

        # print(" len(data_test) = ",len(data_test))
        with torch.no_grad():

            lossps = []
            lossqs = []
            loss_cps = [] 
            loss_cqs= [] 
            loss_accps = []
            loss_accqs = []
            loss_accpxs = []
            loss_accqxs = []
            ts = []

            for tj_id, (_x_traj, _y_traj) in enumerate(data_test):
                x_traj = _x_traj.permute(1, 2, 0).to(self.model.device)
                y_traj = _y_traj.permute(1, 2, 0).to(self.model.device)

                if(max_iter and tj_id>max_iter):
                    break


                init_state = x_traj[0,0:7,:].unsqueeze(0).permute(2, 1, 0)
                re_error = x_traj[0,7,:].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)

                start_time = time.time()

                lossp,lossq, loss_cp, loss_cq, loss_accp,loss_accq,loss_accpx,loss_accqx = self.outputinTrajpq(init_state, x_traj, y_traj,update_period,re_error)

                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time  # 计算所用时间        

                lossps.append(lossp.cpu())
                lossqs.append(lossq.cpu())
                loss_cps.append(loss_cp.cpu()) 
                loss_cqs.append(loss_cq.cpu()) 
                loss_accps.append(loss_accp.cpu())
                loss_accqs.append(loss_accq.cpu())
                loss_accpxs.append(loss_accpx.cpu())
                loss_accqxs.append(loss_accqx.cpu())
                ts.append(elapsed_time)


                print(f' Traj id {tj_id},Loss: {lossp}, LossC {loss_cp}  x_traj {x_traj.shape}  time={elapsed_time}')
            # val_loss /= len(data_test)

        return lossps, lossqs, loss_cps, loss_cqs, loss_accps, loss_accqs, loss_accpxs, loss_accqxs,ts
    



    def outputinTrajpq(self, init_state, x_traj, y_traj,update_period,repro_error):
        lossp = torch.tensor(0.0).to(self.model.device)
        loss_cp = torch.tensor(0.0).to(self.model.device)

        lossq = torch.tensor(0.0).to(self.model.device)
        loss_cq = torch.tensor(0.0).to(self.model.device)

        loss_accp = torch.tensor(0.0).to(self.model.device)
        loss_accq = torch.tensor(0.0).to(self.model.device)

        loss_accpx = torch.tensor(0.0).to(self.model.device)
        loss_accqx = torch.tensor(0.0).to(self.model.device)


        out_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        out_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        self.model.reset_state(init_state,repro_error)
        # output_record_x = []
        # output_record_y = []
        # output_record_o = []
        for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
            if(update_period and ptid % int(update_period)== 0):
                init_state = x_traj[ptid,0:7,:].unsqueeze(0).permute(2, 1, 0)
                repro_error = x_traj[ptid,7,:].unsqueeze(0).permute(2, 1, 0)
                self.model.reset_state(init_state,repro_error)

            # print("x = ",x)
            # print("x = ", x.shape)
            # raise ValueError("Run to here")
            if (torch.any(torch.isnan(x))):
                out = out_p
            else:
                # if(x[4,:])
                # if(torch.norm(x[4,:])<0.1):
                #     init_state = x_traj[ptid,0:7,:].unsqueeze(0).permute(2, 1, 0)
                #     repro_error = x_traj[ptid,7,:].unsqueeze(0).unsqueeze(2).permute(2, 1, 0)
                #     self.model.reset_state(init_state,repro_error)
                # else:
                #     pass

                out = self.model(x).squeeze(2)

            y = y.permute(1,0)
            x = x.permute(1,0)

            lossp += self.criterion(out[:,0:3], y[:,0:3])
            lossq += self.criterion(out[:,3:7], y[:,3:7])


            loss_cp += self.criterion(x[:,0:3], y[:,0:3])
            loss_cq += self.criterion(x[:,3:7], y[:,3:7])

            loss_accp += self.criterion(out[:,0:3]+out_p2[:,0:3], out_p[:,0:3]+out_p[:,0:3])
            loss_accq += self.criterion(out[:,3:7]+out_p2[:,3:7], out_p[:,3:7]+out_p[:,3:7])

            loss_accpx += self.criterion(x[:,0:3]+x_p2[:,0:3], x_p[:,0:3]+x_p[:,0:3])
            loss_accqx += self.criterion(x[:,3:7]+x_p2[:,3:7], x_p[:,3:7]+x_p[:,3:7])

            
            out_p2 = out_p
            out_p = out

            x_p2 = x_p
            x_p = x[:,0:7]

            # output_record_x.append(x[:,0:7].squeeze().cpu().detach().numpy().tolist())
            # output_record_y.append(y[:,0:7].squeeze().cpu().detach().numpy().tolist())
            # output_record_o.append(out[:,0:7].squeeze().cpu().detach().numpy().tolist())
            # print("RUn to here ptid = ",ptid)

        return lossp,lossq, loss_cp, loss_cq, loss_accp,loss_accq,loss_accpx,loss_accqx




    def outputinTraj(self, init_state, x_traj, y_traj,update_period,repro_error):
        loss = torch.tensor(0.0).to(self.model.device)
        loss_c = torch.tensor(0.0).to(self.model.device)
        out_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        out_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        x_p2 = torch.zeros_like(init_state).squeeze(2).to(self.model.device)
        self.model.reset_state(init_state,repro_error)
        output_record_x = []
        output_record_y = []
        output_record_o = []
        # print("init_state shape",init_state.shape)
        is_first = True
        for ptid, (x, y) in enumerate(zip(x_traj,y_traj)):
            if(update_period and ptid % int(update_period)== 0):
                # try:
                init_state = x_traj[ptid,0:7,:].unsqueeze(0).permute(2, 1, 0)
                repro_error = x_traj[ptid,7,:].unsqueeze(0).unsqueeze(2).permute(2, 1, 0)
                self.model.reset_state(init_state,repro_error)


            # print("x = ",x)
            # print("x = ", x.shape)
            # raise ValueError("Run to here")
            if (torch.any(torch.isnan(x))):
                out = out_p
            else:
                # # if(x[4,:])
                # if(torch.norm(x[3,:])<0.1):
                #     init_state = x_traj[ptid,0:7,:].unsqueeze(0).permute(2, 1, 0)
                #     repro_error = x_traj[ptid,7,:].unsqueeze(0).unsqueeze(2).permute(2, 1, 0)
                #     self.model.reset_state(init_state,repro_error)
                # else:
                #     pass

                out = self.model(x).squeeze(2)
                # print(x[4,:])
            # if(out[0,0]==torch.tensor(float('nan'))):
            

            # 使用 torch.any() 检查是否存在任何 nan 值
            # if (torch.any(torch.isnan(out)) and is_first):
            #     print("x = ",x.permute(1, 0))
            #     print("out = ",out)
            #     is_first = False
                # print("out = {0}  {1}  {2}".format(out[0,0]==torch.tensor(float('nan')).to(self.model.device), type(out[0,0]), type(torch.tensor(float('nan')).to(self.model.device)) ))
            # print("out = ",type(out))

            y = y.permute(1,0)
            x = x.permute(1,0)

            """
            Loss 1.0
            """

            loss_c = loss_c*0.95 + self.loss_with_acc(x[:,0:7],x_p, x_p2, y,x)
            
            """
            Loss with min acc
            """
            loss = loss*0.95 + self.loss_with_acc(out,out_p, out_p2, y,x)
            
            out_p2 = out_p
            out_p = out

            x_p2 = x_p
            x_p = x[:,0:7]

            output_record_x.append(x[:,0:7].squeeze().cpu().detach().numpy().tolist())
            output_record_y.append(y[:,0:7].squeeze().cpu().detach().numpy().tolist())
            output_record_o.append(out[:,0:7].squeeze().cpu().detach().numpy().tolist())
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
                          initial_covariance=ini_covariance,
                          args=args)

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
