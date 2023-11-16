import torch
from nnkf import ESKF_Torch,KalmanNet
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings

from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import copy
from pipeline import Pipeline


def main():
    args = general_settings()

    args.n_batch = 1
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

    # KF_model = KalmanNet(system_model= task_model,
    #                       initial_state=ini_state,
    #                       initial_covariance=ini_covariance,
    #                       args=args,
    #                       dt=T_fitler).cuda()

    KF_model = ESKF_Torch(system_model= task_model,
                    initial_state=ini_state,
                    initial_covariance=ini_covariance,
                    args=args)

    instance.setNNModel(KF_model)

    # xs_list, ys_list, os_list = instance.testModelwithpth('best_model_0.001_500_0.0001_seq100_KFNET.pth',50)
    xs_list, ys_list, os_list = instance.testModelwithpth(None,500)
    RegistrationData.view_channels(xs_list[0], ys_list[0])
    RegistrationData.view_channels(xs_list[0], ys_list[0], os_list[0])


    for name, param in KF_model.named_parameters():
        print(name, param.requires_grad)

if  __name__ == "__main__":
    main()