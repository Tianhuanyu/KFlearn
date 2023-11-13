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

    KF_model = KalmanNet(system_model= task_model,
                          initial_state=ini_state,
                          initial_covariance=ini_covariance,
                          args=args,
                          dt=T_fitler).cuda()

    instance.setNNModel(KF_model)

    instance.testModelwithpth('best_model.pth')

if  __name__ == "__main__":
    main()