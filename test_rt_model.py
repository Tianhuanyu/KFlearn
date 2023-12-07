import torch
from nnkf import ESKF_Torch,KalmanNet
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings


def main():
    args = general_settings()
    args.n_batch = 1

    task_model = RobotSensorFusion(state_size=7,
                                   error_state_size=6,
                                   args=args)
    
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
    
    pth = 'best_model_0.001_20_0.0001_seq100_KFNET.pth'
    KF_model.load_state_dict(torch.load(pth))
    KF_model.eval()

    init_state = torch.tensor([0.0]*3+[1.0]+[0.0]*3).unsqueeze(0).unsqueeze(2).cuda()

    x = torch.tensor([0.0]*14).unsqueeze(1).cuda()
    print(x)
    repro_error = x[7,:].unsqueeze(0).unsqueeze(2)

    KF_model.reset_state(init_state,repro_error)

    print(x.shape)
    out = KF_model(x)

    print(out)


if  __name__ == "__main__":
    main()
    




