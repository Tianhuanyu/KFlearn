import torch
from nnkf import ESKF_Torch,KalmanNet,KalmanNetOrigin
from SystemModel import RobotSensorFusion

from finding_ground_truth import RegistrationData, TimeSeriesDataset
from config import general_settings

from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import copy
from pipeline import Pipeline
import numpy as np


def get_mean_std(xlist):
    xlistnp = np.array(xlist)
    return np.mean(xlistnp), np.std(xlistnp)




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
    

    # xs_list, ys_list, os_list = instance.testModelwithpth('best_model_0.001_150_0.0001_seq100_KFNET.pth')
    # lossp,lossq, loss_cp, loss_cq, loss_accp,loss_accq,loss_accpx,loss_accqx,ts =instance.ValidModelwithpth('best_model_0.001_150_0.0001_seq100_KFNET.pth',max_iter=1000)
    # lossp,lossq, loss_cp, loss_cq, loss_accp,loss_accq,loss_accpx,loss_accqx,ts =instance.ValidModelwithpth('best_model.pth', max_iter=100)

    # pmean, pstd = get_mean_std(lossp)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(lossq)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_cp)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_cq)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_accp)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_accq)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_accpx)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))
    # pmean, pstd = get_mean_std(loss_accqx)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))

    # pmean, pstd = get_mean_std(ts)
    # print("loss mean = {0} lose std = {1}".format(pmean, pstd))

    # print()
    xs_list, ys_list, os_list = instance.testModelwithpth('best_model_0.001_500_0.0001_seq100_KFNET.pth',1000)

    # """
    # ESKF
    # """

    KF_model1 = ESKF_Torch(system_model= task_model,
                    initial_state=ini_state,
                    initial_covariance=ini_covariance,
                    args=args)
    
    instance.setNNModel(KF_model1)

    xs_list, ys_list, os_list1 = instance.testModelwithpth(None)


    # """
    # Origin KalmanNet
    
    # """

    # KF_model2 = KalmanNetOrigin(system_model= task_model,
    #                     initial_state=ini_state,
    #                     initial_covariance=ini_covariance,
    #                     args=args,
    #                     dt=T_fitler).cuda()
    # instance.setNNModel(KF_model2)
    # xs_list, ys_list, os_list2 = instance.testModelwithpth('Originbest_model_0.001_150_0.0001_seq200_KFNET.pth')



    # RegistrationData.view_channels(xs_list[0], ys_list[0])
    RegistrationData.view_channels(xs_list[0], ys_list[0],os_list[0],os_list1[0])


    for name, param in KF_model.named_parameters():
        print(name, param.requires_grad)

if  __name__ == "__main__":
    main()