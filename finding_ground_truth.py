
import os
import csv
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
# import open3d as o3d
from sklearn.cluster import DBSCAN

import torch
from torch.utils.data import Dataset, DataLoader

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia



# x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, r_error,r_error2, x, y, z, qw, qx, qy, qz
# saved_state = self.measurement_save.reshape(-1,1).flatten().tolist()+twist.reshape(-1,1).flatten().tolist()+\
#             [self.goal_var_trocar] +[self.goal_var_ee] + \
#                     current_pose.pos.reshape(-1,1).flatten().tolist() + current_pose.rot.reshape(-1,1).flatten().tolist()
INDEX_FOR_REPROJ = 13
INDEX_FOR_REPROJ2 = 14
accept_rate = 0.5
SAMPLE_STEP = 30
T_step = 0.01
DISPLAY = [0,1,2]



class TimeSeriesDataset(Dataset):
    def __init__(self, data_input, data_output, window_size, is_test=False):

        self.data_input = data_input
        self.data_output = data_output
        self.window_size = window_size
        self.is_test = is_test
        if(not is_test):
            self._index = [len(traj)-self.window_size+1 for traj in self.data_input]
        else:
            self._index = [1 for traj in self.data_input]

        self._traj_num = len(self.data_input)
        self.len = sum(self._index)


        print(self._index)
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        k = 0
        temp_id = index
        # raise ValueError("data Exceed {0} vs {1}".format(temp_id, self.len))
        # raise ValueError("data Exceed")
        if(temp_id >= self.len):
            raise ValueError("data Exceed {0} vs {1}".format(temp_id, self.len))

        for i, ref in enumerate(self._index):
            if temp_id<ref:
                k = i
                break
            else:
                temp_id = temp_id-ref
        if(not self.is_test):
            lt = self.window_size
        else:
            lt = len(self.data_input[k])-1
        
        
        x = self.data_input[k][temp_id:temp_id+lt]
        y = self.data_output[k][temp_id:temp_id+lt]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#is_outlier_removed True: remove outliers in this part; otherwise skip


class RegistrationData:
    def __init__(self, 
                 number_list:list = list(range(1,6)), 
                 names:list = "saved_state.csv",
                 is_outlier_removed:bool = False) -> None:
        self.data = {}
        for name in names:
            path_list = RegistrationData.read_paths_from_name(name, number_list)
            self.data[name] = RegistrationData.load_measurements(*path_list)
            # print(self.data[name][0][1])

        self.data_aug_dst,self._pq_dst,_ = self._data_augument(self.data, [0,3], [3,7], reverse="True")
        self.data_aug_src,self._pq_src, self.reproj_error = self._data_augument(self.data, [15,18], [18,22])
        # print(self.data_aug)
        self._twist = self._data_select(self.data, [7,13])


        """
        Here we only use "saved_states to handle all the information"
        """
        target_file_name = names[0]
        self._pq_dst = self._pq_dst[target_file_name]
        self._pq_src = self._pq_src[target_file_name]
        self.reproj_error = self.reproj_error[target_file_name]
        self._twist = self._twist[target_file_name]
        # print("self._pq_src = ",self._pq_src[0])


        # print("self._twist = ",len(self._twist))
        # RegistrationData.view_channels(self._twist_fromsensor[0], self._twist[0])
        # print("self._twist_fromsensor = ",self._twist_fromsensor[0][3])
        # print("self._twist_fromsensor = ",self._twist_fromsensor[0])
        # raise ValueError("RUN TO HERE")

        
        self.hdeye_cali_results = self._registration(self.data_aug_dst[target_file_name],self.data_aug_src[target_file_name])
        self.ground_truth = self._get_ground_truth()

        self._twist_fromsensor = self.getposediff_wrtEE(self._pq_src)

        # print("len(self._pq_dst) = ",len(self._pq_dst))
        # print("len(self.reproj_error) = ",len(self.reproj_error))
        # print("len(self.ground_truth) = ",len(self.ground_truth))


        """
        If you want tos see comparison of traj dst and ground truth, commit out this.
        """
        # RegistrationData.view_Cartesian_pose(self._pq_dst[0], self.ground_truth[0])
        # RegistrationData.view_channels(self._pq_dst[0], self.ground_truth[0])

        # print("Point 1 {0}".format(self._pq_dst[0][0]))
        # print("Point 1_gt {0}".format(self.ground_truth[0][0]))
        e_avgs = []
        e_maxs = []
        for i in range(len(self._pq_dst)):
            e_avg, e_max = RegistrationData.find_errors_of_two_traj(self._pq_dst[i], self.ground_truth[i],self.reproj_error[i])
            print(" e_avg, e_max = {0}   {1}".format(e_avg, e_max))
            e_avgs.append(e_avg)
            e_maxs.append(e_max)
        print("np.var(e_avgs) = ", np.var(e_avgs))
        print("np.var(e_maxs) = ", np.var(e_maxs))
        print("np.mean(e_avgs) = ", np.mean(e_avgs))
        print("np.mean(e_maxs) = ", np.mean(e_maxs))

        # print(" e_avg, e_max = {0}   {1}".format(e_avg, e_max))
        if(is_outlier_removed):
            # self._pq_dst = self.outlier_remove_DBSCAN(self._pq_dst)
            self._pq_dst = self.outlier_with_reference(self._pq_dst, self.ground_truth)

        self._hy_cali = self._get_transformation()

        # print("self._hy_cali = ", len(self._hy_cali))
        print("self.reproj_error = ",max(self.reproj_error[0]))
        print("self.reproj_error = ",min(self.reproj_error[0]))

        """
        If you want to se the distribution of reprojection error, you can commit in this.
        """
        # plt.plot(self.reproj_error[0])
        # plt.show()

    
    def generateIOput(self):
        """
        data loader is not very simple. considering every data is a trajectory as KF needs some time to stablize the states
        The most convincing method is sliding window. Use all the points in the window as a trajectory
        """
        _input = []
        _output = []
        for traj_dst, traj_err, traj_tst, traj_src in zip(self._pq_dst, self.reproj_error, self._twist_fromsensor,self.ground_truth):
            _input_traj = []
            _output_traj = []
            for _dst, _err, _tst, _src in zip(traj_dst, traj_err, traj_tst, traj_src):
                _input_traj.append(_dst + [_err] + _tst)
                _output_traj.append(_src)
            _input.append(_input_traj)
            _output.append(_output_traj)
        return _input, _output
    
    def generateDataLoader(self, window_size:int = 20, is_test:bool = False)-> TimeSeriesDataset:
        _input, _output = self.generateIOput()

        return TimeSeriesDataset(_input, _output, window_size=window_size, is_test = is_test)
    
    def getposediff(self, pose_list):
        _list = []
        for traj in pose_list:
            twist_list = [[0.0]*6]

            for i in range(1,len(traj)):
                _twist_v = (np.asarray(traj[i][0:3])- np.asarray(traj[i-1][0:3]))/T_step
                _twist_av = RegistrationData.quaternion_to_anglevel(traj[i][3:7], traj[i-1][3:7])*100.0

                twist_list.append(
                    _twist_v.tolist() + _twist_av.tolist()
                )
            _list.append(twist_list)
        return _list
    

    def getposediff_wrtEE(self, pose_list):
        _output = []
        _twist_src = self.getposediff(pose_list)
    
        for _src_traj, _rob_traj in zip(_twist_src, pose_list):
            poses = []
            for _src_point, _rob_point in zip(_src_traj, _rob_traj):
                p = _src_point[:3]
                q = _src_point[3:7]
                # T44 = RegistrationData.from_pq2T44(p,q)
                T44 = RegistrationData.from_pq2T44(_rob_point[:3],_rob_point[3:7])



                t = T44[:3,:3].T @ np.asarray(p)
                # print("t = ",t)
                pose = t.flatten().tolist() + q  #RegistrationData.from_T442pose(T_t_ee).flatten().tolist()
                poses.append(pose)

            _output.append(poses)

        return _output
            

            




    @staticmethod
    def quaternion_to_anglevel(quaternion,quaternionlst):
        _twist_temp =  np.asarray(quaternion)- np.asarray(quaternionlst)
        _twist_temp =  2.0*_twist_temp/T_step
        qt_star = RegistrationData.quaternion_conjugate(np.asarray(quaternionlst))
        q = RegistrationData.quaternion_multiply(_twist_temp, qt_star)
        wx, wy, wz = q[1], q[2], q[3]

        return np.array([wx, wy, wz])


    @staticmethod
    def quaternion_conjugate(quaternion):
        w, x, y, z = quaternion
        return np.array([w, -x, -y, -z], dtype=np.float64)
    

    @staticmethod
    def quaternion_multiply(quaternion0, quaternion1):
        w1, x1, y1, z1 = quaternion0
        w2, x2, y2, z2 = quaternion1

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z], dtype=np.float64)



    @ staticmethod
    def find_errors_of_two_traj(traj1, traj2, ws=None):
        e_max = 0.0
        e_sum = 0.0
        w_sum = 0.0
        if ws == None:
            ws = [1.0 for _ in traj1]
        es = []
        for p1, p2, w in zip(traj1, traj2, ws):
            e = np.linalg.norm(np.array(p1[:3])-np.array(p2[:3]))
            # print(p1)
            # print(e)

            if e > e_max:
                e_max = e
            e_sum += e*w
            w_sum += w
            es.append(e)

        # plt.plot(es)
        # plt.show()
        # print("average es", w_sum)
        # print("average es", len(traj1))
        # print("average es", e_sum)
        return e_sum/(w_sum), e_max

    # @ staticmethod
    # def find_mass_center_of_two_traj(traj1, traj2, ws=None):
    #     e_max = 0.0
    #     e_sum = 0.0
    #     w_sum = 0.0
    #     if ws == None:
    #         ws = [1.0 for _ in traj1]
    #     es = []
    #     for p1, p2, w in zip(traj1, traj2, ws):
    #         e = np.array(p1[:3])-np.array(p2[:3])

    #         es.append(e)

        


    #     return 
    
    # def view_every_channel(*measurements_list):
    #     # ax = plt.figure().add_subplot()
    #     # fig = plt.figure()
    #     # N_l=3
    #     # fig, axs = plt.plot()
    #     # ps = []
    #     for measurements in measurements_list:
    #         #len(measurements[0])
    #         # print(N_l)
            
    #         pos = []
    #         for mes in measurements:
    #             # for id in range(N_l):
    #             pos.append(mes)
    #         # ps.append(pos)

    #     # for _pos in ps:
    #         for (id,p) in enumerate(pos):
    #             # ax = fig.add_subplot(N_l, 1, id+1)
    #             plt.plot(p,
    #                     markevery=1)
    #         # ax.set_title('Actual Positions')
    #         # ax.set_xlabel('X (m)')
    #         # ax.set_ylabel('Y (m)')
    #         # ax.legend()
    #     # 显示图表
    #     plt.show()

    @ staticmethod
    def read_paths_from_name(name, number_list):
        return  [os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    str(i),
                    name,
                ) for i in number_list
            ]
    @ staticmethod
    def load_measurements(*paths):
        poses_list = []
        for path in paths:
            data = []
            with open(path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    data.append(row)

            poses = [[float(row[i]) for i in range(len(row))] for row in data]
            poses_list.append(poses)
        return poses_list
    @staticmethod
    def _refine_with_reprojection_error(data, err_list):

        # 获取前20%的元素数量
        num_top_20 = int(len(data) * accept_rate)

        # 对err_list进行排序，但保留原始索引
        sorted_indices = sorted(range(len(err_list)), key=lambda k: err_list[k], reverse=True)

        # 获取前20%的索引
        top_20_indices = sorted_indices[:num_top_20]

        # 使用这些索引来索引list2
        result = [data[i] for i in top_20_indices]
        return result
    
    def _registration(self,data_dst, data_src):
        result = []
        
        for src_tj, dst_tj in zip(data_src, data_dst):
            # print(src_tj)
            src = np.asarray(src_tj)[:, :3]
            dst = np.asarray(dst_tj)[:, :3]
            weight_ap =[-np.linalg.norm(dst_p) for dst_p in dst_tj]

            min_value = min(weight_ap)
            max_value = max(weight_ap)
            normalized_wap = [(x - min_value) / (max_value - min_value) *0.8 +0.2 for x in weight_ap]


            weight = np.asarray(dst_tj)[:, 3]
            # print("src", len(src))

            src = np.asarray(RegistrationData._refine_with_reprojection_error(src,weight))
            dst = np.asarray(RegistrationData._refine_with_reprojection_error(dst,weight))
            weight = np.asarray(RegistrationData._refine_with_reprojection_error(weight,weight))
            normalized_wap = np.asarray(RegistrationData._refine_with_reprojection_error(normalized_wap,weight))
            # print("src", len(src))

            # print(dst)


            T44 = RegistrationData.get_3d_transformation_matrix(dst[::SAMPLE_STEP,::1], src[::SAMPLE_STEP,::1], normalized_wap[::SAMPLE_STEP]*weight[::SAMPLE_STEP])
            result.append(T44)
        return result
            # print("T44 = ",T44)
            # print(dst)
    
    def _data_select(self, data:dict, _index:Tuple[int,int]):
        data_aug = {}
        for key, value in data.items():
            data_aug[key] = []
            for traj in value:
                traj_temp = []
                for point in traj:
                    d = point[_index[0]:_index[1]]
                    traj_temp.append(d)
                data_aug[key].append(traj_temp)
        return data_aug



        
    def _data_augument(self,data:dict, pos_index:Tuple[int, int], ori_index:Tuple[int, int], reverse:bool=False):
        data_aug = {}
        data_with_oris = {}
        data_re_proj_error = {}
        for key, value in data.items():
            data_aug[key] = []
            data_with_oris[key] = []
            data_re_proj_error[key] = []
            for traj in value:
                traj_temp = []
                traj_with_ori = []
                traj_re_proj_error = []
                for point in traj:
                    ori = point[ori_index[0]:ori_index[1]]
                    pos = point[pos_index[0]:pos_index[1]]

                    if(reverse):
                        T44 = RegistrationData.from_pq2T44(pos,ori)
                        T44i = np.linalg.inv(T44)
                        pos, ori = RegistrationData.from_T442pq(T44i)
                        pos = pos.flatten().tolist()
                        ori = ori.flatten().tolist()


                    a, b, c, d = RegistrationData.generate_points_in_xy_plane(pos, ori)
                    # print("a = ", a)
                    w = 1.0 - point[INDEX_FOR_REPROJ]*point[INDEX_FOR_REPROJ2]
                    # RegistrationData.Rerror2weight(point[INDEX_FOR_REPROJ])* RegistrationData.Rerror2weight(point[INDEX_FOR_REPROJ2])
                    traj_temp.append(pos+[w])
                    # traj_temp.append(a+[w])
                    # traj_temp.append(b+[w])
                    # traj_temp.append(c+[w])
                    # traj_temp.append(d+[w])
                    traj_with_ori.append(pos+ori)
                    traj_re_proj_error.append(w)
                # print(traj_temp[-1])
                data_aug[key].append(traj_temp)
                data_with_oris[key].append(traj_with_ori)
                data_re_proj_error[key].append(traj_re_proj_error)
        return data_aug, data_with_oris, data_re_proj_error
    
    @staticmethod
    def Rerror2weight(error, k= 1.0, x0= 0.55):
        w = 1.0 - 1.0/(1 + np.exp(-k*(error- x0)))
        return w

    @staticmethod
    def from_pq2T44(p:list,q:list):
        T44 = np.zeros((4,4))
        T44[3,3] = 1.0

        R = RegistrationData.quaternion_to_rotation_matrix(q)
        T44[:3,:3] = R
        T44[0,3] = p[0]
        T44[1,3] = p[1]
        T44[2,3] = p[2]
        return T44
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        将四元数转换为旋转矩阵

        参数：
        q: 四元数，表示为 [w, x, y, z]

        返回：
        R: 旋转矩阵，3x3的NumPy数组
        """
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        return R

    @ staticmethod
    def generate_points_in_xy_plane(p:list, q:list, dist:float=0.02):
        T44 = RegistrationData.from_pq2T44(p,q)
        # print("dist*T44[:3,0].reshape(-1,1) = ",dist*T44[:3,0].reshape(1,-1).flatten())
        dx = dist*T44[:3,0].reshape(1,-1).flatten()
        dy = dist*T44[:3,1].reshape(1,-1).flatten()
        a = np.asarray(p) + dx + dy
        b = np.asarray(p) - dx + dy
        c = np.asarray(p) - dx - dy
        d = np.asarray(p) + dx - dy

        return a.tolist(), b.tolist(), c.tolist(), d.tolist()
    
    @staticmethod
    def rotation_matrix_to_quaternion(R):
        qw = np.sqrt(1 + np.trace(R)) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        
        return np.array([qw, qx, qy, qz])
    
    @staticmethod
    def from_T442pq(T44):
        R = T44[:3,:3]

        q = RegistrationData.rotation_matrix_to_quaternion(R)
        p = np.array([0.0]*3)
        p[0] = T44[0,3]
        p[1] = T44[1,3]
        p[2] = T44[2,3]
        return p,q
    
    @staticmethod
    def from_T442pose(T44):
        p,q = RegistrationData.from_T442pq(T44)
        pose = np.array([0.0]*7)
        pose[:3] = p
        pose[3:] = q/np.linalg.norm(q)
        return pose

    def outlier_with_reference(self, _src, _ref):
        output = []
        for id,(sp, rp) in enumerate(zip(_src, _ref)):
            X_src = np.asarray(sp)
            X = X_src - np.asarray(rp)

            # 使用 DBSCAN 进行离群点检测
            dbscan = DBSCAN(eps=0.2, min_samples=20)  # eps 和 min_samples 参数需要根据你的数据进行调整
            clusters = dbscan.fit_predict(X)

            # 找到离群点的索引
            outlier_indices = np.where(clusters == -1)[0]

            # 处理离群点
            for index in outlier_indices:
                # 找到上一个和下一个非离群点的索引
                prev_index = next((i for i in range(index - 1, -1, -1) if clusters[i] != -1), None)
                next_index = next((i for i in range(index + 1, len(sp)) if clusters[i] != -1), None)
                
                # 如果找到了相邻的非离群点
                if prev_index is not None and next_index is not None:
                    # 计算平均值
                    new_value = np.mean([X_src[prev_index], X_src[next_index]], axis=0)
                    # 替换离群点的值
                    # print("X[index]  = {0}  to {1}".format(X[index], new_value))
                    X_src[index] = new_value
                elif prev_index is not None:
                    # 如果只找到上一个非离群点
                    X_src[index] = X_src[prev_index]
                elif next_index is not None:
                    # 如果只找到下一个非离群点
                    X_src[index] = X_src[next_index]

            # 将处理后的 NumPy 数组转回二维列表
            
            output.append(X_src.tolist())
        return output





    def outlier_remove_DBSCAN(self, _src):
        # 将数据转换为 NumPy 数组以便使用 DBSCAN
        # print("data",data)
        # print("data",len(data))
        output = []
        for id,data in enumerate(_src):
            X = np.array(data)

            # 使用 DBSCAN 进行离群点检测
            dbscan = DBSCAN(eps=0.2, min_samples=20)  # eps 和 min_samples 参数需要根据你的数据进行调整
            clusters = dbscan.fit_predict(X)

            # 找到离群点的索引
            outlier_indices = np.where(clusters == -1)[0]

            # 处理离群点
            for index in outlier_indices:
                # 找到上一个和下一个非离群点的索引
                prev_index = next((i for i in range(index - 1, -1, -1) if clusters[i] != -1), None)
                next_index = next((i for i in range(index + 1, len(data)) if clusters[i] != -1), None)
                
                # 如果找到了相邻的非离群点
                if prev_index is not None and next_index is not None:
                    # 计算平均值
                    new_value = np.mean([X[prev_index], X[next_index]], axis=0)
                    # 替换离群点的值
                    # print("X[index]  = {0}  to {1}".format(X[index], new_value))
                    X[index] = new_value
                elif prev_index is not None:
                    # 如果只找到上一个非离群点
                    X[index] = X[prev_index]
                elif next_index is not None:
                    # 如果只找到下一个非离群点
                    X[index] = X[next_index]

            # 将处理后的 NumPy 数组转回二维列表
            
            output.append(X.tolist())
        return output

    

    @staticmethod
    def get_3d_transformation_matrix(dst_, src_, weight):
        # 得到的坐标系是T parent=dst  child=src
        dst_points = dst_
        src_points = src_
        num_points = dst_points.shape[0]
        # print(dst_points)
        # print("num_points = {0}".format(num_points))
        # print("Done1")
        dst_points_mean = dst_points.sum(axis=0)/num_points
        src_points_mean = src_points.sum(axis=0)/num_points
        # print("src_points_mean = {0}".format(src_points_mean))
        centerDst = dst_points-dst_points_mean
        centerSrc = src_points-src_points_mean

        # print(centerDst)
        # print(centerSrc)

        dst_mat = np.mat(centerDst)
        src_mat = np.mat(centerSrc)

        matS = src_mat.transpose()*np.diag(weight)*dst_mat
        # print("src_mat = {0}".format(src_mat))
        # print("dst_mat = {0}".format(dst_mat))
        # print(matS)
        # print(centerSrc)
        # raise("Run to here")
        u,s,v = np.linalg.svd(matS)

        det = np.linalg.det(u*v)
        diag = [1,1,det]
        
        matM = np.diag(diag)

        
        matR = u*matM*v
        matt = np.mat(src_points_mean).transpose() - matR*(np.mat(dst_points_mean).transpose())    
        
        
        r_t = np.hstack((matR,matt))
        b = np.zeros(4)
        b[3] = 1.0
        R_T = np.vstack((r_t,b))


        # 四元数 x,y,z,w
        # return R_T
        return np.linalg.inv(R_T)
    
    def _get_ground_truth(self):
        _output = []
        # print("self._pq_src = ", len(self._pq_src))
        for _src_traj, T_hy_cali in zip(self._pq_src, self.hdeye_cali_results):
            poses = []
            for _src_point in _src_traj:
                p = _src_point[:3]
                q = _src_point[3:7]
                T44 = RegistrationData.from_pq2T44(p,q)


                T_t_ee = T_hy_cali @ T44
                pose = RegistrationData.from_T442pose(T_t_ee).flatten().tolist()
                poses.append(pose)

            _output.append(poses)
        return _output
    
    def _get_transformation(self):
        hy_output = []
        for _src_traj, _dst_traj in zip(self._pq_src, self._pq_dst):
            poses = []
            for _src_pose, _dst_pose in zip(_src_traj, _dst_traj):
                p_src = _src_pose[:3]
                q_src = _src_pose[3:7]

                p_dst = _dst_pose[:3]
                q_dst = _dst_pose[3:7]
                
                T_src = RegistrationData.from_pq2T44(p_src,q_src) #basement to robot
                T_dst = RegistrationData.from_pq2T44(p_dst,q_dst) #robot to trocar

                T_d_s = T_dst @ np.linalg.inv(T_src)
                pose = RegistrationData.from_T442pose(T_d_s).flatten().tolist()
                poses.append(pose)
            hy_output.append(poses)
        return hy_output

    
    @staticmethod
    def view_Cartesian_pose(*measurements_list):
        ax = plt.figure().add_subplot(projection='3d')
        for measurements in measurements_list:
            pos = [[] for i in range(7)]
            for mes in measurements:
                pos[0].append(mes[0])
                pos[1].append(mes[1])
                pos[2].append(mes[2])

            
            # for i in range(1,N+1):
            ax.plot(pos[0],pos[1],pos[2],
                    markevery=100)
            # axs[i].set_ylabel('Column {0}'.format(i))
        ax.set_title('Actual Positions')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax.legend()
        # 显示图表
        plt.show()

    @staticmethod
    def view_channels(*measurements_list):
        plt.figure()
        for i, measurements in enumerate(measurements_list):
            pos = [[] for i in range(7)]
            for mes in measurements:
                pos[0].append(mes[DISPLAY[0]])
                pos[1].append(mes[DISPLAY[1]])
                pos[2].append(mes[DISPLAY[2]])

            plt.subplot(3, 1, 1)  # 三行一列，当前激活的是第一个图
            plt.plot(pos[0])  # '-r' 表示红色实线
            # plt.legend(loc='traj'+str(i))

            plt.subplot(3, 1, 2)  # 三行一列，当前激活的是第一个图
            plt.plot(pos[1])  # '-r' 表示红色实线
            # plt.legend(loc='traj'+str(i))

            plt.subplot(3, 1, 3)  # 三行一列，当前激活的是第一个图
            plt.plot(pos[2])  # '-r' 表示红色实线
            # plt.legend(loc='traj'+str(i))
        # 自动调整子图间距
        plt.tight_layout()
        # 显示图形
        plt.show()




def main():
    name_src = "saved_state.csv"
    # name_dst = "pose_br.csv"
    instance = RegistrationData(number_list=list(range(1,7)),names=[name_src])
    data_loader = instance.generateDataLoader()
    print("data_loader = ", type(data_loader))

if  __name__ == "__main__":
    main()

