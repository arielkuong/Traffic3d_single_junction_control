import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os
import pandas as pd
import csv

if __name__ == "__main__":

    args = get_args()
    # eval_file_path = args.save_dir + '/DQN_complex_light/episode_total_rewards_dualdirectioncontrol.npy'
    eval_file_path_1 = args.save_dir + '/DQN_complex_light/episode_total_rewards_singledirectioncontrol.npy'
    # eval_file_path_2 = args.save_dir + '/DQN_complex_light/episode_total_rewards_singledirectioncontrol_withpedestrian.npy'
    # eval_file_path_3 = args.save_dir + '/DQN_complex_light/episode_total_rewards_dualdirectioncontrol_withpedestrian.npy'
    eval_file_path_4 = args.save_dir + '/DQN_complex_light/episode_total_rewards_singledirectioncontrol_statereload.npy'

    # CSVData = pd.read_csv('/home/CAMPUS/180205312/traffic3d-develop/Traffic3D/Assets/Results/TrueRewards.csv')
    # traffic3d_rewards = np.genfromtxt(CSVData, delimiter=",")
    # traffic3d_rewards = traffic3d_rewards[:traffic3d_rewards.shape[0]-1]
    # traffic3d_rewards = traffic3d_rewards.astype(int)
    # traffic3d_ep_rewards = np.add.reduceat(traffic3d_rewards, np.arange(0, len(traffic3d_rewards), 100))
    # print(traffic3d_rewards)
    # print(traffic3d_rewards.shape)
    # print(traffic3d_ep_rewards)
    # print(traffic3d_ep_rewards.shape)
    # data_sumo = traffic3d_ep_rewards[:traffic3d_ep_rewards.shape[0]-1]

    # data = np.load(eval_file_path)[:35]
    data1 = np.load(eval_file_path_1)[:35]
    # data2 = np.load(eval_file_path_2)[:35]
    # data3 = np.load(eval_file_path_3)[:35]
    data4 = np.load(eval_file_path_4)[:35]
    # print(data)

    # x = np.linspace(0, len(data), len(data))
    x1 = np.linspace(0, len(data1), len(data1))
    # x2 = np.linspace(0, len(data2), len(data2))
    # x3 = np.linspace(0, len(data3), len(data3))
    x4 = np.linspace(0, len(data4), len(data4))

    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Reward total', fontsize=16)
    # plt.title(args.env_name, fontsize=20)

    # plt.plot(x, data, color='red', linewidth=2, label='DQN, dual direction control')
    plt.plot(x1, data1, color='red', linewidth=2, label='DQN, single direction control')
    # plt.plot(x2, data2, color='red', linewidth=2, label='DQN, single direction control')
    # plt.plot(x3, data3, color='blue', linewidth=2, label='DQN, dual direction control')
    plt.plot(x4, data4, color='blue', linewidth=2, label='DQN, single direction control, state preload')
    plt.legend(loc='lower right')
    plt.show()
