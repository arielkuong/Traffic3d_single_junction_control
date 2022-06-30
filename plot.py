import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path = args.save_dir + '/DQN/episode_total_rewards.npy'
    eval_file_path_1 = 'episode_total_rewards.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)
        data1 = np.load(eval_file_path_1)
        print(data)

        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Reward total', fontsize=16)
        # plt.title(args.env_name, fontsize=20)

        plt.plot(x, data, color='red', linewidth=2, label='DQN')
        plt.plot(x1, data1, color='blue', linewidth=2, label='Vanilla PG')
        plt.legend(loc='lower right')
        plt.show()
