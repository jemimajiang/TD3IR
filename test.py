from TD3 import TD3
import numpy as np
import cv2
from ImageEnvTest import ImageDenoisingEnv


def evaluate_policy(env, agent, total_steps):
    times = 1  # Perform three evaluations and calculate the average
    len = 5
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        ht_1 = np.zeros(s.shape)
        episode_reward = 0
        for epi_step in range(len):
            a,h_t = agent.choose_action(s,ht_1)  # We do not add noise when evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
            ht_1 = h_t

            # 保存结果s
            save_img = np.clip(s, 0, 1)
            save_img = (save_img[0] * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(f"runs/res/test_{total_steps:04d}_len_{epi_step:01d}.png", save_img)
        evaluate_reward += episode_reward

    return np.floor(evaluate_reward / times)
if __name__ == '__main__':

    env = ImageDenoisingEnv()
    env_evaluate = ImageDenoisingEnv()
    number = 1

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    max_action = float(8)
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = TD3(state_dim, action_dim, max_action)

    # 载入权重文件
    loaded = np.load("/sdb/jm/DRL-code-pytorch-main/7.TD3/data_train/TD3_envImage_number_690.npy",allow_pickle=True)
    total_steps = 690
    print("loaded:")
    print(loaded)
    agent.model = loaded.tolist()
    # 将权重加载到代理模型中
    # agent.load_weights(weights)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    # total_steps = 0  # Record the total steps during the training

    s = env.reset()
    total_steps += 1
    ht_1 = np.zeros(s.shape)
    print("total_steps: {}".format(total_steps))

    evaluate_reward = np.mean(evaluate_policy(env_evaluate, agent, total_steps))
    evaluate_rewards.append(evaluate_reward)
    print("\tevaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

