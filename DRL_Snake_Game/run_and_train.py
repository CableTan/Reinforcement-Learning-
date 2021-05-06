import torch
import RL_Snake
from RL_Snake.Agent import DQNAgent

if __name__ == "__main__":

    board_size = (15,15)
    # dqn_agent = RL_Snake.DQNAgent(board_size, path=None)

    PATH = r'C:\Users\PyCharm Community Edition 2021.1.1\CodeList\DRL_Snake\model\policy_net-200.pth'
    dqn_agent = RL_Snake.DQNAgent(board_size,path = PATH)

   # net load_state_dict(torch.load(PATH,map_location=lambda storage, loc: storage))

     # None

    RL_Snake.run_gui_game(board_size, dqn_agent)


"""
# 直接加载模型
model.load_state_dict(torch.load('./data/my_model.pkl'))
 
#GPU训练的模型加载到CPU上：
model.load_state_dict(torch.load('./data/my_model.pkl', map_location=lambda storage, loc: storage))
 
#加载到GPU1上：
model.load_state_dict(torch.load('./data/my_model.pkl', map_location=lambda storage, loc: storage.cuda(1)))
 
#从GPU1 移动到 GPU0：
model.load_state_dict(torch.load('./data/my_model.pkl', map_location={'cuda:1':'cuda:0'}))
"""