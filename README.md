# TD3
This is a concise Pytorch implementation of TD3(Twin Delayed DDPG) on continuous action space.<br />


## How to use my code?
You can dircetly run 'TD3.py' in your own IDE.<br />

### Trainning and evaluating
python TD3.py

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />

## Reference
[1] Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]//International conference on machine learning. PMLR, 2018: 1587-1596.<br />
