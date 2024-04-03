# A script to test the reward model

# load the torch model 
import torch
from src.reward_training import RewardModel
from RLHFenvironment import RLHFEnv

mode = "env"

if mode == "reward":
    # Create an instance of the model
    model = RewardModel()

    model.load_state_dict(torch.load('models/reward_model.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Create a tensor for a single state-action pair
    test1 = torch.tensor([[20.0, 20.0, 0.0, 1.0]])
    test2 = torch.tensor([[20.0, 20.0, 0.0, -1.0]])
    test4 = torch.tensor([[20.0, 20.0, -1.0, 0.0]])
    test3 = torch.tensor([[20.0, 20.0, 1.0, 0.0]])

    # Run the model
    output1 = model(test1[0][0].unsqueeze(0), test1[0][1].unsqueeze(0), test1[0][2].unsqueeze(0), test1[0][3].unsqueeze(0))
    output2 = model(test2[0][0].unsqueeze(0), test2[0][1].unsqueeze(0), test2[0][2].unsqueeze(0), test2[0][3].unsqueeze(0))
    output3 = model(test3[0][0].unsqueeze(0), test3[0][1].unsqueeze(0), test3[0][2].unsqueeze(0), test3[0][3].unsqueeze(0))
    output4 = model(test4[0][0].unsqueeze(0), test4[0][1].unsqueeze(0), test4[0][2].unsqueeze(0), test4[0][3].unsqueeze(0))

    print(output1)
    print(output2)
    print(output3)
    print(output4)

else:
    # create rlhf env
    env = RLHFEnv()
    state = env.reset()
    print(state)

    # test the step function
    state, reward, done, info = env.step(0)
    print(state, reward, done, info)
    