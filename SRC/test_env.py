import environment
from environment import DPEnvironment

env = DPEnvironment(data_path="Data/Processed/processed_data_train.csv")

print("Reset method being used", env.reset)
obs, info = env.reset()
print("Initial Observation:", obs)
print("Initial Info:", info)

for step in range(10):
    action = env.action_space.sample()  #Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Action={action}, Observation={obs}, Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
    
    if terminated or truncated:
        print("Episode ended. Resetting environment.")
        obs, info = env.reset()