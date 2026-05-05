from stable_baselines3 import DQN
from environment import DPEnvironment
import os

seed = 100
total_timesteps = 50000


# Hyperparameter tuning for DQN Agent: Baseline model, Low Learning Rate, High Gamma, Low Exploration
configs =[

    {"name": "base_dqn_model",
     "learning_rate": 5e-4,
     "gamma": 0.98,
     "exploration_fraction": 0.3,
     "exploration_initial_eps": 1.0,
     "exploration_final_eps": 0.05},

    {"name": "low_lr_dqn_model",
       "learning_rate": 1e-4,
       "gamma": 0.98,
       "exploration_fraction": 0.3,
       "exploration_initial_eps": 1.0,
       "exploration_final_eps": 0.05},

    {"name": "high_gamma_dqn_model",
       "learning_rate": 5e-4,
       "gamma": 0.99,
       "exploration_fraction": 0.3,
       "exploration_initial_eps": 1.0,
       "exploration_final_eps": 0.05},

    {"name": "long_exploration_dqn_model",
         "learning_rate": 5e-4,
         "gamma": 0.98,
         "exploration_fraction": 0.5,
         "exploration_initial_eps": 1.0,
         "exploration_final_eps": 0.1},
]

# Check for models folder and create if needed
os.makedirs("models", exist_ok=True)


# Training pipeline for DQN agents
for config in configs:
    print(f"Training {config['name']}...")
    
    env = DPEnvironment(data_path="Data/Processed/processed_data_train.csv")
    
    model = DQN(policy = "MlpPolicy",
        env = env,
        learning_rate = config["learning_rate"],
        gamma = config["gamma"],
        buffer_size = 20000,
        learning_starts = 2000,
        batch_size = 64,
        train_freq = 4,
        target_update_interval = 1000,
        exploration_fraction = config["exploration_fraction"],
        exploration_initial_eps = config["exploration_initial_eps"],
        exploration_final_eps = config["exploration_final_eps"],
        verbose=1,
        seed=seed
    )

    # Save models to created directory
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{config['name']}.zip")
    print(f"Finished training {config['name']}.\n")