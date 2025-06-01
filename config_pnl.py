ppo_hyperparameters = [
    {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
},
    {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
},
    {
    "n_steps": 1024,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "batch_size": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
},
    {
    "n_steps": 4096,
    "ent_coef": 0.01,
    "learning_rate": 5e-4,
    "batch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.05,
}
]

ddpg_hyperparameters = [
    {"learning_rate": 1e-3,
     "buffer_size": 100_000,
     "batch_size": 64,
     "tau": 0.005,
     "gamma": 0.99,
     "train_freq": 1,
     "gradient_steps": 1},

    {"learning_rate": 3e-4,
     "buffer_size": 300_000,
     "batch_size": 128,
     "tau": 0.005,
     "gamma": 0.98,
     "train_freq": 1,
     "gradient_steps": 1},
    
    {"learning_rate": 5e-4,
     "buffer_size": 50_000,
     "batch_size": 256,
     "tau": 0.01,
     "gamma": 0.95,
     "train_freq": 2,
     "gradient_steps": 1},
         
    {"learning_rate": 7e-4,
     "buffer_size": 150_000,
     "batch_size": 256,
     "tau": 0.005,
     "gamma": 0.99,
     "train_freq": 1,
     "gradient_steps": 1}
]

a2c_hyperparameters = [
    {"n_steps": 5,
     "learning_rate": 0.0007,
     "gamma": 0.99,
     "gae_lambda": 0.95,
     "ent_coef": 0.01,
     "vf_coef": 0.5},

     {"n_steps": 20,
     "learning_rate": 3e-4,
     "gamma": 0.99,
     "gae_lambda": 0.95,
     "ent_coef": 0.01,
     "vf_coef": 0.5},

     {"n_steps": 10,
     "learning_rate": 1e-3,
     "gamma": 0.98,
     "gae_lambda": 0.93,
     "ent_coef": 0.01,
     "vf_coef": 0.5},

     {"n_steps": 64,
     "learning_rate": 5e-4,
     "gamma": 0.99,
     "gae_lambda": 0.97,
     "ent_coef": 0.01,
     "vf_coef": 0.5},
]

sac_hyperparameters = [
    {"learning_rate": 3e-4,
     "batch_size": 256,
     "tau": 0.005,
     "gamma": 0.99,
     "train_freq": 1,
     "gradient_steps": 1},

     {"learning_rate": 1e-4,
     "batch_size": 512,
     "tau": 0.002,
     "gamma": 0.99,
     "train_freq": 1,
     "gradient_steps": 1},

     {"learning_rate": 5e-4,
     "batch_size": 128,
     "tau": 0.01,
     "gamma": 0.98,
     "train_freq": 1,
     "gradient_steps": 2},

     {"learning_rate": 1e-4,
     "batch_size": 256,
     "tau": 0.01,
     "gamma": 0.99,
     "train_freq": 1,
     "gradient_steps": 1},
]

algorithms = {
    "ppo": ppo_hyperparameters,
    "ddpg": ddpg_hyperparameters,
    "a2c": a2c_hyperparameters,
    "sac": sac_hyperparameters
}
