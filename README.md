# Car-Racing-V3-DQN
This project is a deep q-learning solution to the Car-Racing-V3 gymnasium game. This covers reward shaping, state augmentation, action space discretization, and rollout as part of the solution. The environment is getting old and won't let me change the track's winding direction during training. This makes the agent struggle with the few right turns that are generated in the mostly left track. I will try this again with the multi-car wrapper at https://github.com/igilitschenski/multi_car_racing. I will also try with Deep Deterministic Policy Gradient (DDPG) learning in the future. 

Check out my Cart-Pole solution at https://github.com/DyllonDunton1/Cart-Pole-DQN.


# Demo Comparing Base Policy to Truncated Rollout
![demo](https://github.com/user-attachments/assets/60e138ec-f725-455e-bd68-a9f361e67428)
