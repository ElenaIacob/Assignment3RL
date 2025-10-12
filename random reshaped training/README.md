# Differences Between the Two Random Reshaped Reward Scripts

- The new version adds GPU support using torch.device("cuda" if torch.cuda.is_available() else "cpu"), while the original only runs on CPU.
- The new version moves all networks and tensors to the GPU for faster training.
- The new version optimizes every 4 steps instead of every step, improving performance.
- The new version saves results to a /results/ folder instead of /weights/.
- The new version saves model weights (.pth), reward data (.xlsx), and a plot (.png) instead of only weights.
- The new version logs progress every 20 episodes instead of every 10.
- The core DQN logic, reward shaping function, and epsilon decay remain unchanged.