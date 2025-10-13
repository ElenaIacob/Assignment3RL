Strategy 1: Double DQN with n-Step Returns + Curriculum Learning

This strategy builds on a standard DQN by combining three improvements:

	1.	Double DQN logic – uses two networks (online + target) to avoid Q-value overestimation and stabilize training.
	2.	n-Step Returns (n = 5) – lets rewards travel several steps forward, helping the agent learn from delayed feedback.
	3.	Curriculum Learning – starts with shorter poles (easier), then gradually increases pole length as training progresses, so the agent generalizes across different dynamics.


The goal of this strategy is to train an agent that generalizes across different pole lengths in the CartPole-v1 environment, rather than overfitting to the default setup.

Training Setup

	•	Environment: CartPole-v1 (Gymnasium)
	•	Framework: PyTorch
	•	Steps: 800 000
	•	Replay Buffer: 200 000 transitions
	•	Batch Size: 64
	•	Learning Rate: 5 × 10⁻⁴
	•	Epsilon Decay: 300 000 steps
	•	Target Update: every 1 000 steps
	•	n-Step: 5
	•	Grad Clip: 8.0

Training is step-based, so ε decays smoothly and the agent gradually shifts from exploration to exploitation.

How It Works

	1.	The online network predicts Q-values and selects actions.
	2.	The target network evaluates those actions (this separation keeps learning stable).
	3.	Each experience (state, action, reward, next_state, done) is stored in a replay buffer and randomly sampled to break temporal correlation.
	4.	n-step bootstrapping makes learning faster and more robust to delayed rewards.
	5.	Curriculum bounds expand as training goes on: first short poles → then medium → then full range (0.4 – 1.8 m).

After ~800 k steps, the model reached consistent performance across all pole lengths, achieving an average score around 340 – 360.
