# Reinforcement Learning Algorithms: Comparative Study

This repository contains implementations and experiments for three reinforcement learning algorithms across multiple domains, both discrete and continuous. The goal is to evaluate algorithmic behavior, performance, and the impact of design choices and hyperparameters.

## Overview
We implemented and analyzed:
- **REINFORCE with Baseline**  
- **Prioritized Sweeping**  
- **One-Step Actor-Critic**

The experiments were conducted in environments such as:
- Cats vs. Monsters (discrete gridworld)  
- Inverted Pendulum (continuous control)  
- Mountain Car (continuous state, discrete action)  
- DynaMaze  

Results highlight trade-offs between stability, convergence speed, and sample efficiency.

## Algorithms

### 1. REINFORCE with Baseline
- Monte Carlo policy gradient method with variance reduction via a baseline.  
- Applied to **Cats vs. Monsters**, **Inverted Pendulum**, and **Mountain Car**.  
- Policy parametrized using tensors (discrete) or neural networks (continuous).  
- Key insight: stable learning requires tuning γ and learning rates.  

### 2. Prioritized Sweeping
- Model-based method focusing updates on state-action pairs with high priority.  
- Implemented on **Cats vs. Monsters**, **DynaMaze**, and a discretized **Inverted Pendulum**.  
- Results show faster convergence when exploration (ϵ) and learning rate (α) are balanced.  

### 3. One-Step Actor-Critic
- Combines policy gradients (actor) with TD(0) value updates (critic).  
- Applied to **Cats vs. Monsters**.  
- Uses neural networks to estimate policy and value functions.  
- Produces smoother, more incremental learning compared to Monte Carlo REINFORCE.  

## Environments
- **Cats vs. Monsters**: 5×5 gridworld with stochastic transitions, monsters, forbidden furniture, and terminal food state.  
- **Inverted Pendulum**: Continuous angle and velocity dynamics, torque control.  
- **Mountain Car**: Continuous state (position, velocity), discrete actions (left, right, none).  
- **DynaMaze**: Deterministic grid maze with obstacles and terminal goal.  

## Results
- **REINFORCE with Baseline**: Converged to near-optimal policies but showed instability in early episodes.  
- **Prioritized Sweeping**: Significantly reduced training time in discrete domains. Struggled in continuous settings.  
- **One-Step Actor-Critic**: Balanced stability and efficiency, learning incrementally with TD updates.  

📄 [Full Project Report](report/RL_project_report.pdf)

## Implementation Details
- Frameworks: Python, NumPy, PyTorch (for neural network policies & baselines).  
- Optimization: Adam optimizer for policy and value updates.  
- Training: Multiple runs for averaging and stability analysis.  

