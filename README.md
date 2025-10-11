# 🐾 My Hexapod Walk RL Project (Isaac Lab 5.0)

## 🚀 Overview
This project implements a **reinforcement learning tripod-gait controller** for a 12-DOF hexapod robot using **Isaac Lab 5.0**.  
It combines a **predefined tripod reference gait** with a **PPO-based policy** that learns adaptive offsets for smoother, faster, and more stable locomotion.

### ✨ Key Features
- 🧩 Modular Isaac Lab extension (runs outside the core repo)
- 🤖 12-joint hexapod USD model
- ⚙️ Tripod reference gait + RL offset hybrid control
- 🧠 PPO training via `rsl_rl`
- 💡 Easy tuning for rewards, observations, and physics (Sim-to-Real-ready)

---

## 🛠️ Installation

### 1️⃣ Install Isaac Lab 5.0
Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 2️⃣ Clone this repository
> Keep it **outside** the Isaac Lab source tree.

```bash
git clone https://github.com/<your-user>/my_hexapod_walk_rl.git
cd my_hexapod_walk_rl
