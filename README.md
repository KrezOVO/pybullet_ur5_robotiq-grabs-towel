# UR5 Robotic Arm Towel Folding Simulation

This project implements towel folding simulations using a UR5 robotic arm in PyBullet. The simulations include folding a towel along its diagonal and along one edge. The project provides Python scripts to set up the simulation environment, control the UR5 arm's movements, and execute the folding sequences.

## Project Structure

- **agent.py**: Defines the folding strategies and controls the robotic arm's actions for both diagonal and edge folding.
- **env.py**: Sets up the PyBullet simulation environment, including the UR5 arm and towel models.
- **kinematic.py**: Implements kinematics calculations for the UR5 arm, allowing precise control of the end-effector's position and orientation.
- **main_1.py**: Main script to run the diagonal folding simulation.
- **main_2.py**: Main script to run the edge folding simulation.
- **robot.py**: Configures and initializes the UR5 arm's joint positions and movements.
- **utilities.py**: Provides utility functions for error correction and coordinate transformations.

## Installation

### Prerequisites

1. **Python 3.7+**
2. **PyBullet**: Install using the command below:

   ```bash
   pip install pybullet
