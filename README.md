# Reinforcement-Learning
Academic Exercise to Explore Reinforcement Learning with Neural Networks

library Structure
-----------------


    /reflrn
    
The library that defines the principles objects that compose a general framework for building simple RL use cases.

It is structured around a set of interfaces:

### `Environment.py` ###
 * Is the root construct, it has `state` space and rewards
 * Supports the injection of one or more `agents` that can experience one or more episodes.
 
### `Agent.py` ###
* Is the entity that learns
* Selecting actions according to a `policy` and learning to refine the policy based on teh rewards experienced.

### `Policy.py` ###
* Takes state, action rewards and learns to select future actions that will maximise the rewards for am agent in a 
given environment

### `State.py` ###
* An environment specific representation visible to the agent
 
### `ExplorationStrategy.py` ###
* A strategy for selecting actions where deviation from the action predicted by the current policy is required such 
that the agent can experience more of the environment.

### `Model.py` ###
* The Neural Network that is learning the state to action optimisation.

### `ReplayMemory.py` ###
* A full history of all state-action-reward events


    /examples
    
Different types of environment in which to create agents.

    /examples/gridworld
    
An environment of 2D grids, where each grid cell can be assigned a reward. Episodes are modeled by having a specific 
reward that is taken to be a terminal goal when found. The actions are environment specific but in simple case are 
just North, South, East, West.

    /examples/tictactoe
    
A two agent (adversarial) environment playing the simple game of TicTacToe.

