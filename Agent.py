# # <---------------------------- Build the DDPG Agent ---------------------------->

import numpy as np
import copy
from collections import deque, namedtuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


# <------------- Define Hyperparameters -------------------->

BUFFER_SIZE       = int(1e6)  
BATCH_SIZE        = 128        
GAMMA             = 0.99            
TAU               = 1e-3              
LR                = 1e-3               
WEIGHT_DECAY      = 0        
LEARN_EVERY       = 20        
LEARN_NUM         = 10          
OU_SIGMA          = 0.2          
OU_THETA          = 0.15         
EPSILON           = 1.0           
EPSILON_DECAY     = 1e-6


# <----------Check GPU availability -------------->
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# <-------------------Define the Agent Class ------------------->
class Agent():
    """
    The Agent interact with and learn from the Environment
    """
    
    def __init__(self, state_size, action_size, random_seed):
        """
        Initialize Agent Object.
        ----------------------------------------
        Parameters
        ----------------------------------------
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        random_seed(int): random seed
        
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        
        # <------ Actor Network ----------->
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        
        # <----- Critic Network ---------->
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        #<------ Noise ---------->
        self.noise = OUNoise(action_size, random_seed)
        
        # <----------- Replay Memory ------------->
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
    def step(self, state, action, reward, next_state, done, timestep):
        """
        Save Experience in Replay Memory and select randomly from the buffer to learn """
        
        # <-------Save Experience --------->
        self.memory.add(state, action, reward, next_state, done)
        
        # <-------- Learn at given interval, if enough samples are available in the memory ----------->
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
                
    # <-------Obtain Action------------>            
    def act(self, state, add_noise=True):
        """
        Returns Actions for given state based on the current Policy
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)
    
    
    def reset(self):
        self.noise.reset()
       
    
    # <------Update the Policy and Value Parameters with selected Experiences ---->
    def learn(self, experiences, gamma):
        """
        Update the policy and Value Parameters using given batch of experience tuples
        Q_targets = r + y * critic_target(next_state, actor_target(next_state))
        
        actor_target(state)          --- Action
        critic_target(state, action) --- Q-Value
        
        -----------------------------------------    
        Parameters
        -----------------------------------------
        experiences (Tuple[torch.Tensor]) -- tuple(s,a,r,s',done)
        gamma (float)                     -- discount factor
        
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        # <----------------------- Update the Critic -------------------->
        
        #Get predicted next-state actions and Q-values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        #Compute Q targets for current states (y_1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        #Compute Critic Loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        #Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # <----------------------- Update the Actor -------------------->
        
        #Compute the Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        #Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # <----------------------- Update the Target Networks -------------------->
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        # <----------------------- Update the noise -------------------->
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()
        
     
        # <----------------------- Perform Soft Update -------------------->
    def soft_update(self, local_model, target_model, tau):
        """
        Soft Update model parameters
        
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        ---------------------------
        Parameters
        ---------------------------
        mu: long-running mean
        theta: speed of mean reversion
        sigma: volatility parameter
        
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param + (1.0-tau)*target_param.data)


#<------------------------------------Define OUNoise Class ---------------------->
class OUNoise:
    """
    Ornstein-Uhlenbeck process
    
    """
    
    def __init__(self, size, seed, mu=0., theta = OU_THETA, sigma = OU_SIGMA):
        
        """
        Parameters:
        ------------------------------------------
        mu: long-running mean
        theta: speed of mean reversion
        sigma: Volatility Paramter
        ------------------------------------------
        """
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
        
    def reset(self):
        """
        Update internal state (= noise to mean (mu)
        """
        self.state = copy.copy(self.mu)
    
    
    # <-------- Update Internal State ----------->
    def sample(self):
        """
        Update internal state and return it as a noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
    
    
# <--------------------Replay Buffer Class ----------------->
class ReplayBuffer:
    """
    Fixed size buffer that can store experience tuples
    """
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize the Replay buffer object
        --------------------
        Parameters
        --------------------
        buffer_size (int) --- maximum size of buffer
        batch_size (int)  --- size of each training batch
        
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    
    # <--------------------- Add new experience tuple to memory ------------------>
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experiencce tuple to memory
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    
    # <------------- Select Random Batch of Experience --------------------->
    def sample(self):
        """
        Sample a batch of experiences randomly from memory
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    
    def __len__(self):
        """
        Return the Current size of internal memory
        """
        return len(self.memory)