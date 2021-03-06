
# Project - Continous Control

#### Deep Reinforcement Learning Continous Control Project 

<table class="unchanged rich-diff-level-one">
  <thead><tr>
      <th align="left">Reacher Environment</th>
      <th align="left">Robotic Arms</th>
  </tr></thead>
  <tbody>
    <tr>
      <td align="left"><img src="./images/20_Agents.gif" alt="first-view" style="max-width:100%;"></td>
      <td align="left"><img src="./images/robotic_arm_agents.gif" alt="top-view" style="max-width:100%;"></td>
    </tr>
  </tbody>
</table>

### Environment

For this project, I worked with the Reacher environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, I solved the 2nd version of the Unity environment: <br>
The 2nd version contains 20 identical agents, each with its own copy of the environment.

In order to solve the environment, the agents must achieve an average score of +30 (over 100 consecutive episodes, and over all agents).

### Environment Setup

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in ```README.md``` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this complete the project.

>(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

For this project, you will not need to install Unity - this is [Udacity](https://udacity.com) already built the environment, and you can download it from one of the links below. You need only select the environment that matches your operating system:

**Version 2: Twenty (20) Agents**
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above._)

### Run the Notebok

After you have followed the instructions above carefully, you are now ready to run the my notebook.

You can also choose to run the project on Udacity Workspace, but the provided workspace doesn't allow you to use simulator. You can use the simulator locally to watch the agent.

- You can either use the pretrained weights or 
- Run the ```Continuous_Control.ipynb``` 


```python

```
