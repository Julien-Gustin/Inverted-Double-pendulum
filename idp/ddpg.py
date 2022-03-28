from utils import J
from replay import ReplayBuffer
from copy import deepcopy
import torch 
import gym 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DDPG():
    def __init__(self, env, critic, actor, gamma=0.95, tau=0.95, batch_size=100, replay_buffer_size=int(1e6), episodes=500, steps=100, noise=0.1):
        self.env = env 
        self.critic = critic.to(device)
        self.actor = actor.to(device)
        self.gamma = gamma
        self.tau = tau 
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.episodes = episodes
        self.steps = steps 
        self.target_critic = deepcopy(critic).to(device)
        self.target_actor = deepcopy(actor).to(device)
        self.noise = noise

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        #These networks must be updated using exponential averaging w.r.t the critic network and the actor policy, not through the gradients
        for param in self.target_critic.parameters():
            param.requires_grad = False 
        for param in self.target_actor.parameters():
            param.requires_grad = False 

    def critic_loss(self, batch):
        """
        Computes the loss of the critic network
        """
        states, actions, rewards, new_states, done = batch['states'].to(device), batch['actions'].to(device), batch['rewards'].to(device), batch['new_states'].to(device), batch['done'].to(device)
        with torch.no_grad():
            #Select current believe of which action maximises the cumulative reward
            self.target_actor.eval()
            actor_actions = self.target_actor(new_states)
            self.target_actor.train()
            #Input of the target critic network
            x = torch.cat((new_states, actor_actions), 1)

        #Bellman equation
        y = rewards + (1-done)*self.gamma*self.target_critic(x)
        #Input of the critic network
        x = torch.cat((states, actions), 1)
        #compute and return the MSE loss 
        return ((y-self.critic(x))**2).mean()

    def actor_loss(self, batch):
        """
        Computes the loss of the actor network
        """
        states = batch['states'].to(device)
        actor_actions = self.actor(states)
        with torch.no_grad():
            #Input of the critic network
            x = torch.cat((states, actor_actions), 1)
        #Compute and return the loss 
        return -self.critic(x).mean()

    def choose_action(self, state):
        """
        Chooses an action using the actor network and by applying some noise to explore 
        """
        self.actor.eval()
        t_state = torch.Tensor(state).unsqueeze(0).to(device)
        action = self.actor(t_state).detach().to("cpu")
        noise = self.noise * torch.randn(action.shape)
        action += noise
        self.actor.train()
        return torch.clip(action, -1.0, 1.0).numpy()

    def compute_optimal_actions(self, states):
        """
        Choose an action using the actor network
        """
        self.actor.eval()
        states = np.array(states)
        t_state = torch.Tensor(states).to(device)
        action = torch.clip(self.actor(t_state), -1.0, 1.0).detach().to("cpu").numpy()
        self.actor.train()
        if len(action) == 1:
            return action[0]

        return action

    def update_networks(self, batch):
        """
        Performs one step of gradient descent for the critic network and the actor network
        """
        #Update critic network using gradient descent
        self.critic_optimizer.zero_grad()
        closs = self.critic_loss(batch)
        closs.backward()
        self.critic_optimizer.step()

        #Compute the loss for the actor
        self.actor_optimizer.zero_grad()
        aloss = self.actor_loss(batch)
        aloss.backward()
        self.actor_optimizer.step()

        return closs.detach().to("cpu"), -aloss.detach().to("cpu")

    def update_target_networks(self):
        """
        Performs exponential averaging on the parameters of the target networks, with respect to the parameters of the networks
        """
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.mul_(self.tau)
                target_param.data.add_(param.data*(1-self.tau))
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.mul_(self.tau)
                target_param.data.add_(param.data*(1-self.tau))

    def apply(self):
        """
        Train the agent in an online manner
        """
        #Initialize the replay buffer 
        replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        #Algorithm
        for i in range(self.episodes):
            #Initialize a new starting state for the episode
            current_state = self.env.reset()
            
            #Keep track of losses
            critic_losses = []
            actor_losses = []
            for _ in range(self.steps):
                #make a step in the environment
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)

                #store the transition in the replay buffer
                replay_buffer.store((current_state, action, reward, new_state, done))

                #update current state
                current_state = self.env.reset() if done else new_state

                #Sample a batch of size batch_size
                batch = replay_buffer.minibatch(self.batch_size)

                #Perform one step of gradient descend for the networks
                closs, aloss = self.update_networks(batch)
                
                #Remember the losses 
                critic_losses.append(closs)
                actor_losses.append(aloss)

                #Update target networks
                self.update_target_networks()

            avg_critic_loss = torch.mean(torch.Tensor(critic_losses))
            avg_actor_loss = torch.mean(torch.Tensor(actor_losses))
            j = J(self.env, self, self.gamma, 50, 1000)
            print("Episode {}: Critic: {} | Actor: {} | J: {}".format(i+1, avg_critic_loss, avg_actor_loss, j))





