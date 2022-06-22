from models.utils.expected_return import J
from models.utils.replay import ReplayBuffer
from copy import deepcopy

import torch 
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPG():
    def __init__(self, env, critic, actor, exploration, file_extension, gamma=0.99, tau=0.999, batch_size=64, replay_buffer_size=int(1e6), episodes=500, steps=1000, nb_simulation=50):
        self.env = env 
        self.critic = critic.to(device)
        self.actor = actor.to(device)
        self.target_critic = deepcopy(critic).to(device)
        self.target_actor = deepcopy(actor).to(device)

        self.gamma = gamma # discount factor
        self.tau = tau 
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.episodes = episodes
        self.steps = steps 

        self.exploration = exploration
        self.file_extension = file_extension
        self.nb_simulation = nb_simulation

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        #These networks must be updated using exponential averaging w.r.t the critic network and the actor policy, not through the gradients
        for param in self.target_critic.parameters():
            param.requires_grad = False 
        for param in self.target_actor.parameters():
            param.requires_grad = False 

    def critic_loss(self, batch):
        """
        Computes the loss of the critic network
        """
        with torch.no_grad():
            states, actions, rewards, new_states, done = batch['states'].to(device), batch['actions'].to(device), batch['rewards'].to(device), batch['new_states'].to(device), batch['done'].to(device)
            #Select current believe of which action maximises the cumulative reward
            self.target_actor.eval()
            actor_actions = self.target_actor(new_states)
            self.target_actor.train()
            #Bellman equation
            y = rewards + (1-done)*self.gamma*self.target_critic(new_states, actor_actions)

        #compute and return the MSE loss 
        return torch.nn.functional.mse_loss(self.critic(states, actions), y)

    def actor_loss(self, batch):
        """
        Computes the loss of the actor network
        """
        states = batch['states'].to(device)
        actor_actions = self.actor(states)

        self.critic.eval()
        aloss = -self.critic(states, actor_actions).mean()
        self.critic.train()
        
        #Compute and return the loss 
        return aloss

    def choose_action(self, state):
        """
        Chooses an action using the actor network and by applying some noise to explore 
        """
        with torch.no_grad():
            self.actor.eval()
            t_state = torch.Tensor(state).unsqueeze(0).to(device)
            action = self.actor(t_state).detach().to("cpu") + self.exploration()
            self.actor.train()
        return torch.clip(action, -1.0, 1.0).numpy()

    def compute_optimal_actions(self, states):
        """
        Choose an action using the actor network
        """
        with torch.no_grad():
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

        closs = self.critic_loss(batch)
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()

        # Don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False

        #Compute the loss for the actor
        aloss = self.actor_loss(batch)
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True

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
        Train the agent 
        """
        #Initialize the replay buffer 
        replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        J_mean = []
        J_std = []

        current_state = self.env.reset()
        # fill the buffer with random actions
        for i in range(1000):
            action = np.random.uniform(-1.0, 1.0)
            new_state, reward, done, _ = self.env.step([action])
            replay_buffer.store((current_state, action, reward, new_state, done))
            current_state = self.env.reset() if done else new_state

        #Algorithm
        for i in range(1, self.episodes+1):
            #Initialize a new starting state for the episode
            current_state = self.env.reset()
            
            #Keep track of losses
            critic_losses = []
            actor_losses = []
            
            start = time.process_time()

            self.env.seed(self.nb_simulation + i) # not interfer with the computation of J

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
            j = J(self.env, self, self.gamma, self.nb_simulation, 1000)
            J_mean.append(j[0])
            J_std.append(j[1])
            print(time.process_time() - start)
            print("Episode {}: Critic: {} | Actor: {} | J: {}".format(i+1, avg_critic_loss, avg_actor_loss, j))

        torch.save(self.actor.state_dict(), "saved_models/actor_{}_{}_ddpg".format(self.episodes, self.file_extension))
        torch.save(self.critic.state_dict(), "saved_models/critic_{}_{}_ddpg".format(self.episodes, self.file_extension))

        J_mean = np.array(J_mean)
        J_std = np.array(J_std)
        plt.plot(J_mean, label="Expected return")
        plt.ylabel("Expected return J")
        plt.xlabel("Episode")
        plt.legend()
        plt.fill_between(range(self.episodes),J_mean-J_std,J_mean+J_std,alpha=.1)
        plt.savefig("figures/J_{}.png".format(self.file_extension))



