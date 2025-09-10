import numpy as np
import pandas as pd
import seaborn as sns
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation

class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)
        self.wealth = 10
        self.popularity = 0.5

    def exchange(self):
    # Verify agent has some wealth
        if self.wealth > 0:
            # Pick 3 distinct other agents as the "neighborhood"
            neighborhood = self.random.sample(
                [a for a in self.model.agents if a is not self], 3)

            # Choose the neighbor with the highest popularity
            best = max(neighborhood, key=lambda a: a.popularity)

            # Transfer money
            give_amount = self.wealth * self.model.give_proportion
            best.wealth += give_amount
            self.wealth -= give_amount

class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, give_proportion, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.give_proportion = give_proportion
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            agent = MoneyAgent(i+1, self)
            self.schedule.add(agent)

    def adjust_popularity(self):
        """Assign popularity scores to all agents based on wealth rank."""
        # Shuffle to break ties randomly
        agents = self.random.sample(self.agents, len(self.agents))

        # Rank by wealth (lowest = 0, highest = 1)
        sorted_agents = sorted(agents, key=lambda a: a.wealth)
        ranks = np.linspace(0, 1, len(sorted_agents))

        # Bell curve popularity: 0 at extremes, 1 at median
        popularity_curve = 1.0 - 4.0 * (ranks - 0.5) ** 2

        # Assign to agents
        for agent, pop in zip(sorted_agents, popularity_curve):
            agent.popularity = max(0.0, pop)  # just in case of tiny negatives

    def step(self):
        """Advance the model by one step."""

        self.adjust_popularity()
        self.agents.shuffle_do("exchange")

all_wealth = []
for _ in range(100):

    agents = 50
    steps = 50
    give_proportion = 0.2

    # Run the model
    model = MoneyModel(agents, give_proportion)
    for _ in range(steps):
        model.step()

    # Store the results
    for agent in model.agents:
        all_wealth.append(agent.wealth)

# Use seaborn
g = sns.histplot(all_wealth, discrete=True)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents");

import matplotlib.pyplot as plt
plt.show()
