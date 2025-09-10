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

    def exchange(self):
        # Verify agent has some wealth
        if self.wealth > 0:

            other_agent = self.random.choice(self.model.agents)

            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            agent = MoneyAgent(i+1, self)
            self.schedule.add(agent)

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("exchange")

all_wealth = []
for _ in range(100):

    agents = 50
    steps = 50

    # Run the model
    model = MoneyModel(agents)
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
