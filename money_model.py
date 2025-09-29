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
        self.retired= False

    def assess_retirement(self):
        """Decide if the agent should retire this step."""
        if (not self.retired
            and self.model.steps_remaining >= self.model.cutoff_step
            and self.wealth >= (self.model.steps_remaining+3)):
            self.retired = True

    def exchange(self):
        if self.wealth > 0:
            # Pick a recipient: only non-retired agents are eligible
            eligible = [a for a in self.model.agents if not a.retired and a is not self]

            if eligible:
                other_agent = self.random.choice(eligible)
                other_agent.wealth += 1

            # Retired agents still spend, but "into the void" if no eligible recipient
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, steps, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.schedule = RandomActivation(self)
        self.steps_remaining = steps
        self.cutoff_step = steps*0.05

        for i in range(self.num_agents):
            agent = MoneyAgent(i+1, self)
            self.schedule.add(agent)

    def step(self):
        """Advance the model by one step."""
        self.agents.do("assess_retirement")
        self.agents.shuffle_do("exchange")
        self.steps_remaining-=1

all_wealth = []
for _ in range(100):

    agents = 50
    steps = 100

    # Run the model
    model = MoneyModel(agents,steps)
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
