#!/bin/env python
import numpy as np

"""
    Pedict the future using continuous chain markov model with a blackscholes intergrated to it
This function simulates the state transitions using the adapted Black-Scholes dynamics thus allowing 
to factor in the numerous volatilitythat could affect future state prediction. This module is meant 
to make predictions so that the breeder and other stakeholders can know how they are afiring towards 
archiving global goals and to pin point if there could be any gabs in breeding in the future so that
they can even start early interventions
"""

class BlackScholesSimulator:
    def __init__(self, initial_accuracy, initial_connectivity, r, sigma, dt):
        self.initial_accuracy = initial_accuracy
        self.initial_connectivity = initial_connectivity
        self.r = r
        self.sigma = sigma
        self.dt = dt

    def simulate(self, n_steps):
        accuracies = [self.initial_accuracy]
        connectivities = [self.initial_connectivity]

        for step in range(n_steps):
            S_acc = accuracies[-1]
            S_con = connectivities[-1]

            # Simulate changes using Black-Scholes-like dynamics
            dV_acc = self.r * S_acc * self.dt + 0.5 * self.sigma**2 * S_acc**2 * np.random.normal(0, np.sqrt(self.dt))
            dV_con = self.r * S_con * self.dt + 0.5 * self.sigma**2 * S_con**2 * np.random.normal(0, np.sqrt(self.dt))

            # Update states
            new_accuracy = accuracies[-1] + dV_acc
            new_connectivity = connectivities[-1] + dV_con

            # Bound the values within reasonable limits
            new_accuracy = max(0, min(1, new_accuracy))  # Accuracy should stay within [0, 1]
            new_connectivity = max(0, new_connectivity)  # Keep connectivity non-negative

            # Append new states
            accuracies.append(new_accuracy)
            connectivities.append(new_connectivity)

        return accuracies, connectivities

class SimulatorManager:
    def __init__(self, score, initial_connectivity=50, n_steps=5, r=0.03, sigma=0.1, dt=1):
        self.score = score
        self.initial_connectivity = initial_connectivity
        self.n_steps = n_steps
        self.r = r
        self.sigma = sigma
        self.dt = dt

    def get_initial_conditions(self):
        initial_accuracy = self.score / 100  # Starting accuracy is the confidence score for the semantic similarity classification
        return initial_accuracy, self.initial_connectivity

    def simulate_future_steps(self):
        initial_accuracy, initial_connectivity = self.get_initial_conditions()
        simulator = BlackScholesSimulator(initial_accuracy, initial_connectivity, self.r, self.sigma, self.dt)
        accuracies, connectivities = simulator.simulate(self.n_steps)

        # Print predictions
        self.print_predictions(accuracies, connectivities)

    def print_predictions(self, accuracies, connectivities):
        for i in range(self.n_steps + 1):
            print(f"Step {i}: Accuracy = {accuracies[i]:.2f}, Connectivity = {connectivities[i]:.2f}")

# Example usage
score = 80  # Example score, replace with the actual score value
sim_manager = SimulatorManager(score, n_steps=5)
sim_manager.simulate_future_steps()

