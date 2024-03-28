import numpy as np


class CoatiOptimization:
    def __init__(self, cost_function, bounds, max_iter=100, population_size=50, alpha=0.1):
        self.cost_function = cost_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.population_size = population_size
        self.alpha = alpha
        self.dim = len(bounds)
        self.best_solution = None
        self.best_cost = np.inf

    def optimize(self):
        # Initialize population
        population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.population_size, self.dim))

        for _ in range(self.max_iter):
            new_population = population + self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            costs = np.apply_along_axis(self.cost_function, 1, new_population)

            improved_indices = np.where(costs < np.apply_along_axis(self.cost_function, 1, population))
            population[improved_indices] = new_population[improved_indices]

            best_index = np.argmin(costs)
            if costs[best_index] < self.best_cost:
                self.best_solution = population[best_index]
                self.best_cost = costs[best_index]

        return self.best_solution, self.best_cost


def cost_function(x):
    return np.sum(np.square(x))


bounds = np.array([[-5, 5], [-5, 5]])  # Define the bounds for each dimension

optimizer = CoatiOptimization(cost_function, bounds)
best_solution, best_cost = optimizer.optimize()
print("Best solution:", best_solution)
print("Best cost:", best_cost)
