import numpy as np
import matplotlib.pyplot as plt

class FireflyAlgorithm:
    def __init__(self, num_fireflies, num_iterations, alpha, gamma, beta0, dimension, bounds):
        self.num_fireflies = num_fireflies
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.beta0 = beta0
        self.dimension = dimension
        self.bounds = bounds
        
        # Initialize fireflies at fixed positions (5, as per your request)
        self.fireflies = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_fireflies, self.dimension))
        
        # For tracking exploration vs exploitation
        self.distance_history = []
        self.best_fitness_values = []

    def sphere(self, x):
        # Objective function: sphere function to minimize
        return np.sum(x**2)

    def compute_brightness(self):
        # Calculate brightness based on the inverse of the objective function
        return 1.0 / (1.0 + np.array([self.sphere(f) for f in self.fireflies]))

    def update_fireflies(self, brightness):
        # Update firefly positions based on brightness
        new_fireflies = np.copy(self.fireflies)
        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                if brightness[j] > brightness[i]:
                    distance = np.linalg.norm(self.fireflies[j] - self.fireflies[i])
                    beta = self.beta0 * np.exp(-self.gamma * distance**2)  # Attraction
                    random_vector = self.alpha * (np.random.uniform(-0.5, 0.5, self.dimension))
                    new_fireflies[i] = (
                        self.fireflies[i] + beta * (self.fireflies[j] - self.fireflies[i]) + random_vector
                    )
                    # Ensure boundaries are respected
                    new_fireflies[i] = np.clip(new_fireflies[i], self.bounds[0], self.bounds[1])
        return new_fireflies

    def calculate_average_distance(self):
        # Calculate the average distance between all pairs of fireflies
        total_distance = 0
        for i in range(self.num_fireflies):
            for j in range(i + 1, self.num_fireflies):  # Avoid redundant pairs
                total_distance += np.linalg.norm(self.fireflies[i] - self.fireflies[j])
        # Return the average distance
        return total_distance / (self.num_fireflies * (self.num_fireflies - 1) / 2)

    def optimize(self):
        best_solutions = []
        for iteration in range(self.num_iterations):
            brightness = self.compute_brightness()
            self.fireflies = self.update_fireflies(brightness)
            best_firefly = self.fireflies[np.argmax(brightness)]
            best_solutions.append(best_firefly)
            best_fitness = self.sphere(best_firefly)
            self.best_fitness_values.append(best_fitness)
            
            # Track average distance between fireflies to understand exploration vs exploitation
            avg_distance = self.calculate_average_distance()
            self.distance_history.append(avg_distance)
            
            print(f"Iteration {iteration + 1}/{self.num_iterations}: Best Fitness = {best_fitness:.6f}")
        
        return best_solutions, self.best_fitness_values

# Firefly Algorithm parameters
num_fireflies = 20
num_iterations = 15
alpha = 0.5  # Randomness factor
gamma = 1.0  # Absorption coefficient
beta0 = 0.2  # Attraction at zero distance
dimension = 2  # Dimension of the problem space
bounds = [-10, 10]  # Bounds for the problem space

# Initialize Firefly Algorithm
fa = FireflyAlgorithm(num_fireflies, num_iterations, alpha, gamma, beta0, dimension, bounds)

# Run the optimization
best_solutions, best_fitness_values = fa.optimize()

# Display the results
best_solution = best_solutions[-1]
print("\nBest solution found:", best_solution)
print("Function value at best solution:", fa.sphere(best_solution))

# Create a figure with three subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plotting the path of the best solutions (for visualization)
best_solutions = np.array(best_solutions)
axs[0].plot(best_solutions[:, 0], best_solutions[:, 1], '-o')
axs[0].set_title("Best Solutions Over Iterations")
axs[0].set_xlabel("X1")
axs[0].set_ylabel("X2")

# Plotting the best fitness value over iterations
axs[1].plot(best_fitness_values)
axs[1].set_title("Best Function Value over Iterations")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Best Function Value")

# Plotting the average distance between fireflies over iterations
axs[2].plot(fa.distance_history)
axs[2].set_title("Average Distance Between Fireflies (Exploration vs Exploitation)")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Average Distance")

# Display the plots
plt.tight_layout()
plt.show()
