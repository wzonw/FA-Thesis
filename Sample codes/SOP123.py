import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import gamma as gamma_func  # Import gamma as gamma_func
import time

def objective_function(x): #Sample Objective Function Formula for sum of squared elements
    return np.sum(x**2)

class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha=0.5, beta_min=0.2, gamma_val=1.0):
        self.n_fireflies = n_fireflies
        self.n_dim = n_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma_val = gamma_val

        # Initialize fireflies in clusters
        self.fireflies = self.initialize_fireflies()
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")
        self.distance_history = []
        self.best_brightness_history = []
        self.best_solutions = []

    def initialize_fireflies(self): #K-Means Clustering function for Firefly Clustered Initialization
        kmeans = KMeans(n_clusters=self.n_fireflies) 
        clusters = kmeans.fit(np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies * 2, self.n_dim)))
        return clusters.cluster_centers_

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r, beta, gamma): #iteration parameter
        return beta * np.exp(-self.gamma_val * r**2)

    def levy_flight(self, Lambda): #Levy Flight Function
        sigma = (gamma_func(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (gamma_func((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.randn() * sigma
        v = np.random.randn()
        step = u / abs(v)**(1 / Lambda)

        #print(f"step: {step}")
        step = np.clip(step, -1, 1)
        return step

    def calculate_average_distance(self):
        # Calculate the average distance between all pairs of fireflies
        total_distance = 0
        for i in range(self.n_fireflies):
            for j in range(i + 1, self.n_fireflies):  # Avoid redundant pairs
                total_distance += np.linalg.norm(self.fireflies[i] - self.fireflies[j])
        # Return the average distance
        return total_distance / (self.n_fireflies * (self.n_fireflies - 1) / 2)


    def move_fireflies(self, iteration): 
        # Dynamic parameter adjustment
        dynamic_alpha = self.alpha / (1 + iteration / self.max_iter)  # Adjust alpha for exploration-exploitation
        dynamic_beta = self.beta_min + (1 - self.beta_min) * np.exp(-iteration / self.max_iter )  # Beta decay for convergence
        dynamic_gamma = self.gamma_val * (iteration / self.max_iter)  # Gamma decay to balance attraction


        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r, dynamic_beta, dynamic_gamma) #b and y dynamic adjusment
                    L = self.levy_flight(1.5) * np.exp(-iteration / (self.max_iter / 2))  # Levy flight Formula with 1.5 scale for jumps with decaying function to lessen the long jump over iteration
                    #L = np.clip(L, 1, 2) #Limit the Lambda Values to mitigate randomization errors
                    step_size = (1 - r / self.upper_bound) * dynamic_alpha  # Dynamic alpha step size
                    self.fireflies[i] = (
                        self.fireflies[i] * (1 - beta) +
                        self.fireflies[j] * beta +
                        step_size * (np.random.rand(self.n_dim) - 0.5) +
                        L
                    )
                    self.fireflies[i] = np.clip(self.fireflies[i], self.lower_bound, self.upper_bound)

    def calculate_diversity(self):
        # Calculate the diversity (standard deviation) of firefly positions
        return np.std(self.fireflies, axis=0).mean()

    def optimize(self):
        start_time = time.time()
        diversity_over_time = []
        for t in range(self.max_iter):
            self.update_light_intensity()
            self.move_fireflies(t)
            
            # avg_distance = self.calculate_average_distance()
            # self.distance_history.append(avg_distance)

            self.best_brightness_history.append(self.best_intensity)
            self.best_solutions.append(self.best_firefly)

            diversity = self.calculate_diversity()
            diversity_over_time.append(diversity)
            
            print(f"iteration {t+1}/{self.max_iter}: Best Intensity = {self.best_intensity:.6f} ")
        end_time = time.time()
        execution_time_min = (end_time - start_time)/60
        execution_time_sec = end_time - start_time
        print(f"Speed Record in Minute/s: {execution_time_min}")
        print(f"Speed Record in Second/s: {execution_time_sec}")    
        return self.best_firefly, self.best_intensity, diversity_over_time

# Parameters
n_dim = 2
lower_bound = -10
upper_bound = 10
n_fireflies = 20
max_iter = 100
alpha = 0.1
beta_min = 0.4
gamma_val = 1.0

# Initialize and run the algorithm
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity, diversity_over_time = fa.optimize()

print("SOP 123")
print("Best solution:", best_solution)
print(f"Best objective value: {best_intensity:.6f}")


fig, axs = plt.subplots(3,1, figsize=(10,5))
best_solutions = np.array(fa.best_solutions)
best_fitness_value = np.array(fa.best_brightness_history)


axs[0].plot(best_solutions[:,0], best_solutions[:,1], '-o')
axs[0].set_title("Best Solutions Over Iterations")
axs[0].set_xlabel('x1')
axs[0].set_ylabel("x2")


axs[1].plot(best_fitness_value)
axs[1].set_title("Best Function Value over Iteration")
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel("Funciton Values")


axs[2].plot(diversity_over_time)
axs[2].set_title("Diversity Over Iteration")
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel("Diversity")
'''
# Visualization (optional)a
plt.scatter(fa.fireflies[:, 0], fa.fireflies[:, 1], c='yellow', label='Fireflies')
plt.scatter(fa.best_firefly[0], fa.best_firefly[1], c='red', label='Best Solution')
plt.title(f'Iteration {max_iter}')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
'''
plt.tight_layout()
plt.show()


