#Enhanced Algorithm with Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import gamma as gamma_func
import time

#CSV dataset
data = pd.read_csv("healthcare_data.csv")  
hospitals = data["Hospital_code"].values
severity_mapping = {'Extreme': 3, 'Moderate': 2, 'Minor': 1}
severity = data["Severity of Illness"].map(severity_mapping).values  # Apply the mapping to the severity column
max_capacity = data["Available Extra Rooms in Hospital"].values

# Normalize severity scores
severity_normalized = severity / np.sum(severity)

# Updated objective function to prioritize severity
def objective_function(x):
    # Penalty for exceeding max capacity
    allocation_penalty = np.sum(np.maximum(0, x - max_capacity) ** 2)
    
    # Penalty for under-allocating resources relative to severity
    severity_penalty = np.sum((x / np.sum(x) - severity_normalized) ** 2)
    
    # Increase severity's weight in the objective function (optional)
    severity_weight = 10  # Increase this value to give more weight to severity
    return severity_weight * severity_penalty + allocation_penalty

class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha=0.5, beta_min=0.2, gamma_val=1.0):
        self.n_fireflies = n_fireflies
        self.n_dim = n_dim
        self.lower_bound = lower_bound
        self.upper_bound = np.where(upper_bound == 0, 1e-6, upper_bound)
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma_val = gamma_val
        
        self.fireflies = self.initialize_fireflies()
        self.light_intensity = np.zeros(n_fireflies)
        self.best_firefly = None
        self.best_intensity = float("inf")
        self.intensity_history = []

    def initialize_fireflies(self):
        kmeans = KMeans(n_clusters=self.n_fireflies) #k-means clustering
        clusters = kmeans.fit(np.random.uniform(self.lower_bound, self.upper_bound, (self.n_fireflies * 2, self.n_dim)))
        return clusters.cluster_centers_

    def update_light_intensity(self):
        for i in range(self.n_fireflies):
            self.light_intensity[i] = objective_function(self.fireflies[i])
            if self.light_intensity[i] < self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_firefly = self.fireflies[i]

    def attractiveness(self, r, beta, gamma): #iteration parameter
        #dynamic_light_absoption = 1.0 / (1+(iteration/self.max_iter)) 
        return self.beta_min * np.exp(-self.gamma_val * r**2) #* dynamic_light_absoption

    def levy_flight(self, Lambda):
        sigma = (gamma_func(1 + Lambda) * np.sin(np.pi * Lambda / 2) /(gamma_func((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.randn() * sigma
        v = np.random.randn()
        step = u / abs(v) ** (1 / Lambda)
        return step

    def move_fireflies(self, iteration):
        #dynamic_alpha = self.alpha * (1 - iteration / self.max_iter)  # Adjust alpha for exploration-exploitation
        #dynamic_beta = self.beta_min + (1 - self.beta_min) * np.exp(-iteration / (self.max_iter / 2))  # Beta decay for convergence
        #dynamic_gamma = self.gamma_val * (1 - iteration / self.max_iter)  # Gamma decay to balance attraction
        
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[i] > self.light_intensity[j]:
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.attractiveness(r) #iteration with dynamic gamma and beta for parametric adjustment
                    L = self.levy_flight(1.5) #Levy Flight = ()/Scaling = Balanced mix of short & long jumps if lower (1) more long jumps, if lesser more short jumps like gaussian method
                    step_size = (1 - r / self.upper_bound) * dynamic_alpha #Dynamic Alpha Steps
                    self.fireflies[i] = (
                        self.fireflies[i] * (1 - beta) +
                        self.fireflies[j] * beta +
                        step_size * (np.random.rand(self.n_dim) - 0.5) +
                        L
                    )
                    self.fireflies[i] = np.clip(self.fireflies[i], self.lower_bound, self.upper_bound)

    def optimize(self):
        start_time = time.time()
        for t in range(self.max_iter):
            self.update_light_intensity()
            self.move_fireflies(t)
            self.intensity_history.append(self.best_intensity)
            print(f"Iteration {t+1}/{self.max_iter}: Best Intensity = {self.best_intensity:.6e}")
        end_time = time.time()
        execution_time_min = (end_time - start_time)/60
        execution_time_sec = end_time - start_time
        print(f"Speed Record in Minutes: {execution_time_min}") 
        print(f"Speed Record in Seconds: {execution_time_sec}") 
        

        return self.best_firefly, self.best_intensity

# Parameters
n_dim = len(hospitals)  # Number of hospitals
lower_bound = np.zeros(n_dim)
upper_bound = max_capacity
n_fireflies = 20
max_iter = 100
alpha = 0.1 # Potential Adjustment: 0.2-0.3 (improve diversity early in the process)
beta_min = 0.2 # Potential Adjjustment: 0.4 - 0.4 (could accelerate convergence when there’s a smaller search space or highly constrained bounds.)
gamma_val = 1.0 # Potential Adjustment: 0.5 (For high dimensional spaces, could prevent distant fireflies from being ignored too soon)
#^^ change along with the severity weight

# Run optimization
fa = FireflyAlgorithm(n_fireflies, n_dim, lower_bound, upper_bound, max_iter, alpha, beta_min, gamma_val)
best_solution, best_intensity = fa.optimize()
best_solution = np.round(best_solution)

# Display results
#print(f"Best Allocation based on severity: {best_solution}")
print(f"Total Severity Penalty (Objective Value): {best_intensity:.6e}")


# Visualization of resource allocation across hospitals
plt.figure(figsize=(10, 5))
plt.plot(fa.intensity_history, label="Objective Value", color="blue")
plt.plot(fa.intensity_history, label=best_intensity, color="blue")
plt.xlabel("Iteration")
plt.ylabel("Objective Value (Severity Penalty + Allocation Penalty)")
plt.title("Convergence Over Iterations")
plt.legend()
plt.grid()
#plt.show()

# Firefly Movement Visualization
plt.figure(figsize=(10, 5))
initial_positions = fa.initialize_fireflies()
plt.scatter(range(n_dim), initial_positions[0], label="Initial Position (Clustered)", color="orange")
plt.scatter(range(n_dim), best_solution, label="Final Allocation (Optimized)", color="green")
plt.xlabel("Hospital Index")
plt.ylabel("Resource Allocation")
plt.title("Firefly Movement: Initial Clustering to Final Optimization")
plt.legend()
plt.grid()
#plt.show()

