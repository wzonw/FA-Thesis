#Parameter Comparison
#Consistent Result so far with the parameters. Could be enhanced more.
#Parameter at best = a= 0.1, b=0.2, g =1.0
#Try to adjust ang bagal ng run. 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress visualization

# Define a simple optimization function (Sphere Function)
def sphere_function(x):
    return np.sum(x ** 2)

# Firefly Algorithm class with additional observations for convergence and stability
class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dims, alpha, beta, gamma, lower_bound, upper_bound, max_iter):
        self.n_fireflies = n_fireflies
        self.n_dims = n_dims
        self.alpha = alpha  # Randomization factor
        self.beta = beta  # Attractiveness
        self.gamma = gamma  # Absorption
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        
        # Initialize fireflies and best solution
        self.fireflies = np.random.uniform(lower_bound, upper_bound, (n_fireflies, n_dims))
        self.best_firefly = None
        self.best_fitness = np.inf
        
    def distance(self, firefly1, firefly2):
        return np.linalg.norm(firefly1 - firefly2)
    
    def update_firefly_position(self, firefly):
        noise = self.alpha * np.random.uniform(-1, 1, firefly.shape)
        new_position = firefly + noise
        return np.clip(new_position, self.lower_bound, self.upper_bound)
    
    def run(self):
        best_fitness_history = []
        for iteration in tqdm(range(self.max_iter), desc="Running Firefly Algorithm"):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j:
                        # Calculate attractiveness based on distance and gamma (absorption)
                        r = self.distance(self.fireflies[i], self.fireflies[j])
                        beta_ij = self.beta * np.exp(-self.gamma * r ** 2)
                        
                        # Move toward brighter fireflies and apply randomization
                        self.fireflies[i] = (1 - beta_ij) * self.fireflies[i] + beta_ij * self.fireflies[j]
                        self.fireflies[i] = self.update_firefly_position(self.fireflies[i])
            
            # Evaluate fitness and track the best solution
            for i in range(self.n_fireflies):
                fitness = sphere_function(self.fireflies[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_firefly = self.fireflies[i].copy()
            best_fitness_history.append(self.best_fitness)
        
        return self.best_firefly, self.best_fitness, best_fitness_history

# Parameters for experiment
n_fireflies = 20
n_dims = 3 #this states how many we are considering for the firefly.
lower_bound = -10
upper_bound = 10
max_iter = 15

# Experiment with different parameters to control randomization and observe behavior
experiments = [
    {"alpha": 0.1, "beta": 0.2, "gamma": 1.0}, #Best Parameter, make other parameters run in random so we can find the best parameter fit for SOP 1 and 2
    {"alpha": 0.01, "beta": 0.9, "gamma": 1.0},
    {"alpha": 1.0, "beta": 0.4, "gamma": 1.0},
    {"alpha": 2.0, "beta": 0.5, "gamma": 1.0},
]

# Run the Firefly Algorithm with different parameters to observe convergence and stability
results = {}

for config in experiments:
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    
    best_fitness_list = []
    convergence_histories = []
    
    # Run the algorithm multiple times to evaluate stability
    for _ in range(10):
        fa = FireflyAlgorithm(
            n_fireflies=n_fireflies,
            n_dims=n_dims,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_iter=max_iter
        )
        
        best_firefly, best_fitness, convergence_history = fa.run()
        best_fitness_list.append(best_fitness)
        convergence_histories.append(convergence_history)
    
    results[(alpha, beta, gamma)] = {
        "best_fitness_list": best_fitness_list,
        "convergence_histories": convergence_histories
    }

# Analyze and visualize the results to understand randomization control, convergence, and stability
fig, ax = plt.subplots(figsize=(10, 6))

# Plot convergence histories to assess convergence speed and stability
for key, data in results.items():
    alpha, beta, gamma = key
    avg_convergence = np.mean(data["convergence_histories"], axis=0)
    ax.plot(avg_convergence, label=f"alpha={alpha}, beta={beta}, gamma={gamma}")

ax.set_xlabel("Iterations")
ax.set_ylabel("Best Fitness Achieved")
ax.set_title("Convergence Speed and Stability with Different Parameters")
ax.legend()
plt.show()

# Display final results to assess variability and performance
plt.figure()
sns.boxplot(data=[results[key]["best_fitness_list"] for key in results.keys()], orient="v")
plt.xticks(range(len(results.keys())), [f"alpha={key[0]}, beta={key[1]}, gamma={key[2]}" for key in results.keys()])
plt.ylabel("Best Fitness Achieved")
plt.xlabel("Parameter Configuration")
plt.title("Effect of Randomization, Attractiveness, and Absorption on Performance")
plt.show()
