import numpy as np
import matplotlib.pyplot as plt

# Define a simple optimization function (Sphere Function)
def sphere_function(x):
    return np.sum(x ** 2)

# Firefly Algorithm
class FireflyAlgorithm:
    def __init__(self, n_fireflies, n_dims, alpha, beta, gamma, lower_bound, upper_bound, max_iter):
        self.n_fireflies = n_fireflies
        self.n_dims = n_dims
        self.alpha = alpha  # Randomization parameter
        self.beta = beta  # Attractiveness
        self.gamma = gamma  # Light absorption coefficient
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        
        # Initialize fireflies' positions and best solution
        self.fireflies = np.random.uniform(lower_bound, upper_bound, (n_fireflies, n_dims))
        self.best_firefly = None
        self.best_fitness = np.inf
        
    def distance(self, firefly1, firefly2):
        return np.linalg.norm(firefly1 - firefly2)
    
    def update_firefly_position(self, firefly, randomization_factor):
        noise = randomization_factor * np.random.uniform(-1, 1, firefly.shape)
        new_position = firefly + noise
        return np.clip(new_position, self.lower_bound, self.upper_bound)
    
    def run(self):
        for iteration in range(self.max_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j:
                        # Calculate attractiveness
                        r = self.distance(self.fireflies[i], self.fireflies[j])
                        beta_ij = self.beta * np.exp(-self.gamma * r ** 2)
                        
                        # Update position based on attractiveness and randomization
                        self.fireflies[i] = (1 - beta_ij) * self.fireflies[i] + beta_ij * self.fireflies[j]
                        self.fireflies[i] = self.update_firefly_position(self.fireflies[i], self.alpha)
            
            # Evaluate fitness and find best firefly
            for i in range(self.n_fireflies):
                fitness = sphere_function(self.fireflies[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_firefly = self.fireflies[i].copy()
                    
        return self.best_firefly, self.best_fitness

# Run the Firefly Algorithm with different randomization factors to see its impact
randomization_factors = [0.13, 0.15, 9.0, 3.0]
results = {}

# Test each randomization factor
for alpha in randomization_factors:
    fa = FireflyAlgorithm(
        n_fireflies=20,
        n_dims=2,
        alpha=alpha,
        beta=0.2,
        gamma=1.0,
        lower_bound=-5,
        upper_bound=5,
        max_iter=50
    )
    
    best_position, best_fitness = fa.run()
    results[alpha] = (best_position, best_fitness)

# Display the results
print("Firefly Algorithm Results with Different Randomization Factors:")
for alpha, (position, fitness) in results.items():
    print(f"Randomization Factor {alpha}: Best Fitness = {fitness}, Best Position = {position}")

# Visualize the final fireflies' positions with different randomization factors
fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'y']
labels = ['alpha=0.1', 'alpha=0.5', 'alpha=1.0', 'alpha=2.0'] #make the labels based on the randomizaiton factors

for i, alpha in enumerate(randomization_factors):
    positions = results[alpha][0]
    ax.scatter(positions[0], positions[1], c=colors[i], label=labels[i])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Fireflies with Different Randomization Factors')
ax.legend()
plt.show()
