import numpy as np
import matplotlib.pyplot as plt

def flatten(x):
    flat_x_i = []
    shapes = []
    for x_i in x:
        shapes.append(np.array(x_i).shape)
        flat_x_i.append(np.array(x_i).flatten())
        
    flat_x = np.concatenate(flat_x_i)
    return flat_x, shapes
    
def unflatten(flat_x, shapes):
    x = []
    offset = 0
    for shape in shapes:
        lenght = 1
        for n in shape:
            lenght *= n
        x.append(flat_x[offset:offset+lenght].reshape(shape))
        offset += lenght
    return x

class CMA_ES:
    def __init__(self, mean, sigma, population_size, fitness_function):
        flat_mean, shapes = flatten(mean)
        self.shapes = shapes
        self.mean = np.array(flat_mean)
        self.sigma = sigma
        self.population_size = population_size
        self.fitness_function = fitness_function

        self.dimension = len(flat_mean)
        self.cov_matrix = np.identity(self.dimension)
        self.weights = np.log(self.population_size / 2 + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)

        self.c_mu = 4 / (self.dimension + 4)
        self.c_1 = 2 / ((self.dimension + 1.3)**2 + self.mu_eff)
        self.c_sigma = (self.mu_eff + 2) / (self.dimension + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dimension + 1)) - 1) + self.c_sigma

        self.p_sigma = np.zeros(self.dimension)
        self.p_c = np.zeros(self.dimension)

        self.history = []  # To store mean for visualization

    def ask(self):
        solutions = [
            unflatten(
                self.mean + self.sigma * np.random.multivariate_normal(np.zeros(self.dimension), self.cov_matrix),
                self.shapes
            )
            for _ in range(self.population_size)
        ]
        return solutions


    def tell(self, solutions):
        solutions= [flatten(solution)[0] for solution in solutions]
        solutions.sort(key=lambda s: self.fitness_function(unflatten(s,self.shapes)), reverse=False)

        samples = np.array([s for s in solutions])
        new_mean = np.sum(self.weights[:, None] * samples[:len(self.weights)], axis=0)
        mean_shift = new_mean - self.mean

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * mean_shift / self.sigma

        norm_p_sigma = np.linalg.norm(self.p_sigma)
        h_sigma = norm_p_sigma / np.sqrt(1 - (1 - self.c_sigma)**(2 * self.population_size)) < (1.4 + 2 / (self.dimension + 1))

        self.p_c = (1 - self.c_1) * self.p_c + h_sigma * np.sqrt(
            self.c_1 * (2 - self.c_1) * self.mu_eff
        ) * mean_shift / self.sigma

        rank_one_update = np.outer(self.p_c, self.p_c)
        rank_mu_update = np.sum([
            self.weights[i] * np.outer(samples[i] - self.mean, samples[i] - self.mean)
            for i in range(len(self.weights))
        ], axis=0)

        self.cov_matrix = (
            (1 - self.c_1 - self.c_mu) * self.cov_matrix +
            self.c_1 * rank_one_update +
            self.c_mu * rank_mu_update
        )

        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (norm_p_sigma / self.dimension - 1)
        )

        self.mean = new_mean

        # Save the mean for visualization
        self.history.append(self.mean)


# Example usage:
def fitness_function(x):
    return np.sum(x[0]**2)


if __name__=="__main__":
  mean = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0]]
  sigma = 0.3
  population_size = 10
  cma_es = CMA_ES(mean, sigma, population_size, fitness_function)
  
  for generation in range(100):
      solutions = cma_es.ask()
      cma_es.tell(solutions)
      print(f"Generation {generation}: Best fitness = {fitness_function(cma_es.mean)}")
