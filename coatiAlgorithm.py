import numpy as np

# bu class gerekli olan tüm parametreleri alır ve optimize edilecek işlevi ve sınırları tutar
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

    # optimize metodu ile belirli bir iterasyon sayısı boyunca popülasyon güncellenir ve en iyi çözümü ve maliyetini günceller.
    def optimize(self):

        #  Coati Optimizasyon Algoritması için başlangıç popülasyonunu oluşturur.
        population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.population_size, self.dim))

         #Coati Optimizasyon Algoritması'nın ana iterasyonunu içerir.
         # Her iterasyonda, mevcut popülasyonun her bir üyesi için yeni bir çözüm üretilir ve en iyi çözüm güncellenir.
        for _ in range(self.max_iter):
            new_population = population + self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5) # mevcut popülasyondaki her bir üyenin yerine yeni bir çözüm oluşturur.
            costs = np.apply_along_axis(self.cost_function, 1, new_population) #maliyetler hesaplanır

            improved_indices = np.where(costs < np.apply_along_axis(self.cost_function, 1, population)) #  yeni popülasyondaki çözümlerin maliyetlerini ve mevcut popülasyondaki çözümlerin maliyetlerini karşılaştırır. Daha iyi (daha düşük maliyetli) çözümlerin dizinlerini bulur.
            population[improved_indices] = new_population[improved_indices]

            # en iyi çözüm ve maliyet bulunur
            best_index = np.argmin(costs)
            if costs[best_index] < self.best_cost:
                self.best_solution = population[best_index]
                self.best_cost = costs[best_index]

        return self.best_solution, self.best_cost


def cost_function(x):
    return np.sum(np.square(x))


bounds = np.array([[-5, 5], [-5, 5]])  # optimize edilecek değişkenlerin her biri için alt ve üst sınırları içeren bir numpy dizisi

optimizer = CoatiOptimization(cost_function, bounds)
best_solution, best_cost = optimizer.optimize()
print("Best solution:", best_solution)
print("Best cost:", best_cost)
