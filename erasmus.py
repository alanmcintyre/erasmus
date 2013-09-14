import random
import time

import numpy as np
import Image

from agent import Agent

im = Image.open('PerlinNoise2d.png')
imdata = np.mean(np.array(im), axis=2)

n = 4
epsilon = 4
lam = 0.001
maxN = 500

p1, p2 = 30, 267

P = 50 # population size
P_new = .1 # fraction of population to be entirely new each generation
new_per_gen = int(P*P_new)
gens = 100
Q = 3000000
        
# Initial population
agents = [Agent(np.random.standard_t(2, n)) for i in range(P)]

mc = np.zeros((gens,1))
best = np.inf
for generation in range(gens):
    t0 = time.time()
    
    costs = []
    for agent in agents:
        cost = agent.score(imdata, p1, p2, epsilon, lam, n, maxN, Q)
        costs.append((cost, agent))
        
    costs.sort()
    best_cost, best_agent = costs[0]
    
    if best_cost < best:
        path_img = best_agent.render_path(im)
        path_img.save("gen-%06d.png" % generation)
        best = best_cost

    # Create new population
    
    # Keep best individual (Elitism)
    new_agents = [best_agent]
    
    # Add new individuals
    new_agents += [Agent(np.random.standard_t(2, n)) for _ in range(new_per_gen)]

    # Generate the remainder by crossing existing individuals
    for j in range(P-len(new_agents)):
        # Half the time we use the current best as a parent
        if random.randint(0, 1):
            parent1 = best_agent    
        else:
            unused, parent1 = random.choice(costs[:20])
        
        unused, parent2 = random.choice(costs[:20])

        # Crossover
        pivot = random.randint(1, n-2)
        alpha = parent1.alpha.tolist()[:pivot] + parent2.alpha.tolist()[pivot:]
        alpha = np.array(alpha)

        # Mutate
        alpha += 0.1*np.random.randn(*alpha.shape)
        new_agents.append(Agent(alpha))
        
    agents = new_agents

    dt = time.time() - t0
    print 'Generation %03d/%03d: %011.3f (%.1f sec)' % (generation, gens, best_cost, dt)
