import bisect
import glob
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import Image

from agent import Agent


def select(select_probability, scored_agents):
    r = np.random.random()
    i = bisect.bisect(select_probability, r)
    i = np.clip(i - 1, 0, len(select_probability) - 1)
    return scored_agents[i][1]

def run():
    # Remove any old plots
    for f in glob.glob("gen-*.png"):
        os.unlink(f)

    im = Image.open('PerlinNoise2d.png')
    imdata = np.mean(np.array(im), axis=2)

    n = 4
    epsilon = 4
    lam = 0.001
    maxN = 1000
    num_elites = 5

    points = [(25, 50), (25, 150), (25, 250),
              (25, 350), (25, 450)]

    # population size
    P = 50
    # fraction of population to be entirely new each generation
    P_new = .1
    new_per_gen = int(P * P_new)
    gens = 100
    Q = 30000

    # Initial population
    agents = [Agent(np.random.standard_t(2, n)) for _ in range(P)]

    best_scores = []
    best = np.inf
    for generation in range(gens):
        t0 = time.time()

        scored_agents = []
        for agent in agents:
            agent.reset_paths()
            score = 0
            for p1, p2 in points:
                score += agent.score(imdata, p1, p2, epsilon, lam, n, maxN, Q)
            scored_agents.append((score, agent))

        scored_agents.sort()
        scores = np.array([score for score, agent in scored_agents])
        best_cost, best_agent = scored_agents[0]

        if best_cost < best:
            path_img = best_agent.render_paths(im)
            path_img.save("gen-%06d.png" % generation)
            best = best_cost

        # Create new population

        # Keep best individual (Elitism)
        new_agents = [agent for score, agent in scored_agents[:num_elites]]

        # Add new individuals
        for _ in xrange(new_per_gen):
            new_agents.append(Agent(np.random.standard_t(2, n)))

        # Generate the remainder by crossing existing individuals
        select_probability = np.cumsum(1 / scores)
        select_probability /= select_probability[-1]
        for _ in range(P - len(new_agents)):
            # Choose two parents, with probability proportional to 1 / score
            parent1 = select(select_probability, scored_agents)
            parent2 = select(select_probability, scored_agents)

            # Crossover
            pivot = random.randint(1, n - 2)
            p1alpha = parent1.alpha.tolist()
            p2alpha = parent2.alpha.tolist()
            alpha = p1alpha[:pivot] + p2alpha[pivot:]
            alpha = np.array(alpha)

            # Mutate
            alpha += (0.1 * np.random.randn(*alpha.shape))
            new_agents.append(Agent(alpha))

        agents = new_agents

        dt = time.time() - t0
        print 'Generation %03d/%03d: %011.3f (%.1f sec)' % (generation, gens,
                                                            best_cost, dt)
        best_scores.append(best_cost)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(best_scores)
    ax.set_yscale("log")
    fig.savefig("error-plot.png")

if __name__ == '__main__':
    run()