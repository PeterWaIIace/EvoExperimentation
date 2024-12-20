from evogym import sample_robot
import gymnasium as gym
import evogym.envs
import genAlg as ga
import numpy as np 
import os
import time
from multiprocessing import Pool
from Bot import Bot

def generateBots(N = 10):
    bodies = []
    connectomes = []
    scores = []
    body, connections = sample_robot((5,5))
    for _ in range(N):
        print(connections)
        bodies.append(body)
        connectomes.append(connections)
        scores.append(0)
    
    return bodies,connectomes,scores

def run_sim(body):
    B = Bot(body)
    B.run()
    return (body,B.reward)

if __name__ == '__main__':

    bodies, connectomes, scores = generateBots(20)

    while True:
        scores = []
        with Pool(len(bodies)) as p:
            results = p.map(run_sim,bodies)

            bodies = []
            for body, score in results:
                bodies.append(body)
                scores.append(score)
        print(scores)
        bodies = ga.mixAndMutate(bodies,scores)