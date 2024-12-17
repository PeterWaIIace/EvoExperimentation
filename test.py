from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import genAlg as ga
import numpy as np 
import os


def generateBots(N = 10):
    bodies = []
    connectomes = []
    scores = []
    for _ in range(N):
        body, connections = sample_robot((5,5))
        bodies.append(body)
        connectomes.append(connections)
        scores.append(0)
    
    return bodies,connectomes,scores

if __name__ == '__main__':

    bodies, connectomes, scores = generateBots(4)

    while True:
        for n,body in enumerate(bodies):

            print(body,connectomes[n])
            world = EvoWorld.from_json(os.path.join('exported', 'my_evironment.json'))
            world.add_from_array(
                name='robot',
                structure=body,
                x=0,
                y=10,
                connections=connectomes[n]
            )

            sim = EvoSim(world)
            sim.reset()

            viewer = EvoViewer(sim)
            viewer.track_objects('robot')

            for i in range(500):
                sim.set_action(
                    'robot',
                    np.random.uniform(
                        low = 0.6,
                        high = 1.6,
                        size=(sim.get_dim_action_space('robot'),)
                    )
                )
                sim.step()
                viewer.render('screen')
            viewer.close()
        print("evolution")

        bodies = ga.mixAndMutate(bodies,scores)
        connections = ga.mixAndMutate(connectomes,scores)