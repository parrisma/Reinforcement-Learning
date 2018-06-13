import logging

import numpy as np

from examples.gridworld.GridWorldUnitTests.GridActorCritic import GridActorCritic
from examples.gridworld.RenderSimpleGridOneQValues import RenderSimpleGridOneQValues
from examples.gridworld.SimpleGridOne import SimpleGridOne
from reflrn.EnvironmentLogging import EnvironmentLogging

#
# Return a grid as the environment. The aim is to learn the shortest
# path from the start cell to a goal cell, while avoiding the
# penalty cells (if any) . Each step has a negative cost, and it is this that drives
# the push for shortest path in terms of minimising the cost function.
#
# The action space is N,S,E,W=(0,1,2,3) shape (1,4)
# The state is the grid coordinates (x,y) shape (1,2)
#
step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL


def create_grid() -> SimpleGridOne:
    r = 10
    c = 10
    grid = np.full((r, c), step)
    grid[4, 4] = goal
    sg1 = SimpleGridOne(grid_id=1,
                        grid_map=grid,
                        respawn_type=SimpleGridOne.RESPAWN_RANDOM)
    return r, c, sg1


#
# What is the cost of finding a goal state from each corner. Actions are predicted by the actor
# only, so no random actions. As such this can be taken as a measure of the agents improvment in
# heading directly to a goal.
#
def optimal_path_check(grid_env: SimpleGridOne,
                       actor_critic: GridActorCritic):
    r, c = grid_env.shape()

    corners = [(0, 0),
               (0, r - 1),
               (c - 1, 0),
               (c - 1, r - 1)
               ]

    path_cost = 0
    for corner in corners:
        i = 0
        grid_env.reset(coords=corner)
        while not grid_env.episode_complete() and i < 5000:
            path_cost += grid_env.execute_action(actor_critic.select_action(cur_state=grid_env.state(), greedy=True))
            i += 1
    return path_cost


#
#
#
def main():
    lg = EnvironmentLogging("TestGridActorCritic2",
                            "TestGridActorCritic2.log",
                            logging.DEBUG).get_logger()

    n = 0
    r, c, env = create_grid()
    actor_critic = GridActorCritic(env, lg, r, c)
    rdr = RenderSimpleGridOneQValues(num_cols=actor_critic.num_cols,
                                     num_rows=actor_critic.num_rows,
                                     plot_style=RenderSimpleGridOneQValues.PLOT_SURFACE,
                                     do_plot=True)

    env.reset()
    cur_state = env.state()

    while True:
        if env.episode_complete():
            # lg.debug("Optimal path cost : " + str(optimal_path_check(env, actor_critic)))
            env.reset()
            actor_critic.new_episode()
            cur_state = env.state()  # new_state
        else:
            action = actor_critic.select_action(cur_state)
            actor_critic.steps_to_goal += 1
            reward = env.execute_action(action)
            new_state = env.state()
            done = env.episode_complete()
            actor_critic.remember(cur_state, action, reward, new_state, done)
            lg.debug(":: " + str(cur_state) + " -> " + str(action) + " = " + str(reward))
            actor_critic.train()
            cur_state = env.state()  # new_state

            # Visualize.
            n += 1
            print("Iteration Number: " + str(n) + " of episode: " + str(actor_critic.get_num_episodes()))
            qgrid = np.zeros((actor_critic.num_rows, actor_critic.num_cols))
            if n % 10 == 0:
                for i in range(0, actor_critic.num_rows):
                    s = ""
                    for j in range(0, actor_critic.num_cols):
                        st = np.array([i, j]).reshape((1, actor_critic.input_dim))
                        q_vals = actor_critic.critic_model.predict(st)[0]
                        s += str(['N', 'S', 'E', 'W'][np.argmax(q_vals)])
                        s += ' , '

                        for actn in range(0, actor_critic.num_actions):
                            x, y = SimpleGridOne.coords_after_action(i, j, actn)
                            if x >= 0 and y >= 0 and x < actor_critic.num_rows and y < actor_critic.num_cols:
                                if qgrid[x][y] == np.float(0):
                                    qgrid[x][y] = q_vals[actn]
                                else:
                                    qgrid[x][y] += q_vals[actn]
                                    qgrid[x][y] /= np.float(2)
                    lg.debug(s)
                rdr.plot(qgrid)
                lg.debug('--------------------')


if __name__ == "__main__":
    main()
