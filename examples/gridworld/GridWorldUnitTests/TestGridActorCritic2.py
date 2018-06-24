import logging
from typing import Tuple

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
# The curr_coords is the grid coordinates (x,y) shape (1,2)
#
step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL


def create_grid() -> Tuple[int, int, SimpleGridOne]:
    r = 20
    c = 20
    grid = np.full((r, c), step)
    grid[10, 10] = goal
    # grid[40, 40] = goal
    # grid[25, 25] = fire
    # grid[10, 40] = fire
    # grid[40, 10] = fire
    sg1 = SimpleGridOne(grid_id=1,
                        grid_map=grid,
                        respawn_type=SimpleGridOne.RESPAWN_CORNER)
    return r, c, sg1


#
# What is the cost of finding a goal curr_coords from each corner. Actions are predicted by the actor
# only, so no random actions. As such this can be taken as a measure of the agents improvement in
# heading directly to a goal.
#
def optimal_path_check(grid_env: SimpleGridOne,
                       actor_critic: GridActorCritic,
                       lg):
    actor_critic.training_mode_off()

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
        lg.debug("Corner: " + str(corner))
        s = ""
        while not grid_env.episode_complete() and i < 5000:
            path_cost += grid_env.execute_action(
                actor_critic.select_action(cur_state=grid_env.curr_coords(), greedy=True))
            s += "{" + str(grid_env.curr_coords()) + "} - "
            i += 1
        lg.debug(s)

    actor_critic.training_mode_on()
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

    rdra = RenderSimpleGridOneQValues(num_cols=actor_critic.num_cols,
                                      num_rows=actor_critic.num_rows,
                                      plot_style=RenderSimpleGridOneQValues.PLOT_SURFACE,
                                      do_plot=True)

    env.reset()
    cur_state = env.curr_coords()

    while True:
        if env.episode_complete():
            if actor_critic.get_num_episodes() > 10:
                lg.debug("Optimal path cost : " + str(optimal_path_check(env, actor_critic, lg)))
            env.reset()
            actor_critic.new_episode()
            cur_state = env.curr_coords()  # new_state
        else:
            lg.debug("------------ Select Action")
            action = actor_critic.select_action(cur_state)
            lg.debug("------------ Execute Action")
            reward = env.execute_action(action)
            new_state = env.curr_coords()
            done = env.episode_complete()
            lg.debug("------------ Update Replay Memory")
            actor_critic.remember(cur_state, action, reward, new_state, done)
            lg.debug("-S-A-R-S'-D- " + str(cur_state) +
                     " -> " + str(action) +
                     " = " + str(reward) +
                     " -> " + str(new_state) +
                     " : " + str(done)
                     )
            lg.debug("------------ Train")
            actor_critic.train()
            lg.debug("------------ Next Iteration")

            if actor_critic.steps_to_goal % 50 == 0:
                lg.debug("*********** Re-spawn :" + str(env.curr_coords()))
                env.reset()
                actor_critic.steps_to_goal = 1
            else:
                actor_critic.steps_to_goal += 1
            cur_state = env.curr_coords()  # new_state

            # Visualize.
            n += 1
            print("Iteration Number: " + str(n) + " of episode: " + str(actor_critic.get_num_episodes()))
            if n % 10 == 0:
                rdr.plot(actor_critic.qvalue_grid(average=False))
            if n % 11 == 0:
                rdra.plot(env.activity_matrix())
            lg.debug('--------------------')


if __name__ == "__main__":
    main()
