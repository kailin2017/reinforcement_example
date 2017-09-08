from com.kailin.e02.maze import Maze
from com.kailin.e02.rl import QLearningTable

def Update():
    for e in range(100):
        stepCounter =0
        observation = env.reset()
        while True:
            stepCounter +=1
            env.render()

            action = RL.choose_action(str(observation))

            observation_,reward,done = env.step(action)

            RL.learn(str(observation),action,reward,str(observation_))

            observation = observation_
            if done:
                break;
        print('GameOver',stepCounter)

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100,Update)
    env.mainloop()