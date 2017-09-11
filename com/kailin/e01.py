import numpy
import pandas
import time

STATES = 20  # 一維世界寬度
ACTIONS= ['left','right'] # 探索者可使用動作
EPSILON= 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 100
FRESH_TIME = 0.0001

def buildQtable():
    return pandas.DataFrame(numpy.zeros([STATES,len(ACTIONS)]),columns=ACTIONS)

def chooseAction(states,qTable):
    stateActions = qTable.iloc[states,:]
    # 非貪婪模式或未進行探索
    if(numpy.random.uniform()>EPSILON)or (stateActions.all()==0):
        actionName = numpy.random.choice(ACTIONS)
    else:
        actionName = stateActions.argmax()
    return actionName

def getEnvFeedback(S,A):
    if A=='right':
        if S == STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S==0:
            S_ = S
        else:
            S_ = S-1
    return S_,R

def updateEnv(S,episode,stepCounter):
    envList = ['-']*(STATES-1)+['T']
    if S == 'terminal':
        interaction = 'Episode %s:totalSteps=%s'%(episode+1,stepCounter)
        print('\r{}'.format(interaction),end='')
        time.sleep(0)
        print('\r                 ',end='')
    else:
        envList[S] = 'o'
        print('\r{}'.format(''.join(envList)),end='')
        time.sleep(FRESH_TIME)

def rl():
    qTable = buildQtable()
    stepCounterList = []
    for e in range(MAX_EPISODES):
        S = 0
        stepCounter = 0
        isTerminal = False
        updateEnv(S,e,stepCounter)
        while not isTerminal:
            A = chooseAction(S,qTable)
            S_,R = getEnvFeedback(S,A)
            if S_ == 'terminal':
                qTarget = R
                isTerminal = True
                stepCounterList.append(stepCounter)
            else:
                qTarget = R + GAMMA * qTable.iloc[S_,:].max()
            qPredict = qTable.ix[S,A]
            qTable.ix[S,A] += ALPHA*(qTarget-qPredict)
            S = S_
            updateEnv(S,e,stepCounter)
            stepCounter+=1

    return qTable,stepCounterList

if __name__ == '__main__':
    qTable,stepList = rl()
    print('\r\nQ-table:\n')
    print(qTable)
    print(stepList)