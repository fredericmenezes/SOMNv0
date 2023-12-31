# -*- coding: utf-8 -*-
"""SOMNLearning_VAAf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sWkvj27xFtDFLB93mbAz8CtOzKHmbSYc
"""

# -*- coding: utf-8 -*-
"""SomnLearning_VVA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xb_4igG3Uwx4kMRZj4o7X_OzLe3JHTcG

# SOMN+RL
https://medium.com/turing-talks/pouse-um-m%C3%B3dulo-lunar-com-deep-q-learning-1f4395ea764
"""

import gym
import tensorflow as tf
import numpy as np
import random
from absl import flags
from gym import Env
from gym.spaces import Discrete, Tuple, Box
import torch



def fazerDQN(alpha, n_acoes, input_dims, fc1, fc2):
    layers = tf.keras.layers
    DQN = tf.keras.models.Sequential()
    DQN.add(layers.Flatten(input_shape=(input_dims)))
    DQN.add(layers.Dense(fc1, activation='relu'))
    #### dropout  model.add(Dropout(0.2))
    DQN.add(layers.Dense(fc2, activation='relu'))
    DQN.add(layers.Dense(n_acoes, activation='relu'))
    
    DQN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss='huber_loss')
    return DQN

class ExperienceReplay():
    def __init__(self, mem_max, input_dims):
        self.mem_max = mem_max
        #tamanho máximo
        self.mem_counter = 0
        #contador de elementos


        self.s = np.zeros((self.mem_max, *input_dims), dtype=np.float32)
        self.s2 = np.zeros((self.mem_max, *input_dims),dtype=np.float32)
        self.r = np.zeros(self.mem_max, dtype = np.float32)
        self.a = np.zeros(self.mem_max, dtype = np.int32)
        self.terminal = np.zeros(self.mem_max, dtype=np.int32)

    def salvar_experiencia(self, s, a, r, s2, teminado):
      index = self.mem_counter % self.mem_max
      self.s[index] = s # Estado 1
      self.s2[index] = s2 # Estado 2
      self.r[index] = r # Reward da Experiência
      self.a[index] = a # Ação da Experiência
      self.terminal[index] = 1 - int(teminado) # Estado terminal?
      self.mem_counter += 1

    def amostra_aleatoria(self, tamanho_amostra):
        mem_max = min(self.mem_counter, self.mem_max)
        amostra = np.random.choice(mem_max, tamanho_amostra, replace=False)
        s = self.s[amostra]
        s2 = self.s2[amostra]
        r = self.r[amostra]
        a = self.a[amostra]
        terminal = self.terminal[amostra]
        return s, a, r, s2, terminal

class Agente():
    def __init__(self, alpha, gamma, n_acoes, epsilon, tamanho_amostra,
                 input_dims, epsilon_dec=1e-4, epsilon_end=0.01,
                 mem_max=100000, fname='dqn_saveVAft02.h5' ):
        self.acoes = [i for i in range(n_acoes)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.tamanho_amostra = tamanho_amostra
        self.model_file = fname
        self.memoria = ExperienceReplay(mem_max, input_dims)
        self.q_eval = fazerDQN(alpha, n_acoes, input_dims, 256,256)

    def salvar_experiencia(self, s, a, r, s2, terminado):
        self.memoria.salvar_experiencia(s, a, r, s2, terminado)

    def escolher_acao(self, obs):
        if np.random.random() < self.epsilon:
          acao = np.random.choice(self.acoes)
        else:
          s = np.array([obs])      
          acoes = self.q_eval.predict(s)
          acao = np.argmax(acoes)

        return acao

    def aprender(self):
            if self.memoria.mem_counter < self.tamanho_amostra:
                return
            s, a, r, s2, terminados = self.memoria.amostra_aleatoria(self.tamanho_amostra)
            q_eval = self.q_eval.predict(s)
            q_next = self.q_eval.predict(s2)
            #print ('\n A P R E N D E N D O .............', q_eval, q_next)
            q_target = np.copy(q_eval)
            batch_index = np.arange(self.tamanho_amostra, dtype=np.int32)
            q_target[batch_index, a] = r + self.gamma*np.max(q_next, axis=1)*terminados
            loss = self.q_eval.train_on_batch(s, q_target)
            if self.epsilon > self.eps_min:
                self.epsilon = self.epsilon - self.epsilon_dec
#            else:
#                self.eps_min

            return loss

    def save_model(self):
            self.q_eval.save(self.model_file)
    def load_model(self):
            self.q_eval = tf.keras.models.load_model(self.model_file)

# ENVIRONMENT


class Yard:
  
  def __init__(self, Y, numFeat, typFeat):
 
    Yard.Y=Y
    Yard.yard = [0 for _ in range(numFeat)]
    Yard.cont=0
    Yard.space = Y

class Demand:
  cont=1
#  Y=20,M=10,N=10,MAXDO=10,MAXAM=5,MAXPR=1.5,MAXPE=50,MAXFT=5,MAXMT=3,MAXTI=2,MAXEU = 30
  def __init__(self,M:int,N:int, MAXDO:int, MAXAM:int, MAXPR:float, MAXPE:int, MAXFT:int, MAXMT:int, MAXTI:int, MAXEU:int, t: int):
    Demand.M=M
    Demand.N=N
    Demand.MAXDO=MAXDO
    Demand.MAXAM=MAXAM
    Demand.MAXPR=MAXPR
    Demand.MAXPE=MAXPE
    Demand.MAXFT=MAXFT
    Demand.MAXMT=MAXMT
    Demand.MAXTI=MAXTI
    Demand.MAXEU=MAXEU
    Demand.EU = np.random.random(M)*MAXEU
    self.ST = -1                  ###free(-1) received0, ready1, rejected2, produced3, stored4 and delivered5   

 #   self(t) 
    Demand.cont +=1
  
  def __call__(self, t:int):
    
    self.CU = Demand.cont
#    self.PR = random.randrange(3,Demand.MAXPR)  below -----------------
    self.AM = 1 # random.randrange(1,Demand.MAXAM)
    self.PE = random.randint(1,Demand.MAXPE)
    self.ST = 0                  ###received0, ready1, rejected2, produced3, stored4 and delivered5
    self.FT = np.random.randint(0,Demand.MAXFT,self.M) 
    if not np.any(self.FT):
        self.FT[1] = 1

### Tempo    
    self.F = 0
    for i in range(self.M):
      self.F += int(self.FT[i]>0)

    self.LT = int(self.F/2) + 2                      ###  --- 1.0*self.fun_tau() * self.F
    self.DI = t
    self.DO = t + self.LT + random.randint(0,Demand.MAXDO)
    
    self.CO = 0.0
    for j in range(Demand.M):
      self.CO += self.FT[j] * Demand.EU[j]
    self.CO = self.AM * self.CO
#    self.PR = Demand.MAXPR*self.CO
    self.PR = Demand.MAXPE

    self.SP = 1.0*self.fun_gamma() ####* Yard.Y   #SPACE CONSUMPTION FACTOR
    self.VA = 1.0*self.fun_upsilon() 
    self.SU = 1.0*self.fun_sigma() 
    self.TP = self.DO - t

  def fun_gamma(self) -> float:
    x = (self.AM*self.F)/(Demand.MAXAM * self.M)
    return x

  def fun_tau(self) -> float:
    x = (self.AM*self.F)/(Demand.MAXAM * self.M)
    return x

  def fun_upsilon(self) -> float:
    x = self.F/self.M
    return x

  def fun_sigma(self) -> float:
    x = self.F/self.M
    return x
  
#  def fun_beta(self, IN, OU) -> float:
 #   x=0
#    for i in range(self.M):
#      if IN[i]==OU[i]:
#        x+=1
#    x = x/self.M
#    return x
     
#  def calculate_statics(self):


class Somn(Env):

#  def __init__(self):
  def __init__(self, M:int,N:int,Y:int, MAXDO:int, 
               MAXAM:int, MAXPR:int, MAXPE:int, MAXFT:int, 
               MAXMT:int, MAXTI:int, MAXEU:int):
    Somn.time = 1
    self.M=M
    self.N=N
    self.Y=Y
    self.MAXDO=MAXDO
    self.MAXAM=MAXAM
    self.MAXPR=MAXPR
    self.MAXPE=MAXPE
    self.MAXFT=MAXFT
    self.MAXMT=MAXMT
    self.MAXTI=MAXTI
    self.MAXEU=MAXEU
    #self.MT = np.random.randint(0,MAXFT,M)
    self.EU = np.random.random(M)*MAXEU 
    self.BA = np.random.randint(0,MAXFT,M)
    self.IN = np.random.randint(0,MAXFT,M)
    self.OU = np.random.randint(0,MAXFT,M)

#    self.state = []

    print('Inicializado', M, N , Y)

    self.DE = [Demand(M,N,MAXDO,MAXAM,MAXPR,MAXPE,MAXFT,MAXMT,MAXTI,MAXEU,Somn.time) for _ in range(N)]
    self.YA = [Yard(Y,M,MAXFT) for _ in range(Y)]

#### demais inicializações
    self.observation_space = Box(low=0, high=1, shape=(N, 5))
    self.action_space = Discrete(self.MAXDO) # accept to produce or reject


  def readDemand(self):
    for i in range(Demand.N):
      if self.DE[i].ST == -1:    # or self.DE[i].ST == 0: ZERO não pode ser status de livre
        self.DE[i](Somn.time)
 
  def   match_demand_with_inventory(self,limiar: float, t:int)->bool:
    matched = False
    for i in range(Demand.N):
      for y in range(Yard.cont):
        match=0
  #      print('Y...', y, 'YA=', YA[y].yard,Yard.cont, 'l=', limiar)
        for j in range(Demand.M):
          #print('Y(y,j):', y,j, 'Y x D:', self.YA[y].yard[j],self.DE[i].FT[j], 'cont:', Yard.cont, 'l x m:', limiar, match)
          if self.DE[i].FT[j] > 0:
            if self.DE[i].FT[j] <= self.YA[y].yard[j]:
              match=match+1
          ### se for ZERO então não pode ter a caracteristica
          else:
            if self.YA[y].yard[j]==0:
              match=match+1
              
        if match >= limiar:
          #PRINT("\n Match: Casou", Yard.cont)
          self.YA[y].yard = self.YA[Yard.cont-1].yard  ## apaga o registro de match com o último da lista
          Yard.cont -=1
          self.DE[i].ST = 3  ## produced status
          matched = True

      ##PRINT("\n Match: Saiu", Yard.cont)
      return matched

  def product_scheduling (self,t:int, action):
    for i in range(self.N):
      if self.DE[i].ST == 1:
        if self.DE[i].DO  > (t + self.DE[i].LT + action):
          self.DE[i].ST = 3  ## produced status --- remember to run time for each case
          self.OU -= self.DE[i].FT ## CONSOME OS RECURSOS
          self.DE[i].TP = t + self.DE[i].LT  + 2 #   --- trocar por distribuição poison --- ou por algo que dependa de AM random.randint(1,Demand.MAXTI) 
          #PRINT('\n **** PRODUCED because', self.DE[i].DO, '>', t + self.DE[i].LT + action)
        else:
          self.DE[i].ST = 2  ## rejected status
          self.OU -= self.DE[i].FT  ### libera do buffer de produção
          self.BA += self.DE[i].FT ## devolve para o saldo para os próximos
          #PRINT('\n **** REJECTED by DO', self.DE[i].DO, ' <= DI+LT+act', t , self.DE[i].LT , action)
  

  def product_destination(self,  t: int):
    for i in range(Demand.N):
        if self.DE[i].ST == 3:
          if self.DE[i].TP < t:   ### TP eh resultado de LT(#f) + RAND
            if t < self.DE[i].DO:
              self.DE[i].ST = 5  ## produced status --- remember to run time for each case
              #PRINT("\n Destination: Enviou", Yard.cont)
            else:
              self.DE[i].ST = 4  ## produced status
              if Yard.cont < Yard.Y-1:
                self.YA[Yard.cont].yard = self.DE[i].FT
                Yard.cont += 1
                #PRINT("\n Destination: Armazenou no YARD", Yard.cont)
              else:
                self.DE[i].ST = -2  ## NAO CABE ... REJEITADO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)
  
  def stock_covers_demand(self, t):
    covered = True
    
    for i in range(self.N):
      if self.DE[i].ST == 0:

        DF = self.BA - self.DE[i].FT

        OR = np.array([abs(i) if i < 0 else 0 for i in DF])   # O QUE PRECISA SER COMPRADO
        ##PRINT('\n ORDER from ', DF, ':', OR)
        if not np.any(OR):
          self.DE[i].ST=1 
          self.BA -=  np.array(DF)    ### ATUALIZA O SALDO
          self.OU += np.array(DF)   ### ATUALIZA A SAÍDA
          ##PRINT('\n balance:', self.BA,  'because not buying',self.OU)
        else:
          covered = False
          self.IN += np.array(OR)  ## ATUALIZA O TOTAL DE COMPRAVEIS 
          ##PRINT('\n balance: ', self.BA, 'because buying',OR, 'accumulating', self.IN)
    return covered

 # def order_raw_material(self, t: int):
 #   self.IN = [random.randint(0,i) if i > 0 else 0 for i in self.IN]
 #   return self.IN

  def eval_final_states(self)->float:
    totProfit = 0.0
    totReward = 0.0
    totPenalty = 0.0
    for i in range(self.N):
      if self.DE[i].ST == 2:
        self.DE[i].ST = -1      # LIBERA O ESPAÇO APÓS CONTABILIZADO
        totProfit += (self.DE[i].AM * self.DE[i].PR)
        #PRINT('REJECTED vvvvvvvvvvvvvvvvvvvvvvvvvvvv')
      if self.DE[i].ST == -2:
        totPenalty += self.DE[i].CO
        self.DE[i].ST = -1      # LIBERA O ESPAÇO APÓS CONTABILIZADO
        #PRINT('PREJUIZO $$$$$$$$$$$$$$$$$$$$$$$$$')
      if self.DE[i].ST == 4:
        totPenalty += totReward / (Yard.space - Yard.cont+1) ### penalidade inversamente proporcional ao espaço remanescente
        self.DE[i].ST = -1      # LIBERA O ESPAÇO APÓS CONTABILIZADO
        totProfit += (self.DE[i].AM * self.DE[i].PR)
        #PRINT('STORED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
      if self.DE[i].ST == 5:
        totReward += (self.DE[i].AM * self.DE[i].PR)
        self.DE[i].ST = -1      # LIBERA O ESPAÇO APÓS CONTABILIZADO
        totProfit += (self.DE[i].AM * self.DE[i].PR)
        #PRINT('REWARD ******************************')
    totReward -= totPenalty
    print ('REW+PEN+PRO', totReward, totPenalty, totProfit)
    return totReward, totPenalty, totProfit

  def reset(self):
    self.MT = np.random.randint(0,self.MAXFT,self.M)
    self.EU = np.random.random(self.M)*self.MAXEU 
    self.BA = np.random.randint(0,self.MAXFT,self.M)
    self.IN = np.random.randint(0,self.MAXFT,self.M)
    self.OU = np.random.randint(0,self.MAXFT,self.M)
    Somn.time = 0

    self.YA = [Yard(self.Y,self.M,self.MAXFT) for _ in range(self.Y)]

    arrayState = []
    for i in range(self.N):
      self.DE[i](Somn.time)
      aux_row = [self.DE[i].ST,
                 Somn.time,
                 self.DE[i].LT,
#                 self.DE[i].VA,
#                 self.DE[i].SU,
#                 self.DE[i].PR,
                self.DE[i].DO,
                 self.DE[i].TP]
      arrayState.append(aux_row)

    #self.state = torch.from_numpy(np.array(arrayState))
    self.state = np.array(arrayState)
    return self.state




  def step(self, action):
    # Atualiza tudo aqui e devolve o próximo estado: n_state, reward, done, info
    # n_state: próximo estado; reward: recompensa da ação; done: flag de conclusão; info: informaões extras (opcional)
    # primeira versão vai fazer uma iteração para cada episódio ... O Tempo t precisa ser controlado

### receive RAW MATERIAL AND ORDERS (DEMANDS)    
    self.MT = np.array([random.randint(0,i) if i > 0 else 0 for i in self.IN])
    self.readDemand()

### IF PREVIOUS ORDERS INVENTORY AVAILABLE, PLEASE DISPATCH
    if self.match_demand_with_inventory(self.MAXFT/5, Somn.time):  
      self.product_destination(Somn.time)

### ANYWAY, UPDATE BALANCE AND INCOME RAW MATERIAL REGARDING MT RECEIVED    
    self.IN -= self.MT
    self.BA += self.MT

### IF RAW MATERIAL INVENTORY DOES NOT COVER PLEASE REQUEST RAW MATERIAL
    if not self.stock_covers_demand(Somn.time):          
      self.IN = [random.randint(0,i) if i > 0 else 0 for i in self.IN] 

### ANYWAY START PRODUCING AND DISPATCHING
    self.product_scheduling (Somn.time, action)
    self.product_destination(Somn.time)
    Somn.time += 1

### ORDINARY PROCEDURES IN STEP METHOD INCLUDING REWARD BY INSPECTING FINAL STATES
### 1 STATE
    arrayState = []

    for i in range(self.N):
      aux_row = [self.DE[i].ST,  ### -2 --- vai ate --- +5
                 Somn.time,      ### 1 --- vai ate ---  ub_time (3*MAXDO + M)
#                 self.DE[i].SP,
                 self.DE[i].LT,   ### 2 --- vai ate --- ub_LT (M/2 + 2)
#                 self.DE[i].VA,
#                 self.DE[i].SU,
#                 self.DE[i].PR,
                self.DE[i].DO,    ## inicia com 3 (t + LT) ---- vai ate ----  ub_DO (ub_time + ub_LT + MAXDO)
                self.DE[i].TP]    ### 2 --- vai ate --- ub_time + ub_LT  + 2
      arrayState.append(aux_row)

    self.state = np.array(arrayState)

### 2 REWARD                                 
    reward, penalty, exprofit = self.eval_final_states() # aqui vai a função que calcula a recompensa

### 3 FINAL CONDITION
    done = False
    # if penalty>0:
    #   reward =0
    #   #print('\n D -- O -- N -- E --', self.state)
    #   done = True
    
    if Somn.time >= 3*Demand.MAXDO+Demand.M:   ### ub_time = 3*Demand.MAXDO+Demand.M
      #print('\n D -- O -- N -- E --', self.state)
      done = True
    
    info = {} # Informações adicionais
    return self.state, reward, done, info, exprofit


  def render(self):
    print("Current state (RENDER): ", self.state)


# PRINCIPAL


env = Somn(Y=10,M=5,N=5,MAXDO=5,MAXAM=5,MAXPR=1.5,MAXPE=10,MAXFT=5,MAXMT=3,MAXTI=2,MAXEU = 10)
lr = 0.001
n_games = 940
#PRINT('\n shapes', env.observation_space.shape,env.action_space.n)

agent = Agente(gamma=0.99, epsilon = 1.0, alpha=lr, input_dims=env.observation_space.shape,
              n_acoes=env.action_space.n, mem_max=100000,
              tamanho_amostra=64, epsilon_end=0.01)
try:
    agent.load_model()
except:
    pass
scores = []
profit = []
avg_score = []
avg_profi = []
losses = []
avg_loss = []
avg_act = []
actiones = []
actions = []
rewards = []
for epi in range(n_games):
    done = False
    score = 0
    xcore = 0
    loss = 0

    observation = env.reset()
    while not done:
        action = agent.escolher_acao(observation)        
        observation_, reward, done, info, exprofit = env.step(action)
        score += reward
        xcore += exprofit
        actions.append(action)
        rewards.append(reward)
        agent.salvar_experiencia(observation, action, reward, observation_, done)
        observation = observation_
        
        loss_ = agent.aprender()
        if isinstance(loss_, type(None)):
            loss += 0
        else:
            loss += loss_

    scores.append(score)
    losses.append(loss)
    profit.append(xcore)
#    actiones.append(actions)
    
#    if epi >= 100:
    avg_score.append(np.mean(scores[-100:]))
    avg_profi.append(np.mean(profit[-100:]))
    avg_loss.append(np.mean(losses[-100:]))
#      avg_act.append(np.mean(actiones[-100:]))
    
    print("jogos: ", epi, "pontuação: %.2f" % score, "action ", action, "epsilon: %.2f" % agent.epsilon, "reward: %.2f" % reward)
agent.save_model()

"""# PLOTTAGEM"""

x = [i+1 for i in range(100, n_games)]
f = open("log008.txt","w")
f.writelines(str(x))
f.writelines(str('\n avg scores \n'))
f.writelines(str(avg_score))
f.writelines(str('\n avg profit \n'))
f.writelines(str(avg_profi))
f.writelines(str('\n avg loss \n'))
f.writelines(str(avg_loss))
#f.writelines(str('\n avg act \n'))
#f.writelines(str(avg_act))
f.writelines(str('\n Rewards \n'))
f.writelines(str(rewards))
f.writelines(str('\n Action \n'))
f.writelines(str(actions))
f.close()

import matplotlib.pyplot as plt

#x = [i+1 for i in range(100, n_games)]
plt.plot(x, avg_score[100:])
plt.title("Curva de aprendizado")
plt.xlabel("Número de Jogos")
plt.ylabel("Pontuação")
plt.show()
plt.plot(x, avg_loss[100:])
plt.title("Loss")
plt.xlabel("Número de Jogos")
plt.ylabel("loss")
plt.show()

#plt.plot(x, avg_act)
#plt.title("Actions")
#plt.xlabel("Episode")
#plt.ylabel("action avg")
#plt.show()


# plt.plot(x, avg_profi)
# plt.title("Expected profit")
# plt.xlabel("Number of episodes")
# plt.ylabel("Profit")
# plt.show()