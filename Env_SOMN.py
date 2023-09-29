# a biblioteca gym mudou
import gymnasium as gym
from gymnasium import spaces  # Discrete, Box, Tuple,  Dict
from gymnasium import Env

# biblioteca stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3

# outras bibliotecas
import os
import numpy as np
import random
import numpy as np
from absl import flags
import torch
import wandb


class Somn(Env):
    """Custom Environment that follows gym interface."""

    def __init__(
        self,
        M: int,
        N: int,
        Y: int,
        MAXDO: int,
        MAXAM: int,
        MAXPR: int,
        MAXPE: int,
        MAXFT: int,
        MAXMT: int,
        MAXTI: int,
        MAXEU: int,
    ):
        super(Somn).__init__()

        Somn.time = 1
        self.M = M
        self.N = N
        self.Y = Y
        self.MAXDO = MAXDO
        self.MAXAM = MAXAM
        self.MAXPR = MAXPR
        self.MAXPE = MAXPE
        self.MAXFT = MAXFT
        self.MAXMT = MAXMT
        self.MAXTI = MAXTI
        self.MAXEU = MAXEU
        # self.MT = np.random.randint(0,MAXFT,M)
        self.EU = np.random.random(M) * MAXEU
        self.BA = np.random.randint(0, MAXFT, M)
        self.IN = np.random.randint(0, MAXFT, M)
        self.OU = np.random.randint(0, MAXFT, M)

        # self.state = np.zeros((N,5))

        # print('Inicializado', M, N , Y)

        self.DE = [
            Demand(
                M, N, MAXDO, MAXAM, MAXPR, MAXPE, MAXFT, MAXMT, MAXTI, MAXEU, Somn.time
            )
            for _ in range(N)
        ]
        self.YA = [Yard(Y, M, MAXFT) for _ in range(Y)]

        ######################
        #      lb e ub       #
        ######################
        """
    (lb=lowerbound ub=upperbound) para o espaco de Observacao e Acao 
    """

        # ST varia de -2 a 5
        self.lb_ST = -2
        self.ub_ST = 5
        # time varia de 1 a (3*MAXDO + M)
        self.lb_time = 1
        self.ub_time = 10 * self.MAXDO + self.M
        # LT varia de 2 a (M/2 + 2)
        self.lb_LT = 2
        self.ub_LT = int(self.M / 2) + 2
        # DO varia de 3 a (ub_time + ub_LT + MAXDO)
        self.lb_DO = 3
        self.ub_DO = self.ub_time + self.ub_LT + self.MAXDO
        # TP varia de 2 a (ub_time + ub_LT + 2) onde 2 e um ruido
        self.lb_TP = 2
        self.ub_TP = self.ub_time + self.ub_LT + 2
        # MT varia de 0 a MAXFT
        self.lb_MT = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_MT = np.array([MAXFT for _ in range(self.M)]).astype(np.int64)
        # EU varia de 0 a MAXEU
        self.lb_EU = np.array([0 for _ in range(self.M)]).astype(np.float64)
        self.ub_EU = np.array([MAXEU for _ in range(self.M)]).astype(np.float64)
        # BA varia de 0 a MAXFT
        self.lb_BA = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_BA = np.array([MAXFT for _ in range(self.M)]).astype(np.int64)
        # IN varia de 0 a MAXFT
        self.lb_IN = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_IN = np.array([MAXFT for _ in range(self.M)]).astype(np.int64)
        # OU varia de 0 a MAXFT
        self.lb_OU = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_OU = np.array([MAXFT for _ in range(self.M)]).astype(np.int64)

        # lb e ub--- segunda versao (sem a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])

        # # lb e ub--- primeira versao (com a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_time, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_time, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])

        ######################
        #      Espacos       #
        ######################
        """
    Precisa mudar o espaco de acao 
    de acordo com o algoritmo utilizado
    """

        # accept to produce or reject
        # self.action_space = spaces.Box(0, 4, shape=(1,)) # usar o TD3
        self.action_space = spaces.Discrete(self.MAXDO)  # usar com o PPO, DQN, A2C

        # Espaco de observacao (como ficam as demandas depois da acao)
        # self.observation_space = spaces.Box(self.lb, self.ub, dtype=int)
        # self.observation_space = spaces.Dict({'tempo':spaces.Box(self.lb_time, self.ub_time, shape=(1,), dtype=int),
        #                                 'estado': spaces.Box(self.lb, self.ub, dtype=int)})      # versao para MultiInputPolicy
        # self.observation_space = spaces.Dict({'time':spaces.Box(self.lb_time, self.ub_time, shape=(1,), dtype=int),
        #                                'MT': spaces.Box(self.lb_MT, self.ub_MT, dtype=int),
        #                                'EU': spaces.Box(self.lb_EU, self.ub_EU, dtype=float),
        #                                'BA': spaces.Box(self.lb_BA, self.ub_BA, dtype=int),
        #                                'IN': spaces.Box(self.lb_IN, self.ub_IN, dtype=int),
        #                                'OU': spaces.Box(self.lb_OU, self.ub_OU, dtype=int),
        #                                'state': spaces.Box(self.lb, self.ub, dtype=int)})          # versao para MultiInputPolicy

        self.observation_space = spaces.Dict(
            {
                "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "MT": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "EU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "BA": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "IN": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "OU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "state": spaces.Box(
                    low=0.0, high=1.0, shape=(self.N, 4), dtype=np.float64
                ),
            }
        )  # versao para MultiInputPolicy Normalizada

    ######################
    #      funcoes       #
    ######################

    def normaliza(self, x, min, max):
        x_norm = np.array((x - min) / (max - min)).astype(np.float64)
        return x_norm

    def readDemand(self):
        for i in range(Demand.N):
            if (
                self.DE[i].ST == -1
            ):  # or self.DE[i].ST == 0: ZERO não pode ser status de livre
                self.DE[i](Somn.time)

    def match_demand_with_inventory(self, limiar: float, t: int) -> bool:
        matched = False
        for i in range(Demand.N):
            for y in range(Yard.cont):
                match = 0
                # print('Y...', y, 'YA=', YA[y].yard,Yard.cont, 'l=', limiar)
                for j in range(Demand.M):
                    # print('Y(y,j):', y,j, 'Y x D:', self.YA[y].yard[j],self.DE[i].FT[j], 'cont:', Yard.cont, 'l x m:', limiar, match)
                    if self.DE[i].FT[j] > 0:
                        if self.DE[i].FT[j] <= self.YA[y].yard[j]:
                            match = match + 1
                    # se for ZERO então não pode ter a caracteristica
                    else:
                        if self.YA[y].yard[j] == 0:
                            match = match + 1

                if match >= limiar:
                    # print("\n Match: Casou", Yard.cont)
                    self.YA[y].yard = self.YA[
                        Yard.cont - 1
                    ].yard  ## apaga o registro de match com o último da lista
                    Yard.cont -= 1
                    self.DE[i].ST = 3  ## produced status
                    matched = True

            # print("\n Match: Saiu", Yard.cont)
            return matched

    def product_scheduling(self, t: int, action):
        for i in range(self.N):
            if self.DE[i].ST == 1:
                if self.DE[i].DO > (t + self.DE[i].LT + action):
                    self.DE[
                        i
                    ].ST = 3  ## produced status --- remember to run time for each case
                    self.OU -= self.DE[i].FT  ## CONSOME OS RECURSOS
                    self.DE[i].TP = (
                        t + self.DE[i].LT + 3
                    )  #   --- trocar por distribuição poison --- ou por algo que dependa de AM random.randint(1,Demand.MAXTI)
                    # print('\n **** PRODUCED because', self.DE[i].DO, '>', t + self.DE[i].LT + action)
                else:
                    self.DE[i].ST = 2  ## rejected status
                    self.OU -= self.DE[i].FT  ### libera do buffer de produção
                    self.BA += self.DE[i].FT  ## devolve para o saldo para os próximos
                    # print('\n **** REJECTED by DO', self.DE[i].DO, ' <= DI+LT+act', t , self.DE[i].LT , action)

    def product_destination(self, t: int):
        for i in range(Demand.N):
            if self.DE[i].ST == 3:
                if self.DE[i].TP < t:  ### TP eh resultado de LT(#f) + RAND
                    if t < self.DE[i].DO:
                        self.DE[
                            i
                        ].ST = (
                            5  ## produced status --- remember to run time for each case
                        )
                        # print("\n Destination: Enviou", Yard.cont)
                    else:
                        self.DE[i].ST = 4  ## produced status
                        if Yard.cont < Yard.Y - 1:
                            self.YA[Yard.cont].yard = self.DE[i].FT
                            Yard.cont += 1
                            # print("\n Destination: Armazenou no YARD", Yard.cont)
                        else:
                            self.DE[
                                i
                            ].ST = (
                                -2
                            )  ## NAO CABE ... REJEITADO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)

    def stock_covers_demand(self, t):
        covered = True

        for i in range(self.N):
            if self.DE[i].ST == 0:
                DF = self.BA - self.DE[i].FT

                OR = np.array(
                    [abs(i) if i < 0 else 0 for i in DF]
                )  # O QUE PRECISA SER COMPRADO
                # print('\n ORDER from ', DF, ':', OR)
                if not np.any(OR):
                    self.DE[i].ST = 1
                    self.BA -= np.array(DF)  ### ATUALIZA O SALDO
                    self.OU += np.array(DF)  ### ATUALIZA A SAÍDA
                    # print('\n balance:', self.BA,  'because not buying',self.OU)
                else:
                    covered = False
                    self.IN += np.array(OR)  ## ATUALIZA O TOTAL DE COMPRAVEIS
                    # print('\n balance: ', self.BA, 'because buying',OR, 'accumulating', self.IN)
        return covered

    # def order_raw_material(self, t: int):
    # self.IN = [random.randint(0,i) if i > 0 else 0 for i in self.IN]
    # return self.IN

    def eval_final_states(self) -> float:
        totProfit = 0.0
        totReward = 0.0
        totPenalty = 0.0
        for i in range(self.N):
            if self.DE[i].ST == 2:
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                totProfit += self.DE[i].AM * self.DE[i].PR
                # print('REJECTED vvvvvvvvvvvvvvvvvvvvvvvvvvvv')
            if self.DE[i].ST == -2:
                totPenalty += self.DE[i].CO
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                # print('PREJUIZO $$$$$$$$$$$$$$$$$$$$$$$$$')
            if self.DE[i].ST == 4:
                totPenalty += totReward / (
                    Yard.space - Yard.cont + 1
                )  ### penalidade inversamente proporcional ao espaço remanescente
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                totProfit += self.DE[i].AM * self.DE[i].PR
                # print('STORED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if self.DE[i].ST == 5:
                totReward += self.DE[i].AM * self.DE[i].PR
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                totProfit += self.DE[i].AM * self.DE[i].PR
                # print('REWARD ******************************')
        totReward -= totPenalty
        # print ('REW+PEN+PRO', totReward, totPenalty, totProfit)
        return totReward, totPenalty, totProfit

    ######################
    #       step         #
    ######################

    def step(self, action):
        """
        Atualiza tudo aqui e devolve o próximo estado: n_state, reward, done, info
        n_state: próximo estado; reward: recompensa da ação; done: flag de conclusão; info: informaões extras (opcional)
        primeira versão vai fazer uma iteração para cada episódio ... O Tempo t precisa ser controlado
        """

        # receive RAW MATERIAL AND ORDERS (DEMANDS)
        self.MT = np.array([random.randint(0, i) if i > 0 else 0 for i in self.IN])
        self.readDemand()

        # IF PREVIOUS ORDERS INVENTORY AVAILABLE, PLEASE DISPATCH
        if self.match_demand_with_inventory(self.MAXFT / 5, Somn.time):
            self.product_destination(Somn.time)

        # ANYWAY, UPDATE BALANCE AND INCOME RAW MATERIAL REGARDING MT RECEIVED
        self.IN -= self.MT
        self.BA += self.MT

        # IF RAW MATERIAL INVENTORY DOES NOT COVER PLEASE REQUEST RAW MATERIAL
        if not self.stock_covers_demand(Somn.time):
            self.IN = np.array(
                [random.randint(0, i) if i > 0 else 0 for i in self.IN]
            ).astype(np.int64)

        # ANYWAY START PRODUCING AND DISPATCHING
        self.product_scheduling(Somn.time, action)
        self.product_destination(Somn.time)
        Somn.time += 1

        # ORDINARY PROCEDURES IN STEP METHOD INCLUDING REWARD BY INSPECTING FINAL STATES
        # 1 STATE
        arrayState = []

        for i in range(self.N):
            aux_row = [
                self.normaliza(x=self.DE[i].ST, min=self.lb_ST, max=self.ub_ST),
                # Somn.time,
                # self.DE[i].SP,
                self.normaliza(self.DE[i].LT, self.lb_LT, self.ub_LT),
                # self.DE[i].VA,
                # self.DE[i].SU,
                # self.DE[i].PR,
                self.normaliza(self.DE[i].DO, self.lb_DO, self.ub_DO),
                self.normaliza(self.DE[i].TP, self.lb_TP, self.ub_TP),
            ]
            arrayState.append(aux_row)

        self.state = np.array(arrayState)

        # 2 REWARD
        (
            reward,
            penalty,
            exprofit,
        ) = self.eval_final_states()  # aqui vai a função que calcula a recompensa

        # 3 FINAL CONDITION
        done = False
        truncated = False
        # if penalty>0:
        # reward =0
        # print('\n D -- O -- N -- E --', self.state)
        # done = True

        if Somn.time >= self.ub_time:  # 10*Demand.MAXDO + Demand.M   (TEMPOMAX)
            # print('\n D -- O -- N -- E --', self.state)
            done = True

        # Atualiza o upper bounds

        self.ub_MT = max(self.MAXFT, np.amax(self.MT))
        self.ub_BA = max(self.MAXFT, np.amax(self.BA))
        self.ub_IN = max(self.MAXFT, np.amax(self.IN))

        info = {}  # Informações adicionais
        # observation = self.state                                    # by_frederic: retorna quando e um tipo Box
        observation = {
            "time": np.array([self.normaliza(self.time, self.lb_time, self.ub_time)]),
            "MT": self.normaliza(self.MT, self.lb_MT, self.ub_MT),
            "EU": self.normaliza(self.EU, self.lb_EU, self.ub_EU),
            "BA": self.normaliza(self.BA, self.lb_BA, self.ub_BA),
            "IN": self.normaliza(self.IN, self.lb_IN, self.ub_IN),
            "OU": self.normaliza(self.OU, self.lb_OU, self.ub_OU),
            "state": self.state,
        }  # by_frederic: retorna quando e um tipo Dict

        return (
            observation,
            reward,
            done,
            truncated,
            info,
        )  # , exprofit   # by_frederic:

    ######################
    #       reset        #
    ######################

    def reset(self):
        self.MT = np.random.randint(0, self.MAXFT, self.M)
        self.EU = np.random.random(self.M) * self.MAXEU
        self.BA = np.random.randint(0, self.MAXFT, self.M)
        self.IN = np.random.randint(0, self.MAXFT, self.M)
        self.OU = np.random.randint(0, self.MAXFT, self.M)
        Somn.time = 1

        self.YA = [Yard(self.Y, self.M, self.MAXFT) for _ in range(self.Y)]

        arrayState = []
        for i in range(self.N):
            self.DE[i](Somn.time)
            aux_row = [
                self.normaliza(x=self.DE[i].ST, min=self.lb_ST, max=self.ub_ST),
                # Somn.time,
                # self.DE[i].SP,
                self.normaliza(x=self.DE[i].LT, min=self.lb_LT, max=self.ub_LT),
                # self.DE[i].VA,
                # self.DE[i].SU,
                # self.DE[i].PR,
                self.normaliza(x=self.DE[i].DO, min=self.lb_DO, max=self.ub_DO),
                self.normaliza(x=self.DE[i].TP, min=self.lb_TP, max=self.ub_TP),
            ]
            arrayState.append(aux_row)

        self.state = np.array(arrayState)

        info = dict()
        # observation = (self.state, info)                           # by_frederic retorna quando o tipo é Box
        observation = {
            "time": np.array([self.normaliza(self.time, self.lb_time, self.ub_time)]),
            "MT": self.normaliza(self.MT, self.lb_MT, self.ub_MT),
            "EU": self.normaliza(self.EU, self.lb_EU, self.ub_EU),
            "BA": self.normaliza(self.BA, self.lb_BA, self.ub_BA),
            "IN": self.normaliza(self.IN, self.lb_IN, self.ub_IN),
            "OU": self.normaliza(self.OU, self.lb_OU, self.ub_OU),
            "state": self.state,
        }  # by_frederic: retorna quando e um tipo Dict

        return (observation, info)  # by_frederic: para se adequar ao Gymnasium

    ######################
    #       render       #
    ######################

    def render(self):
        # print("Current state (RENDER): \n", self.state)
        pass

    ######################
    #       close        #
    ######################

    def close(self):
        pass
