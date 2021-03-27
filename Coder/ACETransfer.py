import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import uuid
import logging

from Coder.ACE import ACE

def build_ACETransfer(fitness_size, ind_params):
    return ACETransfer(fitness_size=fitness_size,
               unit_number_range=ind_params.unit_num,
               value_boundary=ind_params.value_boundary,
               classes=ind_params.classes,
               layers=ind_params.layers,
               channels=ind_params.channels,
               keep_prob=ind_params.keep_prob,
               drop_path_keep_prob=ind_params.drop_path_keep_prob,
               use_aux_head=ind_params.use_aux_head,
               code_assist=ind_params.transfer_nas_learning_engine)

class ACETransfer(ACE):
    def __init__(self,
                 fitness_size=2,
                 unit_number_range=(10, 20),
                 value_boundary=(0, 15),
                 classes=10,
                 layers=6,
                 channels=48,
                 keep_prob=0.8,
                 drop_path_keep_prob=0.8,
                 use_aux_head=True,
                 code_assist=None,
                 **kwargs):
        super(ACETransfer, self).__init__(
            fitness_size = fitness_size,
            unit_number_range = unit_number_range,
            value_boundary = value_boundary,
            classes = classes,
            layers = layers,
            channels = channels,
            keep_prob = keep_prob,
            drop_path_keep_prob = drop_path_keep_prob,
            use_aux_head = use_aux_head,
            code_assist = code_assist
        )

    def code_init(self):
        if self.code_assist is not None:
            dec = np.array([
                self.code_assist.getTop(10)[random.randint(0,9)]
                if random.random() < 0.8
                else [np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
            low=self.vb[0], high=self.vb[1]), np.random.randint(low=self.vb[0], high=self.vb[1])]
            for _ in range(random.randint(self.unr[0], self.unr[1]))
            ])
        else:
            dec = np.array([[np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
            low=self.vb[0], high=self.vb[1]), np.random.randint(low=self.vb[0], high=self.vb[1])] for i in range(random.randint(self.unr[0], self.unr[1]))])
            logging.error("code_assist is None!")
        return dec

class transferNASLearningEngin:
    def __init__(self):
        self.knowledgeMap = dict()
        self.epslion = 0 #init bais
        self.generalMetric = None

    def parserArchitecture(self, architrecture):
        componentMap = dict()
        architrecture = architrecture.replace("<--->", "-").split("-")
        componentSize = len(architrecture)
        for component in architrecture:
            if component not in componentMap:
                componentMap[component] = 0.0
            componentMap[component] += 1.0/componentSize
        return componentMap

    def preLoad(self, path):
        knowledgeBase = list()
        with open(path) as f:
            for sample in f.readlines():
                [architecture, parameter, error] = sample.split(",")
                #architecture = architecture.replace("<--->","-").split("-")
                #architecture = np.array([np.array([int(i[0]), int(i[1]), int(i[2])]) for i in [unit.split(".") for unit in architecture]])
                parameter = float(parameter.replace("M",""))
                error = float(error)
                knowledgeBase.append((architecture, parameter, error))
        return knowledgeBase

    def preLearning(self, knowledgeBase, knowledgeDimension=1): #[architecture:string , parameter:float, error: float,...]
        if knowledgeBase is None:
            return False
        sampleSize = len(knowledgeBase)
        for i in range(knowledgeDimension):
            self.knowledgeMap[i] = dict()

        for i in range(sampleSize):
            sample = knowledgeBase[i]
            gama = sample[1:]
            componentMap = self.parserArchitecture(sample[0])
            
            for component in componentMap:
                 proportion = componentMap[component]
                 for metric in self.knowledgeMap:
                     if component not in self.knowledgeMap[metric].keys():
                         self.knowledgeMap[metric][component] = [0,0]
                     self.knowledgeMap[metric][component][0] += gama[metric]*proportion
        for metric in self.knowledgeMap:
            for component in self.knowledgeMap[metric]:
                self.knowledgeMap[metric][component][0]/sampleSize
        self.calculateGeneralMetric()
        return True

    def update(self, knowledgeBase):
        if knowledgeBase is None:
            return False
        sampleSize = len(knowledgeBase)
        
        for i in range(sampleSize):
            sample = knowledgeBase[i]
            gama = sample[1:]
            componentMap = self.parserArchitecture(sample[0])
        
            for component in componentMap:
                proportion = componentMap[component]
                for metric in self.knowledgeMap:
                    if component not in self.knowledgeMap[metric].keys():
                        self.knowledgeMap[metric][component] = [0,0]
                        self.knowledgeMap[metric][component][0] += gama[metric]*proportion
                    else:
                        self.knowledgeMap[metric][component][1] = 0.9*self.knowledgeMap[metric][component][1] + 
                                                                  0.1*(gama[metric]*proportion-self.knowledgeMap[metric][component][0])
        self.calculateGeneralMetric()
        return True

    def getTop(self, number, order="INC"):
        if len(self.knowledgeMap) == 0:
            raise Exception("TransferNASLearningEngin does not initialized.")
        if self.generalMetric is None:
            self.calculateGeneralMetric()
        componentList = list()
        generalMetricList = list()
        for component, metric in self.generalMetric:
            componentList.append(np.array([np.int(element) for element in component.split(".")]))
            generalMetricList.append(metric)

        componentList = np.array(componentList)
        #generalMetricList = np.array(generalMetricList)
        if order == "INC":
            index = np.argsort(generalMetricList)
        else:
            index = np.argsort(generalMetricList)[::-1]
        componentList = componentList[index]
        #generalMetricList = generalMetricList[index]
        return np.array(componentList[:number])

    def calculateGeneralMetric(self):
        if self.generalMetric is None:
            self.generalMetric = list()
        generalMetricDic = dict()
        componentSize = len(self.knowledgeMap[0])

        for metric in self.knowledgeMap:
            for component in self.knowledgeMap[metric]:
                if component not in generalMetricDic:
                    generalMetricDic[component] = 0.0
                generalMetricDic[component] += sum(self.knowledgeMap[metric][component])
        
        for component in generalMetricDic:
            self.generalMetric.append((component, generalMetricDic[component]/componentSize))
        