import torch
import numpy as np
import time
from torch.autograd import Variable

class BAB_SHELL():
    def __init__(self):
        self.eps=1e-6
        self.num_variables = 6
        
        self.variables = None
                       
        self.cost_lb = None
        self.cost_ub = None
        
        self.min_cost_lb = None
        self.min_cost_ub = None
        
        self.furture_search = None
        
        self.desk = None
        
        self.expand_ratio = 2**6
        
        self.min_desk = 1e6 # initial is a comparatively large value
        
        self._ini()
        
        print self.variables.shape

    def _ini(self):
        self.variables = np.zeros((1,2*self.num_variables))
        for i in range(self.variables.shape[1]):
            if i%2 == 0:
                self.variables[0,i] = 0
            else:
                self.variables[0,i] = 1

    def run(self):
        self._ini()
        for i in range(10):
            print 'round: ', i
            self._f()
            self._brand_and_bound()
            print 'size'
            print self.variables.shape[1]
            self._expand()
            print 'after expansion'
            print self.variables.shape[1]
    
    def _eval_matrix_yielder(self):
        length_set = 500000
        vector_length = self.variables.shape[0]
        n_itr = vector_length/length_set
        
        if vector_length%length_set != 0:
            n_itr = n_itr + 1
        
        for i in range(n_itr):
            a = i*length_set
            if (i+1)*length_set < vector_length:
                b = (i+1)*length_set
            else:
                b = vector_length
            yield self.variables[a:b,:]
            
    def _f(self):
        ###########
        #print 'x_lb'
        #print x_lb
        for v_np in self._eval_matrix_yielder():
            v = Variable(torch.from_numpy(v_np).float(), volatile = True)

            ft = flowchart()
            rst = ft(v)

            cost_lb = rst[0]
            cost_ub = rst[1]
            if self.cost_lb == None :
                self.cost_lb = cost_lb.data.numpy()
                self.cost_ub = cost_ub.data.numpy()
            else:
                self.cost_lb = np.stack(self.cost_lb, cost_lb.data.numpy(),axis = 0)
                self.cost_ub = np.stack(self.cost_ub, cost_ub.data.numpy(),axis = 0)
    
    def _brand_and_bound(self):
        # Labeling All as need search for next round
        labels = np.ones_like(self.variables[0])
        
        # Labeling those that has equal ub and lb
        labels[np.abs(self.cost_lb-self.cost_ub)<0.001] = -1
        
        for i in range(len(self.variables[0])):
            if labels[i] == -1:
                # Pushing the equal lb,ub to desk container
                self.desk.append(
                    (self.variables[i],
                     self.cost_lb[i],
                     self.cost_ub[i]))
        
        # Update the desk container, remove those not useful
        self.update_desk_min()
        
        # Recalculate the min of ub
        self._total_cost_lower_bound()
        
        # Reupdate the container, remove those that not useful
        self.update_desk_min()
        
        # Labeling those that do not need further search
        temp=self.cost_lb-self.min_total_cost_ub
        labels[temp>0] = 0
        ################
        #print 'labels'
        #print labels
        #print 'Total Cost lower bound'
        #print self.TotalCost_lb
        #print 'min total cost ub'
        #print self.min_total_cost_ub

        self.furture_search = self.variables[labels==1, :]
    
    def update_desk_min(self):
        new_desk = []

        if len(self.desk) > 0:
            for i in self.desk:
                if (i[-1]-self.min_desk) > 0:
                    self.min_desk = i[-1]
        
        for i in self.desk:
            if (i[-1]-self.min_total_cost_ub) > 0:
                pass
            else:
                new_desk.append(i)

        self.desk = new_desk
 
    def _total_cost_lower_bound(self):
        self.min_total_cost_ub = min(np.min(self.cost_ub),self.min_desk)

    def _expand(self):
        origin_size = self.future_search.shape[0]
        self.variables = np.zeros((self.expand_ratio*origin_size, self.variables.shape[1]))
        
        for i in range(self.expand_ratio):
            length = origin_size
            for j in range(self.num_variables):
                judge = i
                for k in range(j):
                    judge = judge/2
                    
                if judge%2 == 0 :
                    a = self.future_search[:,j*2]
                else:
                    a = (self.future_search[:,j*2] + self.future_search[:,j*2+1])*0.5
                
                self.variables[i*length:(i+1)*length, j*2] = a
                self.variables[i*length:(i+1)*length, j*2+1] = a +                     (self.future_search[:, j*2+1] -                     self.future_search[:, j*2])*0.5


class flowchart(torch.nn.Module):
    def __init__(self):
        super(flowchart,self).__init__()
        self.Ttop = [130,120,110,150,140]
        self.Tbot = [140,140,135,120,145]
        self.Pcost = 1
        self.Pump_cost =10
        self._dH= [10,20,5,15,10]
        self.range=10
        
        self.sn_top = None
        self.sn_bot = None
        self.test_size = None
        
        self.min_deltaT = 5
    
    def _ini(self, sn_top, sn_bot, test_size):
        self.sn_top = sn_top
        self.sn_bot = sn_bot
        self.test_size = test_size
        
        self.Tin_top_lb  = Variable(torch.zeros((self.test_size, self.sn_top)).float(),volatile = True)
        self.Tout_top_lb = Variable(torch.zeros((self.test_size, self.sn_bot)).float(),volatile = True)
        self.dH_top= Variable(torch.ones((self.test_size, self.sn_bot)).float(),volatile = True)
        
        self.Tin_bot_lb  = Variable(torch.zeros((self.test_size, self.sn_top)).float(),volatile = True)
        self.Tout_bot_lb = Variable(torch.zeros((self.test_size, self.sn_bot)).float(),volatile = True)
        
        self.Tin_top_ub  = Variable(torch.zeros((self.test_size, self.sn_top)).float(),volatile = True)
        self.Tout_top_ub = Variable(torch.zeros((self.test_size, self.sn_bot)).float(),volatile = True)
        
        self.Tin_bot_ub  = Variable(torch.zeros((self.test_size, self.sn_top)).float(),volatile = True)
        self.Tout_bot_ub = Variable(torch.zeros((self.test_size, self.sn_bot)).float(),volatile = True)
        self.dH_bot = Variable(torch.ones((self.test_size, self.sn_bot)).float(),volatile = True)
        
        self.Cost_Eq_lb = Variable(torch.zeros(self.test_size).float(),volatile = True)
        self.Cost_Utl_lb = Variable(torch.zeros(self.test_size).float(),volatile = True)
        
        self.Cost_Eq_ub = Variable(torch.zeros(self.test_size).float(),volatile = True)
        self.Cost_Utl_ub = Variable(torch.zeros(self.test_size).float(),volatile = True)
        
    def _flowchart(self, p_input):
        p_lb = p_input[:,0:6]
        p_ub = p_input[:,6:12]
        
        # 10 streams
        # Tin/out_top_lb
        # Tin/out_bot_lb
        for i in range(5):
            self.Tin_top_lb[:,i] = self.Ttop[i] + self.range*p_lb[:,i]
            self.Tout_top_lb[:,i] = self.Ttop[i] - 1 + self.range*p_lb[:,i]
            self.dH_top[:,i] = self._dH[i]*self.dH_top[:,i]
            
            self.Tin_bot_lb[:,i] = self.Tbot[i] + self.range*p_lb[:,i] + self.min_deltaT
            self.Tout_bot_lb[:,i] = self.Tbot[i] + 1 + self.range*p_lb[:,i] + self.min_deltaT
            self.dH_bot[:,i] = self._dH[i]*self.dH_bot[:,i]
        
        # Tout_top_ub
        # Tout_bot_ub
        for i in range(5):
            self.Tin_top_ub[:,i] = self.Ttop[i] + self.range*p_ub[:,i]
            self.Tout_top_ub[:,i] = self.Ttop[i] - 1 + self.range*p_ub[:,i]

            self.Tin_bot_ub[:,i] = self.Tbot[i] + self.range*p_ub[:,i] + self.min_deltaT
            self.Tout_bot_ub[:,i] = self.Tbot[i] + 1 + self.range*p_ub[:,i] + self.min_deltaT

            
        #update Tin_top
        self.Tin_top_ub[:,4] = self.Ttop[i]+self.range*p_ub[:,4] + self.range*p_ub[:,5]
        self.Tout_top_ub[:,4] = self.Ttop[i]- 1 + self.range*p_ub[:,4] + self.range*p_ub[:,5]
        
        self.Cost_Eq_lb = p_lb[:,0]+p_lb[:,1]+p_lb[:,2]+p_lb[:,3]+p_lb[:,4]+self.Pump_cost*p_lb[:,5]
        self.Cost_Eq_ub = p_lb[:,0]+p_lb[:,1]+p_lb[:,2]+p_lb[:,3]+p_lb[:,4]+self.Pump_cost*p_lb[:,5]
        
        ###########
        #print 'Tin_top_ub'
        #print self.Tin_top_ub
        #print 'Tin_top_lb'
        #print self.Tin_top_lb
        
        # the following lb means it calculates the lb of Qh
        self.Tin_lb = torch.cat((self.Tin_top_ub,self.Tin_bot_lb), 1)
        self.Tout_lb = torch.cat((self.Tout_top_ub,self.Tout_bot_lb), 1)
        self.dH = torch.cat((self.dH_top, self.dH_bot), 1)
        
        self.Qh_Tin_lb = Variable(torch.zeros(self.Tin_lb.size()))
        self.Qh_Tout_lb = Variable(torch.zeros(self.Tout_lb.size()))
        
        # the following ub means it calculates the ub of Qh
        self.Tin_ub = torch.cat((self.Tin_top_lb,self.Tin_bot_lb), 1)
        self.Tout_ub = torch.cat((self.Tout_top_lb,self.Tout_bot_lb), 1)
        
        self.Qh_Tin_ub = Variable(torch.zeros(self.Tin_ub.size()))
        self.Qh_Tout_ub = Variable(torch.zeros(self.Tout_ub.size()))
        
        # Cal Qh lower and upper bound
        self.CalQh_lb()
        self.CalQh_ub()
        
    def _CalSingleQh_lb(self,T):
        #Hot stream Qh Cal
        #Calculate the deltaT
        T = T.unsqueeze(dim=1)
        ##################
        #print 'Tin_lb and Tout_lb'
        #print self.Tin_ub
        #print self.Tout_ub
        deltaT = (self.Tin_lb - T).clamp(min=0) - (self.Tout_lb - T).clamp(min=0)
        #Calculate the sum of H
        #print 'delta T'
        #print deltaT
        sh = self.dH*deltaT
        #print 'sh'
        #print sh
        #mask those that are cold stream
        sh[(self.Tin_lb - self.Tout_lb)<0] = 0
        #print 'after mask'
        #print sh
        Qh_hot = torch.sum(sh,1)
    
        #Cold stream Qh Cal
        #Calculate the deltaT
        deltaT = (self.Tout_lb - T).clamp(min=0) - (self.Tin_lb - T).clamp(min=0)
        sh = self.dH*deltaT
        #mask those that are hot stream
        sh[(self.Tin_lb - self.Tout_lb)>0] = 0
        #print 'cold stream sh'
        #print sh
        Qh_cold = torch.sum(sh,1)
        return Qh_hot - Qh_cold
    
    def _CalSingleQh(self, Tin, Tout, dH, T):
        #Hot stream Qh Cal
        #Calculate the deltaT
        T = T.unsqueeze(dim=1)
        deltaT = (Tin - T).clamp(min=0) - (self.Tout - T).clamp(min=0)
        #Calculate the sum of H
        sh = dH*deltaT
        #mask those that are cold stream
        sh[(Tin - Tout)<0] = 0
        Qh_hot = torch.sum(sh,1)
    
        #Cold stream Qh Cal
        #Calculate the deltaT
        deltaT = (Tout_ub - T).clamp(min=0) - (Tin_ub - T).clamp(min=0)
        sh = dH*deltaT
        #mask those that are hot stream
        sh[(Tin_ub - Tout_ub)>0] = 0
        Qh_cold = torch.sum(sh,1)
        return Qh_hot - Qh_cold
   
    def CalQh(self, Qh_Tin, Qh_Tout, Tin, Tout, dH):
        #############
        #print 'begin of CalQh_lb'
        for i in range(self.Tin_lb.size()[1]):
            Qh_Tin[:,i] = self._CalSingleQh(Tin[:,i])
        for i in range(self.Tout_lb.size()[1]):
            Qh_Tout[:,i] = self._CalSingleQh(Tout[:,i])
        Qh = torch.min(torch.min(self.Qh_Tin, 1)[0],torch.min(self.Qh_Tout, 1)[0])
        return Qh
        
        
    def forward(self, input_value):
        self._ini((input_value.size()[1])/2-1,(input_value.size()[1])/2-1,input_value.size()[0])
        self._flowchart(input_value)
        return self.Cost_Utl_lb+self.Cost_Eq_lb, self.Cost_Utl_ub+self.Cost_Eq_ub
