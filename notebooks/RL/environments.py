import numpy as np

class windy_gridworld():
    '''
        The class implements the environment of windy gridworld (see Sutton & Barto (2018) example 6.5).
    '''
    
    def __init__(self, gridwind_dict, goal_state, start_state=None, prob=0):
        '''
            params:
                gridwind_dict: a dictionary in which keys represent states and values represent the strength of upwind
                start_state: start state
                goal_state: goal state
                prob: Probability that the wind varying by +1/-1 from the mean values
        '''
        self.action_list = ['up', 'down', 'right', 'left']
        self.wind = gridwind_dict
        self.state_dict = {}
        k = 0
        for i in gridwind_dict.keys():
            self.state_dict[i] = k
            k += 1
            
        self.goal_state = goal_state
        self.start_state = start_state
        self.prob = prob 
        
    def seed(self, random_seed=None):
        np.random.seed(random_seed)
        
    def reset(self, state=None):
        '''
            Reset the environment using the provided state, using specified start state if none given
        '''
        if state:
            self.state = state
        else:
            if self.start_state:
                self.state = self.start_state
            else:
                self.state = np.random.choice(self.state_dict.keys())
                
        if self.state == self.goal_state:
            self.terminated = True
        else:
            self.terminated = False

            
    def step(self, a):
        '''
            Simulate rewards and next state based on current state s and action a
        '''
        assert a in self.action_list, 'Illegal action taken'
        assert self.terminated==False, 'Already terminated'
        
        s = self.state
        
        if s == self.goal_state:
            self.state = s
            self.terminated = True
            return 0
        else:
            up = self.wind[s] + np.random.choice([-1,0,1],p=[self.prob, 1-2*self.prob, self.prob])
            downflag, upflag = False, False
            while (s[0], s[1]+up) not in self.state_dict:
                if up > 0:
                    up -= 1
                    downflag = True
                else:
                    up += 1
                    upflag = True
                
            temp = (s[0], s[1]+up)
            
            if a=='up':
                if not upflag:
                    s_next = (temp[0],temp[1]+1)
                else:
                    s_next = temp
            elif a=='down':
                if not downflag:
                    s_next = (temp[0],temp[1]-1)
                else:
                    s_next = temp
            elif a=='right':
                s_next = (temp[0]+1,temp[1])
            elif a=='left':
                s_next = (temp[0]-1,temp[1])
                
            if s_next not in self.state_dict: 
                # Hit boundary
                s_next = temp
            
            self.state = s_next
            if self.state == self.goal_state:
                self.terminated = True
            return -1
        

        import numpy as np

class cliff_walking():
    '''
        The class implements the environment of cliff walking (see Sutton & Barto (2018) example 6.6).
    '''
    
    def __init__(self, size, prob=1):
        '''
            params:
                size: size of the grid, in the format of (x,y)
                prob: Probability that fall off the cliff
        '''
        self.size = size
        self.action_list = ['up', 'down', 'right', 'left']
        self.state_dict = {}
        k = 0
        for x in range(size[0]):
            for y in range(size[1]):
                self.state_dict[(x,y)] = k
                k += 1
            
        self.goal_state = (size[0]-1,0)
        self.start_state = (0,0)
        self.prob = prob 
        
    def seed(self, random_seed=None):
        np.random.seed(random_seed)
        
    def reset(self, state=None):
        '''
            Reset the environment using the provided state, using specified start state if none given
        '''
        if state:
            self.state = state
        else:
            if self.start_state:
                self.state = self.start_state
            else:
                self.state = np.random.choice(self.state_dict.keys())
                
        if self.state == self.goal_state:
            self.terminated = True
        else:
            self.terminated = False

            
    def step(self, a):
        '''
            Simulate rewards and next state based on current state s and action a
        '''
        assert a in self.action_list, 'Illegal action taken'
        assert self.terminated==False, 'Already terminated'
        
        s = self.state
        
        if s == self.goal_state:
            self.state = s
            self.terminated = True
            return 0
        else:
            if a=='up':
                s_next = (s[0],min(s[1]+1,self.size[1]-1))
            elif a=='down':
                s_next = (s[0],max(s[1]-1,0))
            elif a=='left':
                s_next = (max(0,s[0]-1),s[1])
            else:
                s_next = (min(self.size[0]-1,s[0]+1),s[1])
        
            reward = -1
            if (s_next[1]==0) and (s_next[0]>0) and (s_next[0]<self.size[0]-1):
                # Steping on the cliff
                if np.random.uniform(0,1) < self.prob:
                    s_next = (0,0)
                    reward = -100
            
            self.state = s_next
            if self.state == self.goal_state:
                self.terminated = True
            return reward