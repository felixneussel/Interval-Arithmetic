# This module contains the classes Interval and Intervalvector.
# In the class Interval, the magic methods for standard arithmetic operations
# have been overwritten. Arithmetic operations on intervals can thus be coded
# in the same way as for numeric types.
# In the class IntervalVector, the __getitem__() magic method is overwritten,
# enabling us to treat IntervalVectors like numpy arrays.
import numpy as np
    
class Interval:
    def __init__(self,lb,ub):
        if lb > ub:
            raise ValueError('Lower interval bound must be smaller than or equal to the upper bound')
        self.lb = lb
        self.ub = ub
        self.box_mean = 0.5*lb + 0.5*ub

    def __str__(self):
        return f'[{self.lb} , {self.ub}]'
    
    def __add__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            other = Interval(other,other)
        return Interval(self.lb + other.lb, self.ub + other.ub)
    
    def __neg__(self):
        return Interval(-self.ub,-self.lb)
    
    def __sub__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            other = Interval(other,other)
        return self.__add__(-other)
    
    def __rsub__(self,other):
            return Interval(other - self.ub,other-self.lb)
    
    def __mul__(self,other):
        if isinstance(other,Interval):
            interval_hull = [self.lb * other.lb,self.lb*other.ub,self.ub*other.lb,self.ub*other.ub]
            return Interval(min(interval_hull),max(interval_hull))
        elif isinstance(other,int) or isinstance(other,float):
            if other >= 0:
                return Interval(other*self.lb,other*self.ub)
            else:
                return Interval(other*self.ub,other*self.lb)
            
    def __truediv__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            return self.__mul__(1/other)
        if other.lb <= 0 and other.ub >= 0:
            raise ValueError('Division with interval containing 0')
        return self.__mul__(Interval(1/other.ub,1/other.lb))
    
    def __rtruediv__(self,other):
            return Interval(other/self.ub,other/self.lb)
    
    def __ge__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            return self.lb >= other
        return self.lb >= other.ub
    
    def __le__(self,other):
        return self.ub <= other
    
    def __gt__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            return self.lb > other
        return self.lb > other.ub
    
    def __lt__(self,other):
        return self.ub < other
    
    def __pow__(self,other):
        if isinstance(other,int) and other >= 0:
            if other%2==1 or (other%2==0 and (self > 0 or self < 0)):
                return Interval(min(self.lb**other,self.ub**other),max(self.lb**other,self.ub**other))
            else:
                return Interval(0,max(self.lb**other,self.ub**other))
        else:
            raise ValueError('Exponent must be positive integer')
        
    def sqrt(self):
        if not self >= 0:
            raise ValueError('Interval must be nonnegative for sqrt.')
        return Interval(np.sqrt(self.lb),np.sqrt(self.ub))
    
    def contains(self,other):
        return other <= self.ub and other >= self.lb
    
    def abs(self):
        if self.contains(0):
            return Interval(0,max(abs(self.ub),abs(self.lb)))
        else:
            return Interval(min(abs(self.lb),abs(self.ub)),max(abs(self.lb),abs(self.ub)))
        
    def log(self):
        return Interval(np.log(self.lb),np.log(self.ub))
    
    def exp(self):
        return Interval(np.exp(self.lb),np.exp(self.ub))
        
    __rmul__ = __mul__
    __radd__ = __add__

class IntervalVector:
    def __init__(self,box):
        if isinstance(box[0],Interval):
            self.intervals = box
        else:
            self.intervals = [Interval(row[0],row[1]) for row in box]
        self.box_mean = np.array([i.box_mean for i in self.intervals])
        self.len = len(self.intervals)

    def __str__(self):
        output = '(' + self.intervals[0].__str__() + '\n'
        for i in self.intervals[1:]:
            output += f' {i.__str__()}\n'
        return output[:-1] + ')'
    
    def __getitem__(self,key):
        return self.intervals[key]
    
    def __mul__(self,other):
        if isinstance(other,Interval):
            return IntervalVector([other * i for i in self])
        else:
            raise NotImplementedError('Multiplication between Interval-Vectors is not defined.')

    
    def norm(self):
        return sum([i**2 for i in self]).sqrt()
    