import numpy as np


def test1(x,w):
    print('test1 is running')
    return [1,0,1,0]


def test0(x,w):
    print('test0 is running')
    return [0 for t in x]


class AdaBoost(object):
    def __init__(self,cls=[test1,test0]):
        self.M=len(cls)
        self.c=np.zeros(self.M)
        self.w=np.array([])
        self.n=0
        self.f=cls
        pass

    def fit(self,x,y):
        self.w=np.ones(len(y),dtype=float)
        self.w=1/len(y)*self.w
        for m in range(self.M):
            pre = self.f[m](x,self.w)
            e=0
            # print(self.w)
            # print(pre)
            for i in range(len(pre)):
                if pre[i]==y[i]:
                    e=e+1
            e=e/len(pre)
            # print(e)
            self.c[m]=np.log((1-e)/e)
            print(self.c[m])
            for i in range(len(pre)):
                if pre[i]!=y[i]:
                    self.w[i]=self.w[i]*((1-e)/e)
            # print("m=",m)
            # print(self.w)
            self.w=self.w/self.w.sum()
            # print(self.w)

    def predict(self, x):
        s = np.array(np.zeros(len(x)))
        for m in range(len(self.c)):
            s = s + self.c[m] * np.array(self.f[m](x,self.w))
        return [0 if t < 0 else 1 for t in s]

if __name__ == '__main__':
    x=[0,1,0,0]
    y=[1,0,1,0]
    a=AdaBoost()
    a.fit(x,y)
    print(a.predict(x))