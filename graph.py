import numpy

class graph:
    def __init__(self,link):
        # self.N=[]
        # self.n=link.shape[0]
        # for i in range(self.n):
        #     Ni=[]
        #     for j in range(self.n):
        #         if(link[i,j]>0):
        #             Ni.append(j)
        #     self.N.append(Ni)
        self.N=link
        self.n=len(link)

    def isdiscovered(k):
        return self.discovered[k]

    def getneighbors(self,list0,max_dis=10):
        discovered=numpy.zeros(self.n)
        distance=1
        list_now=list0
        if(type(list_now)==int):
            list_now=[list_now]
        while(1):
            neighbor_i=[]
            for i in list_now:
                for j in self.N[i]:
                    if(discovered[j]==0):
                        neighbor_i.append(j)
                        discovered[j]=distance
            distance+=1
            list_now=neighbor_i
            if (len(neighbor_i)==0) or distance>max_dis:
                return discovered
        
    def distance(self,list1,list2):
        dis=numpy.zeros(shape=(len(list1),len(list2)))
        if(type(list1)==int):
            list1=[list1]
        ix=0
        for i in list1:
            
            #print(ix)
            a=self.getneighbors(i,5)

            item=numpy.zeros(shape=(1,len(list2)))
            iy=0
            for j in list2:
                # if(a[j]==2):
                #     print(a[j])
                #dis[ix,iy]=max(a[j]-1,0)
                item[0,iy]=max(a[j]-1,0)
                iy+=1
            dis[ix,:]=item
            ix+=1
        return dis
