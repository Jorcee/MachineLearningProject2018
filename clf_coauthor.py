import numpy
import re
from graph import graph
class clf_coauthor:

    def preprocess(self,name):
        return re.sub("[^A-Za-z]","",name).lower()

    def __init__(self):
        pass

    def train(self,data,label):
        print('clf_org train start')
        pass
        print('clf_org train end')

    def predict(self,paper1,paper2):
        namelist1=[i['name'] for i in paper1['authors']]
        namelist2=[i['name'] for i in paper2['authors']]
        if len(set(namelist1)&set(namelist2))>1:
            return 1
        else:
            return 0

    def matrix(self,data):
        n=len(data)
        authors_of_papers=[]
        authors_total={}
        for paper in data:
            authors_one_paper=[]
            for author in paper['authors']:
                name=self.preprocess(author['name'])
                authors_one_paper.append(name)
                if name in authors_total:
                    authors_total[name]+=1
                else:
                    authors_total[name]=1
            authors_of_papers.append(authors_one_paper)
        # find who is the author to be classified
        name_want=""
        m=0
        for name,times in authors_total.items():
            if(times>m):
                name_want=name
                m=times

        del authors_total[name_want]

        # add serial number
        index=0
        for name in authors_total:
            authors_total[name]=index+n
            index+=1

        #creat link matrix
        link_mat = numpy.zeros(shape=(n+len(authors_total),n+len(authors_total)))

        for i in range(n):
            indexs=[i]
            for author in authors_of_papers[i]:
                if (author != name_want):
                    indexs.append(authors_total[author])
            
            #print(indexs)
            for j in indexs:
                for k in indexs:
                    if(j!=k):
                        link_mat[j,k]=1

        # calculate the mini-distance matrix

        graph1=graph(link_mat)
        dis=graph1.distance(list(range(n)),list(range(n)))

        correlation=numpy.zeros(shape=(n,n))
        correlation[numpy.where(dis>0)]=1/numpy.sqrt(dis[numpy.where(dis>0)])
        # correlation matrix is a symmetric matrix
        return correlation