import numpy as np

def twolist_sorting(x,y,alpha):
    if len(x)!=len(y):
        print("Error: two lists for twolist-sorting are not of same length:",len(x),len(y))
        exit()
    x1=x/np.sum(x)
    y1=y/np.sum(y)
    s1=alpha*x1+(1-alpha)*y1
    indexes=np.argsort(s1)
    return indexes

def n_list_sorting(arrays,weights): #QUIQUIURG maybe normalization is not sufficient: maybe a point that is important for list A, due to normalization is penalized w.r.t. a point for list B
    s1=np.zeros(len(arrays[0]))
    ref_len=len(arrays[0])
    for i_a,arr in enumerate(arrays):
        if len(arr)!=ref_len:
            print("Error: two lists for nlist-sorting are not of same length: 0,",i_a)
            exit()
        arrsum=np.sum(arr)
        if arrsum!=0.0:
            norm_arr=len(arrays[0])*arr/np.sum(arr) #QUIQUIURG is this really allowing for a comparison of lists with different maximums?
        else:
            if (arr==0.0).all():
                norm_arr=arr 
            else:
                norm_arr=arr #QUIQUIURG probably this is not ok in a situation where not everything is strictly 0
        #print(f"{weights[i_a]=} {norm_arr=}")
        s1+=weights[i_a]*norm_arr
        print(f"{weights[i_a]=} {arrsum=} {(arr==0.0).all()=}")
    indexes=np.argsort(s1)
    return indexes,s1

"""
def diversity_scores(array,npoints): #is npoints this identical to batch_size?
    scores=np.zeros(len(array),dtype=int)
    #count=0
    linspace=np.linspace(np.min(array),np.max(array),npoints) #QUIQUIURG min and max may vary depending on the AL cycle, and: they are diversity not also novelty.
    for lin in linspace:
        #count+=1
        ind=np.argmin(np.abs(array-lin))
        #scores[ind]=count
        scores[ind]=1
    indexes=np.sort(np.where(scores>0))
    return scores, indexes
"""
def diversity_scores(array,npoints): #is npoints this identical to batch_size?
    scores=np.zeros(len(array)) #,dtype=int)
    #count=0
    linspace=np.linspace(np.min(array),np.max(array),npoints) #QUIQUIURG min and max may vary depending on the AL cycle, and: they are diversity not also novelty.
    counts=np.zeros(npoints+1) #,dtype=int)
    for i_a,val in enumerate(array):
        #count+=1
        ind=np.argmin(np.abs(val-linspace))
        #scores[ind]=count
        scores[ind]=1
        #counts[ind+1]+=0.001
        #scores[i_a]=ind+1+counts[ind+1] #QUIQUIURG perform a clustering instead?
    indexes=np.sort(np.where(scores>0))
    #indexes=np.argsort(scores)
    return scores, indexes
    
if __name__=='__main__':

    a = np.array([8,        20,     7,    1,    3,  11,  14])
    b = np.array([321.3, 121.5, 123.4, 10.9,  7.8, 5.6, 4.3])
    c = np.array([  0.1,   0.0,   0.3,  0.2, 0.15, 0.4, 0.6])

    print("Two list:")

    #for alpha in [0.5]:
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        indexes=twolist_sorting(a,b,alpha)
        #for idx in indexes:
        #    print(idx,":",x[idx],y[idx])
        print(indexes)

    print("\nN list:")
    print(f"{np.argsort(a)=}")
    print(f"{np.argsort(b)=}")
    print(f"{np.argsort(c)=}")

    for weights in [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.], [1.,1.,0.], [1.,1.,1.]]:
        indexes,s1=n_list_sorting([a,b,c],weights)
        print(weights,indexes)

    print("\nLinspace test:")

    #ys=np.array(           [0.0, 0.1, 0.3, 0.35, 0.7, 0.8, 0.82])
    #ys=np.array(           [0.0, 0.8, 0.82, 0.35, 0.7, 0.1, 0.3])
    ys=np.array(           [0.0, 0.8, 0.82, 0.35, 0.7, 0.1, 0.3, 0.12, 0.19, 0.9])
    ##desirability0=np.array([1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 1.0 ])
    ##desirability0=np.array([1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 1.0 ])

    """
    scores=np.zeros(len(ys),dtype=int)
    npoints=5 #is this identical to batch_size?
    count=0
    #linspace=np.linspace(0,len(ys),npoints-1)
    linspace=np.linspace(np.min(ys),np.max(ys),npoints)
    print(linspace)
    for lin in linspace:
        count+=1
        ind=np.argmin(np.abs(ys-lin))
        scores[ind]=count
    
    """

    npoints=4 #is this identical to batch_size?
    scores, indexes = diversity_scores(ys, npoints)
    print(f"{ys=}")
    print(f"{scores=}")
    print(f"{indexes=} {ys[indexes]=}")