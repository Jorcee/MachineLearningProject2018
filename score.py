def score(cls_pred, cls_real,mod='F1'):
    # the order of cls_real and cls_pred can be changed
    #print(cls_pred)
    #input()
    #correct pairs
    cps=0
    for i in cls_pred:
        for j in cls_real:
            a=len(set(i)&set(j))
            cps += a*(a-1)/2
    #total pairs
    tps=0
    for i in cls_real:
        a=len(i)
        tps += a*(a-1)/2
    #predicted pairs
    pps=0
    for i in cls_pred:
        a=len(i)
        pps += a*(a-1)/2
    if mod=='F1':
        return 2*cps/(tps+pps)
    if mod=='precise':
        return cps/pps
    if mod=='recall':
        return cps/tps