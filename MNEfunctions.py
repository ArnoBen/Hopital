#We need to know at which epochs the patient loses consciousness and wakes up
def getLOCROC(epochs, style = 'epoch', ptsPerEpoch = 250): 
    passed_LOC = False
    LOC_epoch, ROC_epoch = [0,0]
    for i in range(epochs.events.shape[0]):
        if epochs.events[i][2] == 2 and passed_LOC == False: 
            LOC_epoch = i; passed_LOC = True;
        if epochs.events[i][2] == 3:
            ROC_epoch = i; break;
    if LOC_epoch == 0: raise ValueError('LOC is at 0.')
    if ROC_epoch == 0: raise ValueError('ROC is at 0.')
    if style == 'epoch': return [LOC_epoch, ROC_epoch]
    if style == 'time' : return [LOC_epoch / ptsPerEpoch, ROC_epoch / ptsPerEpoch]