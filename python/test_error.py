import neuralgpu as ngpu

try:
    print ("ok0")
    neu=ngpu.CreateNeuron("aeif_cond_be", 1, 1)
    print ("ok1")
except:
    print ("wrong")
    
