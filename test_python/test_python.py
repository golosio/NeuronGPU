import neuralgpu

neuralgpu.ConnectMpiInit()

neur = neuralgpu.CreateNeuron("AEIF", 10, 1)
poiss_gen = neuralgpu.CreatePoissonGenerator(10, 12000.0);

neuralgpu.SetNeuronParams("I_e", 1, 9, 1000.0)

neuralgpu.Connect(poiss_gen, neur, 0, 0.05, 2.0)
record = neuralgpu.CreateRecord("test.txt", ["V_m","V_m","V_m"], [0, 1, 2],
                       [0, 0, 0])

neuralgpu.Simulate()

nrows=neuralgpu.GetRecordDataRows(record)
ncol=neuralgpu.GetRecordDataColumns(record)

print nrows, ncol

data_list = neuralgpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]

import pylab

pylab.figure(1)
pylab.plot(t, V1)

pylab.figure(2)
pylab.plot(t, V2)

pylab.show()
