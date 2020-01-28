import neuralgpu as ngpu

neuron = ngpu.Create("aeif_cond_beta", 40)
neuron1 = neuron[0:19]
neuron2 = neuron[20:39]

conn_dict={"rule": "fixed_indegree", "indegree": 20}
syn_dict={"weight": 1.0, "delay":1.0, "receptor":0}

ngpu.Connect(neuron1, neuron2[0:0], conn_dict, syn_dict)

conn_id_list = ngpu.GetConnections(target=neuron2[0])

for conn_id in conn_id_list:
    print(ngpu.GetStatus(conn_id))
