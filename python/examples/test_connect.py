import ctypes
import neuralgpu as ngpu

N = 5

neuron = ngpu.Create("aeif_cond_beta", 2*N)
neuron_even = []
neuron_odd = []
for i in range(N):
    neuron_even.append(neuron[2*i])
    neuron_odd.append(neuron[2*i+1])

even_to_odd_delay = []
even_to_odd_weight = []
odd_to_even_delay = []
odd_to_even_weight = []
for itgt in range(N):
    ite = 2*itgt
    ito = 2*itgt + 1
    for isrc in range(N):
        ise = 2*isrc
        iso = 2*isrc + 1
        even_to_odd_delay.append(2.0*N*ise + ito)
        even_to_odd_weight.append(100.0*(2.0*N*ise + ito))
        odd_to_even_delay.append(2.0*N*iso + ite)
        odd_to_even_weight.append(100.0*(2.0*N*iso + ite))


conn_dict={"rule": "all_to_all"}
even_to_odd_syn_dict={
    "weight_array":even_to_odd_weight,
    "delay_array":even_to_odd_delay}
  
odd_to_even_syn_dict={
    "weight_array":odd_to_even_weight,
    "delay_array":odd_to_even_delay}
  
ngpu.Connect(neuron_even, neuron_odd, conn_dict, even_to_odd_syn_dict);
ngpu.Connect(neuron_odd, neuron_even, conn_dict, odd_to_even_syn_dict);

# Even to all
conn_id = ngpu.GetConnections(neuron_even, neuron)
conn_stat_list = ngpu.GetConnectionStatus(conn_id);
print("########################################")
print("Even to all")
for i in range(len(conn_id)): # len(conn_stat_list)
    i_source = conn_stat_list[i].i_source;
    i_target = conn_stat_list[i].i_target;
    i_port = conn_stat_list[i].i_port;
    i_syn = conn_stat_list[i].i_syn;
    weight = conn_stat_list[i].weight;
    delay = conn_stat_list[i].delay;
    print("  i_source ", i_source)
    print("  i_target ", i_target)
    print("  i_port ", i_port)
    print("  i_syn ", i_syn)
    print("  weight ", weight)
    print("  delay ", delay)
    print()

print("########################################")

