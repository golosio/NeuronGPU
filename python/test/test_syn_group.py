import sys
import neuralgpu as ngpu

syn_group = ngpu.CreateSynGroup("test_syn_model")
ngpu.SetSynGroupParam(syn_group, "fact", 0.5)
ngpu.SetSynGroupParam(syn_group, "offset", 2.5)
print(ngpu.GetStatus(syn_group))

fact = ngpu.GetSynGroupParam(syn_group, "fact")
print("fact: ", fact)

print(ngpu.GetStatus(syn_group, "fact"))

print(ngpu.GetStatus(syn_group, "offset"))

print(ngpu.GetStatus(syn_group, ["fact", "offset"]))

print(ngpu.GetStatus(syn_group, ["offset", "fact"]))


