""" Python wrapper for the C shared library neuralgpu"""
import sys, platform
import ctypes, ctypes.util

lib_path="/home/golosio/lib/libpyneuralgpu.so"
_neuralgpu=ctypes.CDLL(lib_path)


NeuralGPU_SetRandomSeed = _neuralgpu.NeuralGPU_SetRandomSeed
NeuralGPU_SetRandomSeed.argtypes = (ctypes.c_ulonglong,)
NeuralGPU_SetRandomSeed.restype = ctypes.c_int
def SetRandomSeed(seed):
    "Set seed for random number generation"
    return NeuralGPU_SetRandomSeed(ctypes.c_ulonglong(seed))


NeuralGPU_SetTimeResolution = _neuralgpu.NeuralGPU_SetTimeResolution
NeuralGPU_SetTimeResolution.argtypes = (ctypes.c_float,)
NeuralGPU_SetTimeResolution.restype = ctypes.c_int
def SetTimeResolution(time_res):
    "Set time resolution in ms"
    return NeuralGPU_SetTimeResolution(ctypes.c_float(time_res))


NeuralGPU_GetTimeResolution = _neuralgpu.NeuralGPU_GetTimeResolution
NeuralGPU_GetTimeResolution.restype = ctypes.c_float
def GetTimeResolution():
    "Get time resolution in ms"
    return NeuralGPU_GetTimeResolution()


NeuralGPU_CreateNeuron = _neuralgpu.NeuralGPU_CreateNeuron
NeuralGPU_CreateNeuron.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int)
NeuralGPU_CreateNeuron.restype = ctypes.c_int
def CreateNeuron(model_name, n_neurons, n_receptors):
    "Create a neuron group"
    c_model_name = ctypes.create_string_buffer(str.encode(model_name), len(model_name)+1)
    return NeuralGPU_CreateNeuron(c_model_name, ctypes.c_int(n_neurons), ctypes.c_int(n_receptors)) 


NeuralGPU_CreatePoissonGenerator = _neuralgpu.NeuralGPU_CreatePoissonGenerator
NeuralGPU_CreatePoissonGenerator.argtypes = (ctypes.c_int, ctypes.c_float)
NeuralGPU_CreatePoissonGenerator.restype = ctypes.c_int
def CreatePoissonGenerator(n_nodes, rate):
    "Create a poisson-distributed spike generator"
    return NeuralGPU_CreatePoissonGenerator(ctypes.c_int(n_nodes), ctypes.c_float(rate)) 


NeuralGPU_CreateSpikeGenerator = _neuralgpu.NeuralGPU_CreateSpikeGenerator
NeuralGPU_CreateSpikeGenerator.argtypes = (ctypes.c_int,)
NeuralGPU_CreateSpikeGenerator.restype = ctypes.c_int
def CreateSpikeGenerator(n_nodes):
    "Create a spike generator"
    return NeuralGPU_CreateSpikeGenerator(ctypes.c_int(n_nodes)) 


NeuralGPU_CreateRecord = _neuralgpu.NeuralGPU_CreateRecord
NeuralGPU_CreateRecord.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int); 
NeuralGPU_CreateRecord.restype = ctypes.c_int
def CreateRecord(file_name, var_name_list, i_neuron_list, i_receptor_list):
    "Create a record of neuron variables"
    n_neurons = len(i_neuron_list)
    c_file_name = ctypes.create_string_buffer(str.encode(file_name), len(file_name)+1)    
    array_int_type = ctypes.c_int * n_neurons
    array_char_pt_type = ctypes.POINTER(ctypes.c_char) * n_neurons;
    c_var_name_list=[]
    for i in range(n_neurons):
        c_var_name = ctypes.create_string_buffer(str.encode(var_name_list[i]), len(var_name_list[i])+1)
        c_var_name_list.append(c_var_name);
    return NeuralGPU_CreateRecord(c_file_name, array_char_pt_type(*c_var_name_list), array_int_type(*i_neuron_list),
                                  array_int_type(*i_receptor_list), ctypes.c_int(n_neurons)) 


NeuralGPU_GetRecordDataRows = _neuralgpu.NeuralGPU_GetRecordDataRows
NeuralGPU_GetRecordDataRows.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordDataRows.restype = ctypes.c_int
def GetRecordDataRows(i_record):
    "Get record n. of rows"
    return NeuralGPU_GetRecordDataRows(ctypes.c_int(i_record))


NeuralGPU_GetRecordDataColumns = _neuralgpu.NeuralGPU_GetRecordDataColumns
NeuralGPU_GetRecordDataColumns.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordDataColumns.restype = ctypes.c_int
def GetRecordDataColumns(i_record):
    "Get record n. of columns"
    return NeuralGPU_GetRecordDataColumns(ctypes.c_int(i_record))


NeuralGPU_GetRecordData = _neuralgpu.NeuralGPU_GetRecordData
NeuralGPU_GetRecordData.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordData.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
def GetRecordData(i_record):
    "Get record data"
    data_arr_pt = NeuralGPU_GetRecordData(ctypes.c_int(i_record))
    nr = GetRecordDataRows(i_record)
    nc = GetRecordDataColumns(i_record)
    data_list = []
    for ir in range(nr):
        row_list = []
        for ic in range(nc):
            row_list.append(data_arr_pt[ir][ic])
            
        data_list.append(row_list)
        
    return data_list    


NeuralGPU_SetNeuronParams = _neuralgpu.NeuralGPU_SetNeuronParams
NeuralGPU_SetNeuronParams.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int, ctypes.c_float)
NeuralGPU_SetNeuronParams.restype = ctypes.c_int
def SetNeuronParams(param_name, i_node, n_neurons, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    return NeuralGPU_SetNeuronParams(c_param_name, ctypes.c_int(i_node), ctypes.c_int(n_neurons), ctypes.c_float(val)) 


NeuralGPU_SetNeuronVectParams = _neuralgpu.NeuralGPU_SetNeuronVectParams
NeuralGPU_SetNeuronVectParams.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int)
NeuralGPU_SetNeuronVectParams.restype = ctypes.c_int
def SetNeuronVectParams(param_name, i_node, n_neurons, params_list):
    "Set neuron vector parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    vect_size = len(params)
    array_float_type = ctypes.c_float * vect_size
    return NeuralGPU_SetNeuronVectParams(c_param_name, ctypes.c_int(i_node), ctypes.c_int(n_neurons),
                                         array_float_type(*params_list), ctypes.c_int(vect_size))  

NeuralGPU_SetSpikeGenerator = _neuralgpu.NeuralGPU_SetSpikeGenerator
NeuralGPU_SetSpikeGenerator.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float))
NeuralGPU_SetSpikeGenerator.restype = ctypes.c_int
def SetSpikeGenerator(i_node, spike_time_list, spike_height_list):
    "Set spike generator spike times and heights"
    n_spikes = len(spike_time_list)
    array_float_type = ctypes.c_float * n_spikes
    return NeuralGPU_SetSpikeGenerator(ctypes.c_int(i_node), ctypes.c_int(n_spikes),
                                       array_float_type(*spike_time_list), array_float_type(*spike_height_list)) 


NeuralGPU_Calibrate = _neuralgpu.NeuralGPU_Calibrate
NeuralGPU_Calibrate.restype = ctypes.c_int
def Calibrate():
    "Calibrate simulation"
    return NeuralGPU_Calibrate()


NeuralGPU_Simulate = _neuralgpu.NeuralGPU_Simulate
NeuralGPU_Simulate.restype = ctypes.c_int
def Simulate():
    "Simulate neural activity"
    return NeuralGPU_Simulate()


NeuralGPU_ConnectMpiInit = _neuralgpu.NeuralGPU_ConnectMpiInit
NeuralGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
NeuralGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = ctypes.POINTER(ctypes.c_char) * argc;
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(str.encode(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    return NeuralGPU_ConnectMpiInit(ctypes.c_int(argc), array_char_pt_type(*c_var_name_list))


NeuralGPU_MpiId = _neuralgpu.NeuralGPU_MpiId
NeuralGPU_MpiId.restype = ctypes.c_int
def MpiId():
    "Get MPI Id"
    return NeuralGPU_MpiId()


NeuralGPU_MpiNp = _neuralgpu.NeuralGPU_MpiNp
NeuralGPU_MpiNp.restype = ctypes.c_int
def MpiNp():
    "Get MPI Np"
    return NeuralGPU_MpiNp()


NeuralGPU_ProcMaster = _neuralgpu.NeuralGPU_ProcMaster
NeuralGPU_ProcMaster.restype = ctypes.c_int
def ProcMaster():
    "Get MPI ProcMaster"
    return NeuralGPU_ProcMaster()


NeuralGPU_MpiFinalize = _neuralgpu.NeuralGPU_MpiFinalize
NeuralGPU_MpiFinalize.restype = ctypes.c_int
def MpiFinalize():
    "Finalize MPI"
    return NeuralGPU_MpiFinalize()


NeuralGPU_RandomInt = _neuralgpu.NeuralGPU_RandomInt
NeuralGPU_RandomInt.argtypes = (ctypes.c_size_t,)
NeuralGPU_RandomInt.restype = ctypes.POINTER(ctypes.c_uint)
def RandomInt(n):
    "Generate n random integers in CUDA memory"
    return NeuralGPU_RandomInt(ctypes.c_size_t(n))


NeuralGPU_RandomUniform = _neuralgpu.NeuralGPU_RandomUniform
NeuralGPU_RandomUniform.argtypes = (ctypes.c_size_t,)
NeuralGPU_RandomUniform.restype = ctypes.POINTER(ctypes.c_float)
def RandomUniform(n):
    "Generate n random floats with uniform distribution in (0,1) in CUDA memory"
    return NeuralGPU_RandomUniform(ctypes.c_size_t(n))


NeuralGPU_RandomNormal = _neuralgpu.NeuralGPU_RandomNormal
NeuralGPU_RandomNormal.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float)
NeuralGPU_RandomNormal.restype = ctypes.POINTER(ctypes.c_float)
def RandomNormal(n, mean, stddev):
    "Generate n random floats with normal distribution in CUDA memory"
    return NeuralGPU_RandomNormal(ctypes.c_size_t(n), ctypes.c_float(mean), ctypes.c_float(stddev))


NeuralGPU_RandomNormalClipped = _neuralgpu.NeuralGPU_RandomNormalClipped
NeuralGPU_RandomNormalClipped.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                          ctypes.c_float)
NeuralGPU_RandomNormalClipped.restype = ctypes.POINTER(ctypes.c_float)
def RandomNormalClipped(n, mean, stddev, vmin, vmax):
    "Generate n random floats with normal clipped distribution in CUDA memory"
    return NeuralGPU_RandomNormalClipped(ctypes.c_size_t(n), ctypes.c_float(mean), ctypes.c_float(stddev),
                                         ctypes.c_float(vmin), ctypes.c_float(vmax))


NeuralGPU_Connect = _neuralgpu.NeuralGPU_Connect
NeuralGPU_Connect.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_float, ctypes.c_float)
NeuralGPU_Connect.restype = ctypes.c_int
def Connect(i_source_neuron, i_target_neuron, i_port, weight, delay):
    "Connect two neurons"
    return NeuralGPU_Connect(ctypes.c_int(i_source_neuron), ctypes.c_int(i_target_neuron), ctypes.c_ubyte(i_port), ctypes.c_float(weight), ctypes.c_float(delay))



NeuralGPU_ConnectMpiInit = _neuralgpu.NeuralGPU_ConnectMpiInit
NeuralGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
NeuralGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = ctypes.POINTER(ctypes.c_char) * argc;
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(str.encode(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    return NeuralGPU_ConnectMpiInit(ctypes.c_int(argc), array_char_pt_type(*c_var_name_list))

