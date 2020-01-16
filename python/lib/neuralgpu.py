""" Python wrapper for the shared library neuralgpu"""
import sys, platform
import ctypes, ctypes.util

lib_path="/home/golosio/lib/libneuralgpu_C.so"
_neuralgpu=ctypes.CDLL(lib_path)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_char_p = ctypes.POINTER(ctypes.c_char)
c_void_p = ctypes.c_void_p

class NodeSeq(object):
    def __init__(self, i0, n):
        self.i0 = i0
        self.n = n

    def Subseq(self, first, last):
        if first<0 | last<first:
            raise ValueError("Sequence subset range error")
        if last>=self.n:
            raise ValueError("Sequence subset out of range")
        return NodeSeq(self.i0 + first, last - first + 1)
    def __getitem__(self, i):
        if type(i)==slice:
            if i.step != None:
                raise ValueError("Subsequence cannot have a step")
            return self.Subseq(i.start, i.stop)
 
        if i<0:
            raise ValueError("Sequence index cannot be negative")
        if i>=self.n:
            raise ValueError("Sequence index out of range")
        return self.i0 + i
    def ToList(self):
        return list(range(self.i0, self.i0 + self.n))

conn_rule_name = ("one_to_one", "all_to_all", "fixed_total_number",
                  "fixed_indegree", "fixed_outdegree")
    
NeuralGPU_GetErrorMessage = _neuralgpu.NeuralGPU_GetErrorMessage
NeuralGPU_GetErrorMessage.restype = ctypes.POINTER(ctypes.c_char)
def GetErrorMessage():
    "Get error message from NeuralGPU exception"
    message = ctypes.cast(NeuralGPU_GetErrorMessage(), ctypes.c_char_p).value
    return message
 
NeuralGPU_GetErrorCode = _neuralgpu.NeuralGPU_GetErrorCode
NeuralGPU_GetErrorCode.restype = ctypes.c_ubyte
def GetErrorCode():
    "Get error code from NeuralGPU exception"
    return NeuralGPU_GetErrorCode()
 
NeuralGPU_SetOnException = _neuralgpu.NeuralGPU_SetOnException
NeuralGPU_SetOnException.argtypes = (ctypes.c_int,)
def SetOnException(on_exception):
    "Define whether handle exceptions (1) or exit (0) in case of errors"
    return NeuralGPU_SetOnException(ctypes.c_int(on_exception))

SetOnException(1)

NeuralGPU_SetRandomSeed = _neuralgpu.NeuralGPU_SetRandomSeed
NeuralGPU_SetRandomSeed.argtypes = (ctypes.c_ulonglong,)
NeuralGPU_SetRandomSeed.restype = ctypes.c_int
def SetRandomSeed(seed):
    "Set seed for random number generation"
    ret = NeuralGPU_SetRandomSeed(ctypes.c_ulonglong(seed))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetTimeResolution = _neuralgpu.NeuralGPU_SetTimeResolution
NeuralGPU_SetTimeResolution.argtypes = (ctypes.c_float,)
NeuralGPU_SetTimeResolution.restype = ctypes.c_int
def SetTimeResolution(time_res):
    "Set time resolution in ms"
    ret = NeuralGPU_SetTimeResolution(ctypes.c_float(time_res))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetTimeResolution = _neuralgpu.NeuralGPU_GetTimeResolution
NeuralGPU_GetTimeResolution.restype = ctypes.c_float
def GetTimeResolution():
    "Get time resolution in ms"
    ret = NeuralGPU_GetTimeResolution()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetMaxSpikeBufferSize = _neuralgpu.NeuralGPU_SetMaxSpikeBufferSize
NeuralGPU_SetMaxSpikeBufferSize.argtypes = (ctypes.c_int,)
NeuralGPU_SetMaxSpikeBufferSize.restype = ctypes.c_int
def SetMaxSpikeBufferSize(max_size):
    "Set maximum size of spike buffer per node"
    ret = NeuralGPU_SetMaxSpikeBufferSize(ctypes.c_int(max_size))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetMaxSpikeBufferSize = _neuralgpu.NeuralGPU_GetMaxSpikeBufferSize
NeuralGPU_GetMaxSpikeBufferSize.restype = ctypes.c_int
def GetMaxSpikeBufferSize():
    "Get maximum size of spike buffer per node"
    ret = NeuralGPU_GetMaxSpikeBufferSize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_CreateNeuron = _neuralgpu.NeuralGPU_CreateNeuron
NeuralGPU_CreateNeuron.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int)
NeuralGPU_CreateNeuron.restype = ctypes.c_int
def CreateNeuron(model_name, n_nodes, n_ports):
    "Create a neuron group"
    c_model_name = ctypes.create_string_buffer(str.encode(model_name), len(model_name)+1)
    i_node =NeuralGPU_CreateNeuron(c_model_name, ctypes.c_int(n_nodes), ctypes.c_int(n_ports))
    ret = NodeSeq(i_node, n_nodes)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_CreatePoissonGenerator = _neuralgpu.NeuralGPU_CreatePoissonGenerator
NeuralGPU_CreatePoissonGenerator.argtypes = (ctypes.c_int, ctypes.c_float)
NeuralGPU_CreatePoissonGenerator.restype = ctypes.c_int
def CreatePoissonGenerator(n_nodes, rate):
    "Create a poisson-distributed spike generator"
    i_node = NeuralGPU_CreatePoissonGenerator(ctypes.c_int(n_nodes), ctypes.c_float(rate)) 
    ret = NodeSeq(i_node, n_nodes)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_CreateSpikeGenerator = _neuralgpu.NeuralGPU_CreateSpikeGenerator
NeuralGPU_CreateSpikeGenerator.argtypes = (ctypes.c_int,)
NeuralGPU_CreateSpikeGenerator.restype = ctypes.c_int
def CreateSpikeGenerator(n_nodes):
    "Create a spike generator"
    i_node = NeuralGPU_CreateSpikeGenerator(ctypes.c_int(n_nodes)) 
    ret = NodeSeq(i_node, n_nodes)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_CreateRecord = _neuralgpu.NeuralGPU_CreateRecord
NeuralGPU_CreateRecord.argtypes = (c_char_p, ctypes.POINTER(c_char_p), c_int_p, c_int_p, ctypes.c_int)
NeuralGPU_CreateRecord.restype = ctypes.c_int
def CreateRecord(file_name, var_name_list, i_node_list, i_port_list):
    "Create a record of neuron variables"
    n_nodes = len(i_node_list)
    c_file_name = ctypes.create_string_buffer(str.encode(file_name), len(file_name)+1)    
    array_int_type = ctypes.c_int * n_nodes
    array_char_pt_type = c_char_p * n_nodes
    c_var_name_list=[]
    for i in range(n_nodes):
        c_var_name = ctypes.create_string_buffer(str.encode(var_name_list[i]), len(var_name_list[i])+1)
        c_var_name_list.append(c_var_name)

    ret = NeuralGPU_CreateRecord(c_file_name,
                                 array_char_pt_type(*c_var_name_list),
                                 array_int_type(*i_node_list),
                                 array_int_type(*i_port_list),
                                 ctypes.c_int(n_nodes))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetRecordDataRows = _neuralgpu.NeuralGPU_GetRecordDataRows
NeuralGPU_GetRecordDataRows.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordDataRows.restype = ctypes.c_int
def GetRecordDataRows(i_record):
    "Get record n. of rows"
    ret = NeuralGPU_GetRecordDataRows(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetRecordDataColumns = _neuralgpu.NeuralGPU_GetRecordDataColumns
NeuralGPU_GetRecordDataColumns.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordDataColumns.restype = ctypes.c_int
def GetRecordDataColumns(i_record):
    "Get record n. of columns"
    ret = NeuralGPU_GetRecordDataColumns(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetRecordData = _neuralgpu.NeuralGPU_GetRecordData
NeuralGPU_GetRecordData.argtypes = (ctypes.c_int,)
NeuralGPU_GetRecordData.restype = ctypes.POINTER(c_float_p)
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
        
    ret = data_list    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronScalParam = _neuralgpu.NeuralGPU_SetNeuronScalParam
NeuralGPU_SetNeuronScalParam.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float)
NeuralGPU_SetNeuronScalParam.restype = ctypes.c_int
def SetNeuronScalParam(param_name, i_node, n_nodes, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetNeuronScalParam(c_param_name, ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes),
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronVectParam = _neuralgpu.NeuralGPU_SetNeuronVectParam
NeuralGPU_SetNeuronVectParam.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int,
                                          c_float_p, ctypes.c_int)
NeuralGPU_SetNeuronVectParam.restype = ctypes.c_int
def SetNeuronVectParam(param_name, i_node, n_nodes, params_list):
    "Set neuron vector parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    vect_size = len(params_list)
    array_float_type = ctypes.c_float * vect_size
    ret = NeuralGPU_SetNeuronVectParam(c_param_name, ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes),
                                       array_float_type(*params_list),
                                       ctypes.c_int(vect_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtScalParam = _neuralgpu.NeuralGPU_SetNeuronPtScalParam
NeuralGPU_SetNeuronPtScalParam.argtypes = (c_char_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_float)
NeuralGPU_SetNeuronPtScalParam.restype = ctypes.c_int
def SetNeuronPtScalParam(param_name, node_pt, n_nodes, val):
    "Set neuron list scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetNeuronPtScalParam(c_param_name, c_void_p(node_pt),
                                          ctypes.c_int(n_nodes),
                                          ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtVectParam = _neuralgpu.NeuralGPU_SetNeuronPtVectParam
NeuralGPU_SetNeuronPtVectParam.argtypes = (c_char_p, ctypes.c_void_p,
                                           ctypes.c_int, c_float_p,
                                           ctypes.c_int)
NeuralGPU_SetNeuronPtVectParam.restype = ctypes.c_int
def SetNeuronPtVectParam(param_name, node_pt, n_nodes, params_list):
    "Set neuron list vector parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    vect_size = len(params_list)
    array_float_type = ctypes.c_float * vect_size
    ret = NeuralGPU_SetNeuronPtVectParam(c_param_name,
                                          ctypes.c_void_p(node_pt),
                                          ctypes.c_int(n_nodes),
                                          array_float_type(*params_list),
                                          ctypes.c_int(vect_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronScalParam = _neuralgpu.NeuralGPU_IsNeuronScalParam
NeuralGPU_IsNeuronScalParam.argtypes = (c_char_p, ctypes.c_int)
NeuralGPU_IsNeuronScalParam.restype = ctypes.c_int
def IsNeuronScalParam(param_name, i_node):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    ret = (NeuralGPU_IsNeuronScalParam(c_param_name, ctypes.c_int(i_node))!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronVectParam = _neuralgpu.NeuralGPU_IsNeuronVectParam
NeuralGPU_IsNeuronVectParam.argtypes = (c_char_p, ctypes.c_int)
NeuralGPU_IsNeuronVectParam.restype = ctypes.c_int
def IsNeuronVectParam(param_name, i_node):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_IsNeuronVectParam(c_param_name, ctypes.c_int(i_node))!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def SetNeuronParam(param_name, nodes, val):
    "Set neuron group scalar or vector parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)

    if type(nodes)==NodeSeq:
        if IsNeuronScalParam(param_name, nodes.i0):
            SetNeuronScalParam(param_name, nodes.i0, nodes.n, val)
        elif IsNeuronVectParam(param_name, nodes.i0):
            SetNeuronVectParam(param_name, nodes.i0, nodes.n, val)
        else:
            raise ValueError("Unknown neuron parameter")
    else:
        node_arr = (ctypes.c_int * len(nodes))(*nodes) 
        node_arr_pt = ctypes.cast(node_arr, ctypes.c_void_p)    

        if IsNeuronScalParam(param_name, nodes[0]):
            NeuralGPU_SetNeuronPtScalParam(c_param_name, node_arr_pt,
                                           len(nodes), val)
        elif IsNeuronVectParam(param_name, nodes[0]):
            vect_size = len(val)
            array_float_type = ctypes.c_float * vect_size
            NeuralGPU_SetNeuronPtVectParam(c_param_name,
                                           node_arr_pt,
                                           len(nodes),
                                           array_float_type(*val),
                                           ctypes.c_int(vect_size))
        else:
            raise ValueError("Unknown neuron parameter")

    
NeuralGPU_SetSpikeGenerator = _neuralgpu.NeuralGPU_SetSpikeGenerator
NeuralGPU_SetSpikeGenerator.argtypes = (ctypes.c_int, ctypes.c_int, c_float_p,
                                        c_float_p)
NeuralGPU_SetSpikeGenerator.restype = ctypes.c_int
def SetSpikeGenerator(i_node, spike_time_list, spike_height_list):
    "Set spike generator spike times and heights"
    n_spikes = len(spike_time_list)
    array_float_type = ctypes.c_float * n_spikes
    ret = NeuralGPU_SetSpikeGenerator(ctypes.c_int(i_node),
                                      ctypes.c_int(n_spikes),
                                      array_float_type(*spike_time_list),
                                      array_float_type(*spike_height_list)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_Calibrate = _neuralgpu.NeuralGPU_Calibrate
NeuralGPU_Calibrate.restype = ctypes.c_int
def Calibrate():
    "Calibrate simulation"
    ret = NeuralGPU_Calibrate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_Simulate = _neuralgpu.NeuralGPU_Simulate
NeuralGPU_Simulate.restype = ctypes.c_int
def Simulate():
    "Simulate neural activity"
    ret = NeuralGPU_Simulate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ConnectMpiInit = _neuralgpu.NeuralGPU_ConnectMpiInit
NeuralGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(c_char_p))
NeuralGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = c_char_p * argc
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(str.encode(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    ret = NeuralGPU_ConnectMpiInit(ctypes.c_int(argc),
                                   array_char_pt_type(*c_var_name_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_MpiId = _neuralgpu.NeuralGPU_MpiId
NeuralGPU_MpiId.restype = ctypes.c_int
def MpiId():
    "Get MPI Id"
    ret = NeuralGPU_MpiId()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_MpiNp = _neuralgpu.NeuralGPU_MpiNp
NeuralGPU_MpiNp.restype = ctypes.c_int
def MpiNp():
    "Get MPI Np"
    ret = NeuralGPU_MpiNp()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ProcMaster = _neuralgpu.NeuralGPU_ProcMaster
NeuralGPU_ProcMaster.restype = ctypes.c_int
def ProcMaster():
    "Get MPI ProcMaster"
    ret = NeuralGPU_ProcMaster()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_MpiFinalize = _neuralgpu.NeuralGPU_MpiFinalize
NeuralGPU_MpiFinalize.restype = ctypes.c_int
def MpiFinalize():
    "Finalize MPI"
    ret = NeuralGPU_MpiFinalize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RandomInt = _neuralgpu.NeuralGPU_RandomInt
NeuralGPU_RandomInt.argtypes = (ctypes.c_size_t,)
NeuralGPU_RandomInt.restype = ctypes.POINTER(ctypes.c_uint)
def RandomInt(n):
    "Generate n random integers in CUDA memory"
    ret = NeuralGPU_RandomInt(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RandomUniform = _neuralgpu.NeuralGPU_RandomUniform
NeuralGPU_RandomUniform.argtypes = (ctypes.c_size_t,)
NeuralGPU_RandomUniform.restype = c_float_p
def RandomUniform(n):
    "Generate n random floats with uniform distribution in (0,1) in CUDA memory"
    ret = NeuralGPU_RandomUniform(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RandomNormal = _neuralgpu.NeuralGPU_RandomNormal
NeuralGPU_RandomNormal.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float)
NeuralGPU_RandomNormal.restype = c_float_p
def RandomNormal(n, mean, stddev):
    "Generate n random floats with normal distribution in CUDA memory"
    ret = NeuralGPU_RandomNormal(ctypes.c_size_t(n), ctypes.c_float(mean),
                                 ctypes.c_float(stddev))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RandomNormalClipped = _neuralgpu.NeuralGPU_RandomNormalClipped
NeuralGPU_RandomNormalClipped.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                          ctypes.c_float)
NeuralGPU_RandomNormalClipped.restype = c_float_p
def RandomNormalClipped(n, mean, stddev, vmin, vmax):
    "Generate n random floats with normal clipped distribution in CUDA memory"
    ret = NeuralGPU_RandomNormalClipped(ctypes.c_size_t(n),
                                        ctypes.c_float(mean),
                                        ctypes.c_float(stddev),
                                        ctypes.c_float(vmin),
                                        ctypes.c_float(vmax))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ConnectMpiInit = _neuralgpu.NeuralGPU_ConnectMpiInit
NeuralGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(c_char_p))
NeuralGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = c_char_p * argc
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(str.encode(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    ret = NeuralGPU_ConnectMpiInit(ctypes.c_int(argc),
                                   array_char_pt_type(*c_var_name_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_Connect = _neuralgpu.NeuralGPU_Connect
NeuralGPU_Connect.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_float, ctypes.c_float)
NeuralGPU_Connect.restype = ctypes.c_int
def SingleConnect(i_source_node, i_target_node, i_port, weight, delay):
    "Connect two nodes"
    ret = NeuralGPU_Connect(ctypes.c_int(i_source_node),
                            ctypes.c_int(i_target_node),
                            ctypes.c_ubyte(i_port), ctypes.c_float(weight),
                            ctypes.c_float(delay))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ConnSpecInit = _neuralgpu.NeuralGPU_ConnSpecInit
NeuralGPU_ConnSpecInit.restype = ctypes.c_int
def ConnSpecInit():
    "Initialize connection rules specification"
    ret = NeuralGPU_ConnSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetConnSpecParam = _neuralgpu.NeuralGPU_SetConnSpecParam
NeuralGPU_SetConnSpecParam.argtypes = (c_char_p, ctypes.c_int)
NeuralGPU_SetConnSpecParam.restype = ctypes.c_int
def SetConnSpecParam(param_name, val):
    "Set connection parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetConnSpecParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ConnSpecIsParam = _neuralgpu.NeuralGPU_ConnSpecIsParam
NeuralGPU_ConnSpecIsParam.argtypes = (c_char_p,)
NeuralGPU_ConnSpecIsParam.restype = ctypes.c_int
def ConnSpecIsParam(param_name):
    "Check name of connection parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_ConnSpecIsParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SynSpecInit = _neuralgpu.NeuralGPU_SynSpecInit
NeuralGPU_SynSpecInit.restype = ctypes.c_int
def SynSpecInit():
    "Initializa synapse specification"
    ret = NeuralGPU_SynSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_SetSynSpecIntParam = _neuralgpu.NeuralGPU_SetSynSpecIntParam
NeuralGPU_SetSynSpecIntParam.argtypes = (c_char_p, ctypes.c_int)
NeuralGPU_SetSynSpecIntParam.restype = ctypes.c_int
def SetSynSpecIntParam(param_name, val):
    "Set synapse int parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetSynSpecIntParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_SetSynSpecFloatParam = _neuralgpu.NeuralGPU_SetSynSpecFloatParam
NeuralGPU_SetSynSpecFloatParam.argtypes = (c_char_p, ctypes.c_float)
NeuralGPU_SetSynSpecFloatParam.restype = ctypes.c_int
def SetSynSpecFloatParam(param_name, val):
    "Set synapse float parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetSynSpecFloatParam(c_param_name, ctypes.c_float(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_SetSynSpecFloatPtParam = _neuralgpu.NeuralGPU_SetSynSpecFloatPtParam
NeuralGPU_SetSynSpecFloatPtParam.argtypes = (c_char_p, ctypes.c_void_p)
NeuralGPU_SetSynSpecFloatPtParam.restype = ctypes.c_int
def SetSynSpecFloatPtParam(param_name, arr):
    "Set synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    if type(arr) is list:
        arr = (ctypes.c_float * len(arr))(*arr) 
    arr_pt = ctypes.cast(arr, ctypes.c_void_p)    
    ret = NeuralGPU_SetSynSpecFloatPtParam(c_param_name, arr_pt)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SynSpecIsIntParam = _neuralgpu.NeuralGPU_SynSpecIsIntParam
NeuralGPU_SynSpecIsIntParam.argtypes = (c_char_p,)
NeuralGPU_SynSpecIsIntParam.restype = ctypes.c_int
def SynSpecIsIntParam(param_name):
    "Check name of synapse int parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_SynSpecIsIntParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SynSpecIsFloatParam = _neuralgpu.NeuralGPU_SynSpecIsFloatParam
NeuralGPU_SynSpecIsFloatParam.argtypes = (c_char_p,)
NeuralGPU_SynSpecIsFloatParam.restype = ctypes.c_int
def SynSpecIsFloatParam(param_name):
    "Check name of synapse float parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_SynSpecIsFloatParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SynSpecIsFloatPtParam = _neuralgpu.NeuralGPU_SynSpecIsFloatPtParam
NeuralGPU_SynSpecIsFloatPtParam.argtypes = (c_char_p,)
NeuralGPU_SynSpecIsFloatPtParam.restype = ctypes.c_int
def SynSpecIsFloatPtParam(param_name):
    "Check name of synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_SynSpecIsFloatPtParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_ConnectSeq = _neuralgpu.NeuralGPU_ConnectSeq
NeuralGPU_ConnectSeq.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
NeuralGPU_ConnectSeq.restype = ctypes.c_int

NeuralGPU_ConnectGroup = _neuralgpu.NeuralGPU_ConnectGroup
NeuralGPU_ConnectGroup.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_int)
NeuralGPU_ConnectGroup.restype = ctypes.c_int

def Connect(source, target, conn_dict, syn_dict): 
    "Connect two node groups"
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
        
    ConnSpecInit()
    SynSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
                SetConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")
                
        elif ConnSpecIsParam(param_name):
            SetConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")

    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            SetSynSpecIntParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatParam(param_name):
            SetSynSpecFloatParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NeuralGPU_ConnectSeq(source.i0, source.n, target.i0, target.n)

    else:
        if type(source)==NodeSeq:
            source_list = source.ToList()
        else:
            source_list = source
            
        if type(target)==NodeSeq:
            target_list = target.ToList()
        else:
            target_list = target

        source_arr = (ctypes.c_int * len(source_list))(*source_list) 
        source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        target_arr = (ctypes.c_int * len(target_list))(*target_list) 
        target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    

        ret = NeuralGPU_ConnectGroup(source_arr_pt, len(source_list),
                                      target_arr_pt, len(target_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RemoteConnectSeq = _neuralgpu.NeuralGPU_RemoteConnectSeq
NeuralGPU_RemoteConnectSeq.argtypes = (ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int)
NeuralGPU_RemoteConnectSeq.restype = ctypes.c_int

NeuralGPU_RemoteConnectGroup = _neuralgpu.NeuralGPU_RemoteConnectGroup
NeuralGPU_RemoteConnectGroup.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int,
                                         ctypes.c_void_p, ctypes.c_int)
NeuralGPU_RemoteConnectGroup.restype = ctypes.c_int

def RemoteConnect(i_source_host, source, i_target_host, target,
                  conn_dict, syn_dict): 
    "Connect two node groups of differen mpi hosts"
    if (type(i_source_host)!=int) | (type(i_target_host)!=int):
        raise ValueError("Error in host index")
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
        
    ConnSpecInit()
    SynSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
                SetConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")
                
        elif ConnSpecIsParam(param_name):
            SetConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")

    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            SetSynSpecIntParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatParam(param_name):
            SetSynSpecFloatParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NeuralGPU_RemoteConnectSeq(i_source_host, source.i0, source.n,
                                          i_target_host, target.i0, target.n)
    else:
        if type(source)==NodeSeq:
            source_list = source.ToList()
        else:
            source_list = source
            
        if type(target)==NodeSeq:
            target_list = target.ToList()
        else:
            target_list = target

        source_arr = (ctypes.c_int * len(source_list))(*source_list) 
        source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        target_arr = (ctypes.c_int * len(target_list))(*target_list) 
        target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    

        ret = NeuralGPU_RemoteConnectGroup(i_source_host, source_arr_pt,
                                            len(source_list),
                                            i_target_host, target_arr_pt,
                                            len(target_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret
