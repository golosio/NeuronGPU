""" Python interface for NeuralGPU"""
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
    def __len__(self):
        return self.n
    
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


NeuralGPU_SetSimTime = _neuralgpu.NeuralGPU_SetSimTime
NeuralGPU_SetSimTime.argtypes = (ctypes.c_float,)
NeuralGPU_SetSimTime.restype = ctypes.c_int
def SetSimTime(sim_time):
    "Set neural activity simulated time in ms"
    ret = NeuralGPU_SetSimTime(ctypes.c_float(sim_time))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_Create = _neuralgpu.NeuralGPU_Create
NeuralGPU_Create.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int)
NeuralGPU_Create.restype = ctypes.c_int
def Create(model_name, n_node=1, n_ports=1):
    "Create a neuron group"
    c_model_name = ctypes.create_string_buffer(str.encode(model_name), len(model_name)+1)
    i_node =NeuralGPU_Create(c_model_name, ctypes.c_int(n_node), ctypes.c_int(n_ports))
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_CreatePoissonGenerator = _neuralgpu.NeuralGPU_CreatePoissonGenerator
NeuralGPU_CreatePoissonGenerator.argtypes = (ctypes.c_int, ctypes.c_float)
NeuralGPU_CreatePoissonGenerator.restype = ctypes.c_int
def CreatePoissonGenerator(n_node, rate):
    "Create a poisson-distributed spike generator"
    i_node = NeuralGPU_CreatePoissonGenerator(ctypes.c_int(n_node), ctypes.c_float(rate)) 
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_CreateRecord = _neuralgpu.NeuralGPU_CreateRecord
NeuralGPU_CreateRecord.argtypes = (c_char_p, ctypes.POINTER(c_char_p), c_int_p, c_int_p, ctypes.c_int)
NeuralGPU_CreateRecord.restype = ctypes.c_int
def CreateRecord(file_name, var_name_list, i_node_list, i_port_list):
    "Create a record of neuron variables"
    n_node = len(i_node_list)
    c_file_name = ctypes.create_string_buffer(str.encode(file_name), len(file_name)+1)    
    array_int_type = ctypes.c_int * n_node
    array_char_pt_type = c_char_p * n_node
    c_var_name_list=[]
    for i in range(n_node):
        c_var_name = ctypes.create_string_buffer(str.encode(var_name_list[i]), len(var_name_list[i])+1)
        c_var_name_list.append(c_var_name)

    ret = NeuralGPU_CreateRecord(c_file_name,
                                 array_char_pt_type(*c_var_name_list),
                                 array_int_type(*i_node_list),
                                 array_int_type(*i_port_list),
                                 ctypes.c_int(n_node))
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
NeuralGPU_SetNeuronScalParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronScalParam.restype = ctypes.c_int
def SetNeuronScalParam(i_node, n_node, param_name, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetNeuronScalParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronArrayParam = _neuralgpu.NeuralGPU_SetNeuronArrayParam
NeuralGPU_SetNeuronArrayParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuralGPU_SetNeuronArrayParam.restype = ctypes.c_int
def SetNeuronArrayParam(i_node, n_node, param_name, param_list):
    "Set neuron array parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronArrayParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       array_float_type(*param_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtScalParam = _neuralgpu.NeuralGPU_SetNeuronPtScalParam
NeuralGPU_SetNeuronPtScalParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronPtScalParam.restype = ctypes.c_int
def SetNeuronPtScalParam(nodes, param_name, val):
    "Set neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    ret = NeuralGPU_SetNeuronPtScalParam(node_pt,
                                         ctypes.c_int(n_node), c_param_name,
                                         ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtArrayParam = _neuralgpu.NeuralGPU_SetNeuronPtArrayParam
NeuralGPU_SetNeuronPtArrayParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuralGPU_SetNeuronPtArrayParam.restype = ctypes.c_int
def SetNeuronPtArrayParam(nodes, param_name, param_list):
    "Set neuron list array parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronPtArrayParam(node_pt,
                                          ctypes.c_int(n_node),
                                          c_param_name,
                                          array_float_type(*param_list),
                                          ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronScalParam = _neuralgpu.NeuralGPU_IsNeuronScalParam
NeuralGPU_IsNeuronScalParam.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronScalParam.restype = ctypes.c_int
def IsNeuronScalParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    ret = (NeuralGPU_IsNeuronScalParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronPortParam = _neuralgpu.NeuralGPU_IsNeuronPortParam
NeuralGPU_IsNeuronPortParam.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronPortParam.restype = ctypes.c_int
def IsNeuronPortParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_IsNeuronPortParam(ctypes.c_int(i_node), c_param_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_IsNeuronArrayParam = _neuralgpu.NeuralGPU_IsNeuronArrayParam
NeuralGPU_IsNeuronArrayParam.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronArrayParam.restype = ctypes.c_int
def IsNeuronArrayParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = (NeuralGPU_IsNeuronArrayParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_SetNeuronScalVar = _neuralgpu.NeuralGPU_SetNeuronScalVar
NeuralGPU_SetNeuronScalVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronScalVar.restype = ctypes.c_int
def SetNeuronScalVar(i_node, n_node, var_name, val):
    "Set neuron scalar variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = NeuralGPU_SetNeuronScalVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronArrayVar = _neuralgpu.NeuralGPU_SetNeuronArrayVar
NeuralGPU_SetNeuronArrayVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuralGPU_SetNeuronArrayVar.restype = ctypes.c_int
def SetNeuronArrayVar(i_node, n_node, var_name, var_list):
    "Set neuron array variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronArrayVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       array_float_type(*var_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtScalVar = _neuralgpu.NeuralGPU_SetNeuronPtScalVar
NeuralGPU_SetNeuronPtScalVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronPtScalVar.restype = ctypes.c_int
def SetNeuronPtScalVar(nodes, var_name, val):
    "Set neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    ret = NeuralGPU_SetNeuronPtScalVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtArrayVar = _neuralgpu.NeuralGPU_SetNeuronPtArrayVar
NeuralGPU_SetNeuronPtArrayVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuralGPU_SetNeuronPtArrayVar.restype = ctypes.c_int
def SetNeuronPtArrayVar(nodes, var_name, var_list):
    "Set neuron list array variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                             len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronPtArrayVar(node_pt,
                                        ctypes.c_int(n_node),
                                        c_var_name,
                                        array_float_type(*var_list),
                                        ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronScalVar = _neuralgpu.NeuralGPU_IsNeuronScalVar
NeuralGPU_IsNeuronScalVar.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronScalVar.restype = ctypes.c_int
def IsNeuronScalVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    ret = (NeuralGPU_IsNeuronScalVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_IsNeuronPortVar = _neuralgpu.NeuralGPU_IsNeuronPortVar
NeuralGPU_IsNeuronPortVar.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronPortVar.restype = ctypes.c_int
def IsNeuronPortVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = (NeuralGPU_IsNeuronPortVar(ctypes.c_int(i_node), c_var_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_IsNeuronArrayVar = _neuralgpu.NeuralGPU_IsNeuronArrayVar
NeuralGPU_IsNeuronArrayVar.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_IsNeuronArrayVar.restype = ctypes.c_int
def IsNeuronArrayVar(i_node, var_name):
    "Check name of neuron array variable"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = (NeuralGPU_IsNeuronArrayVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetNeuronParamSize = _neuralgpu.NeuralGPU_GetNeuronParamSize
NeuralGPU_GetNeuronParamSize.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_GetNeuronParamSize.restype = ctypes.c_int
def GetNeuronParamSize(i_node, param_name):
    "Get neuron parameter array size"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_GetNeuronParamSize(ctypes.c_int(i_node), c_param_name) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetNeuronParam = _neuralgpu.NeuralGPU_GetNeuronParam
NeuralGPU_GetNeuronParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NeuralGPU_GetNeuronParam.restype = c_float_p
def GetNeuronParam(i_node, n_node, param_name):
    "Get neuron parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    data_pt = NeuralGPU_GetNeuronParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name)

    array_size = GetNeuronParamSize(i_node, param_name)
    data_list = []
    for i_node in range(n_node):
        row_list = []
        for i in range(array_size):
            row_list.append(data_pt[i_node*array_size + i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetNeuronPtParam = _neuralgpu.NeuralGPU_GetNeuronPtParam
NeuralGPU_GetNeuronPtParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NeuralGPU_GetNeuronPtParam.restype = c_float_p
def GetNeuronPtParam(nodes, param_name):
    "Get neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NeuralGPU_GetNeuronPtParam(node_pt,
                                         ctypes.c_int(n_node), c_param_name)
    array_size = GetNeuronParamSize(nodes[0], param_name)

    data_list = []
    for i_node in range(n_node):
        row_list = []
        for i in range(array_size):
            row_list.append(data_pt[i_node*array_size + i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetArrayParam = _neuralgpu.NeuralGPU_GetArrayParam
NeuralGPU_GetArrayParam.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_GetArrayParam.restype = c_float_p
def GetArrayParam(i_node, n_node, param_name):
    "Get neuron array parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NeuralGPU_GetArrayParam(ctypes.c_int(i_node1), c_param_name)
        array_size = GetNeuronParamSize(i_node1, param_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetNeuronListArrayParam(node_list, param_name):
    "Get neuron array parameter"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NeuralGPU_GetArrayParam(ctypes.c_int(i_node), c_param_name)
        array_size = GetNeuronParamSize(i_node, param_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NeuralGPU_GetNeuronVarSize = _neuralgpu.NeuralGPU_GetNeuronVarSize
NeuralGPU_GetNeuronVarSize.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_GetNeuronVarSize.restype = ctypes.c_int
def GetNeuronVarSize(i_node, var_name):
    "Get neuron variable array size"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = NeuralGPU_GetNeuronVarSize(ctypes.c_int(i_node), c_var_name)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetNeuronVar = _neuralgpu.NeuralGPU_GetNeuronVar
NeuralGPU_GetNeuronVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NeuralGPU_GetNeuronVar.restype = c_float_p
def GetNeuronVar(i_node, n_node, var_name):
    "Get neuron variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    data_pt = NeuralGPU_GetNeuronVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name)

    array_size = GetNeuronVarSize(i_node, var_name)

    data_list = []
    for i_node in range(n_node):
        row_list = []
        for i in range(array_size):
            row_list.append(data_pt[i_node*array_size + i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_GetNeuronPtVar = _neuralgpu.NeuralGPU_GetNeuronPtVar
NeuralGPU_GetNeuronPtVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NeuralGPU_GetNeuronPtVar.restype = c_float_p
def GetNeuronPtVar(nodes, var_name):
    "Get neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NeuralGPU_GetNeuronPtVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name)
    array_size = GetNeuronVarSize(nodes[0], var_name)

    data_list = []
    for i_node in range(n_node):
        row_list = []
        for i in range(array_size):
            row_list.append(data_pt[i_node*array_size + i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetArrayVar = _neuralgpu.NeuralGPU_GetArrayVar
NeuralGPU_GetArrayVar.argtypes = (ctypes.c_int, c_char_p)
NeuralGPU_GetArrayVar.restype = c_float_p
def GetArrayVar(i_node, n_node, var_name):
    "Get neuron array variable"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NeuralGPU_GetArrayVar(ctypes.c_int(i_node1), c_var_name)
        array_size = GetNeuronVarSize(i_node1, var_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def GetNeuronListArrayVar(node_list, var_name):
    "Get neuron array variable"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NeuralGPU_GetArrayVar(ctypes.c_int(i_node), c_var_name)
        array_size = GetNeuronVarSize(i_node, var_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetNeuronStatus(nodes, var_name):
    "Get neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    if type(nodes)==NodeSeq:
        if (IsNeuronScalParam(nodes.i0, var_name) |
            IsNeuronPortParam(nodes.i0, var_name)):
            ret = GetNeuronParam(nodes.i0, nodes.n, var_name)
        elif IsNeuronArrayParam(nodes.i0, var_name):
            ret = GetArrayParam(nodes.i0, nodes.n, var_name)
        elif (IsNeuronScalVar(nodes.i0, var_name) |
              IsNeuronPortVar(nodes.i0, var_name)):
            ret = GetNeuronVar(nodes.i0, nodes.n, var_name)
        elif IsNeuronArrayVar(nodes.i0, var_name):
            ret = GetArrayVar(nodes.i0, nodes.n, var_name)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:
        if (IsNeuronScalParam(nodes[0], var_name) |
            IsNeuronPortParam(nodes[0], var_name)):
            ret = GetNeuronPtParam(nodes, var_name)
        elif IsNeuronArrayParam(nodes[0], var_name):
            ret = GetNeuronListArrayParam(nodes, var_name)
        elif (IsNeuronScalVar(nodes[0], var_name) |
              IsNeuronPortVar(nodes[0], var_name)):
            ret = GetNeuronPtVar(nodes, var_name)
        elif IsNeuronArrayVar(nodes[0], var_name):
            ret = GetNeuronListArrayVar(nodes, var_name)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    return ret


NeuralGPU_GetNScalVar = _neuralgpu.NeuralGPU_GetNScalVar
NeuralGPU_GetNScalVar.argtypes = (ctypes.c_int,)
NeuralGPU_GetNScalVar.restype = ctypes.c_int
def GetNScalVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuralGPU_GetNScalVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetScalVarNames = _neuralgpu.NeuralGPU_GetScalVarNames
NeuralGPU_GetScalVarNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetScalVarNames.restype = ctypes.POINTER(c_char_p)
def GetScalVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNScalVar(i_node)
    var_name_pp = ctypes.cast(NeuralGPU_GetScalVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(var_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list

NeuralGPU_GetNPortVar = _neuralgpu.NeuralGPU_GetNPortVar
NeuralGPU_GetNPortVar.argtypes = (ctypes.c_int,)
NeuralGPU_GetNPortVar.restype = ctypes.c_int
def GetNPortVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuralGPU_GetNPortVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetPortVarNames = _neuralgpu.NeuralGPU_GetPortVarNames
NeuralGPU_GetPortVarNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetPortVarNames.restype = ctypes.POINTER(c_char_p)
def GetPortVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNPortVar(i_node)
    var_name_pp = ctypes.cast(NeuralGPU_GetPortVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(var_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list


NeuralGPU_GetNScalParam = _neuralgpu.NeuralGPU_GetNScalParam
NeuralGPU_GetNScalParam.argtypes = (ctypes.c_int,)
NeuralGPU_GetNScalParam.restype = ctypes.c_int
def GetNScalParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuralGPU_GetNScalParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetScalParamNames = _neuralgpu.NeuralGPU_GetScalParamNames
NeuralGPU_GetScalParamNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetScalParamNames.restype = ctypes.POINTER(c_char_p)
def GetScalParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNScalParam(i_node)
    param_name_pp = ctypes.cast(NeuralGPU_GetScalParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list

NeuralGPU_GetNPortParam = _neuralgpu.NeuralGPU_GetNPortParam
NeuralGPU_GetNPortParam.argtypes = (ctypes.c_int,)
NeuralGPU_GetNPortParam.restype = ctypes.c_int
def GetNPortParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuralGPU_GetNPortParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetPortParamNames = _neuralgpu.NeuralGPU_GetPortParamNames
NeuralGPU_GetPortParamNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetPortParamNames.restype = ctypes.POINTER(c_char_p)
def GetPortParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNPortParam(i_node)
    param_name_pp = ctypes.cast(NeuralGPU_GetPortParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NeuralGPU_GetNArrayParam = _neuralgpu.NeuralGPU_GetNArrayParam
NeuralGPU_GetNArrayParam.argtypes = (ctypes.c_int,)
NeuralGPU_GetNArrayParam.restype = ctypes.c_int
def GetNArrayParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuralGPU_GetNArrayParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetArrayParamNames = _neuralgpu.NeuralGPU_GetArrayParamNames
NeuralGPU_GetArrayParamNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetArrayParamNames.restype = ctypes.POINTER(c_char_p)
def GetArrayParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNArrayParam(i_node)
    param_name_pp = ctypes.cast(NeuralGPU_GetArrayParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NeuralGPU_GetNArrayVar = _neuralgpu.NeuralGPU_GetNArrayVar
NeuralGPU_GetNArrayVar.argtypes = (ctypes.c_int,)
NeuralGPU_GetNArrayVar.restype = ctypes.c_int
def GetNArrayVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuralGPU_GetNArrayVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuralGPU_GetArrayVarNames = _neuralgpu.NeuralGPU_GetArrayVarNames
NeuralGPU_GetArrayVarNames.argtypes = (ctypes.c_int,)
NeuralGPU_GetArrayVarNames.restype = ctypes.POINTER(c_char_p)
def GetArrayVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNArrayVar(i_node)
    var_name_pp = ctypes.cast(NeuralGPU_GetArrayVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(var_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list




def SetNeuronStatus(nodes, var_name, val):
    "Set neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_var_name = ctypes.create_string_buffer(str.encode(var_name),
                                               len(var_name)+1)
    if type(nodes)==NodeSeq:
        if IsNeuronScalParam(nodes.i0, var_name):
            SetNeuronScalParam(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortParam(nodes.i0, var_name) |
              IsNeuronArrayParam(nodes.i0, var_name)):
            SetNeuronArrayParam(nodes.i0, nodes.n, var_name, val)
        elif IsNeuronScalVar(nodes.i0, var_name):
            SetNeuronScalVar(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortVar(nodes.i0, var_name) |
              IsNeuronArrayVar(nodes.i0, var_name)):
            SetNeuronArrayVar(nodes.i0, nodes.n, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:        
        if IsNeuronScalParam(nodes[0], var_name):
            SetNeuronPtScalParam(nodes, var_name, val)
        elif (IsNeuronPortParam(nodes[0], var_name) |
              IsNeuronArrayParam(nodes[0], var_name)):
            SetNeuronPtArrayParam(nodes, var_name, val)
        elif IsNeuronScalVar(nodes[0], var_name):
            SetNeuronPtScalVar(nodes, var_name, val)
        elif (IsNeuronPortVar(nodes[0], var_name) |
              IsNeuronArrayVar(nodes[0], var_name)):
            SetNeuronPtArrayVar(nodes, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")


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
def Simulate(sim_time=1000.0):
    "Simulate neural activity"
    SetSimTime(sim_time)
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
    if (type(arr) is list)  | (type(arr) is tuple):
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


def DictToArray(param_dict, array_size):
    dist_name = None
    arr = None
    low = -1.0e35
    high = 1.0e35
    mu = None
    sigma = None
    
    for param_name in param_dict:
        pval = param_dict[param_name]
        if param_name=="array":
            dist_name = "array"
            arr = pval
        elif param_name=="distribution":
            dist_name = pval
        elif param_name=="low":
            low = pval
        elif param_name=="high":
            high = pval
        elif param_name=="mu":
            mu = pval
        elif param_name=="sigma":
            sigma = pval
        else:
            raise ValueError("Unknown parameter name in dictionary")

    if dist_name=="array":
        if (type(arr) is list) | (type(arr) is tuple):
            if len(arr) != array_size:
                raise ValueError("Wrong array size.")
            arr = (ctypes.c_float * len(arr))(*arr)
            #array_pt = ctypes.cast(arr, ctypes.c_void_p)
            #return array_pt
        return arr
    elif dist_name=="normal":
        return RandomNormal(array_size, mu, sigma)
    elif dist_name=="normal_clipped":
        return RandomNormalClipped(array_size, mu, sigma, low, high)
    else:
        raise ValueError("Unknown distribution")


def RuleArraySize(conn_dict, source, target):
    if conn_dict["rule"]=="one_to_one":
        array_size = len(source)
    elif conn_dict["rule"]=="all_to_all":
        array_size = len(source)*len(target)
    elif conn_dict["rule"]=="fixed_total_number":
        array_size = conn_dict["total_num"]
    elif conn_dict["rule"]=="fixed_indegree":
        array_size = len(target)*conn_dict["indegree"]
    elif conn_dict["rule"]=="fixed_outdegree":
        array_size = len(source)*conn_dict["outdegree"]
    else:
        raise ValueError("Unknown number of connections for this rule")
    return array_size


def SetSynParamFromArray(param_name, par_dict, array_size):
    arr_param_name = param_name + "_array"
    if (not SynSpecIsFloatPtParam(arr_param_name)):
        raise ValueError("Synapse parameter cannot be set by"
                         " arrays or distributions")
    arr = DictToArray(par_dict, array_size)
    array_pt = ctypes.cast(arr, ctypes.c_void_p)
    SetSynSpecFloatPtParam(arr_param_name, array_pt)
    

    

NeuralGPU_ConnectSeqSeq = _neuralgpu.NeuralGPU_ConnectSeqSeq
NeuralGPU_ConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int)
NeuralGPU_ConnectSeqSeq.restype = ctypes.c_int

NeuralGPU_ConnectSeqGroup = _neuralgpu.NeuralGPU_ConnectSeqGroup
NeuralGPU_ConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_int)
NeuralGPU_ConnectSeqGroup.restype = ctypes.c_int

NeuralGPU_ConnectGroupSeq = _neuralgpu.NeuralGPU_ConnectGroupSeq
NeuralGPU_ConnectGroupSeq.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int)
NeuralGPU_ConnectGroupSeq.restype = ctypes.c_int

NeuralGPU_ConnectGroupGroup = _neuralgpu.NeuralGPU_ConnectGroupGroup
NeuralGPU_ConnectGroupGroup.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int)
NeuralGPU_ConnectGroupGroup.restype = ctypes.c_int

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
    
    array_size = RuleArraySize(conn_dict, source, target)
    
    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            SetSynSpecIntParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                SetSynParamFromArray(param_name, fpar, array_size)
            else:
                SetSynSpecFloatParam(param_name, fpar)

        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NeuralGPU_ConnectSeqSeq(source.i0, source.n, target.i0, target.n)
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NeuralGPU_ConnectSeqGroup(source.i0, source.n, target_arr_pt,
                                            len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NeuralGPU_ConnectGroupSeq(source_arr_pt, len(source),
                                            target.i0, target.n)
        else:
            ret = NeuralGPU_ConnectGroupGroup(source_arr_pt, len(source),
                                              target_arr_pt, len(target))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_RemoteConnectSeqSeq = _neuralgpu.NeuralGPU_RemoteConnectSeqSeq
NeuralGPU_RemoteConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int)
NeuralGPU_RemoteConnectSeqSeq.restype = ctypes.c_int

NeuralGPU_RemoteConnectSeqGroup = _neuralgpu.NeuralGPU_RemoteConnectSeqGroup
NeuralGPU_RemoteConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int)
NeuralGPU_RemoteConnectSeqGroup.restype = ctypes.c_int

NeuralGPU_RemoteConnectGroupSeq = _neuralgpu.NeuralGPU_RemoteConnectGroupSeq
NeuralGPU_RemoteConnectGroupSeq.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int)
NeuralGPU_RemoteConnectGroupSeq.restype = ctypes.c_int

NeuralGPU_RemoteConnectGroupGroup = _neuralgpu.NeuralGPU_RemoteConnectGroupGroup
NeuralGPU_RemoteConnectGroupGroup.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                              ctypes.c_int, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_int)
NeuralGPU_RemoteConnectGroupGroup.restype = ctypes.c_int

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
        
    array_size = RuleArraySize(conn_dict, source, target)    
        
    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            SetSynSpecIntParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                SetSynParamFromArray(param_name, fpar, array_size)
            else:
                SetSynSpecFloatParam(param_name, fpar)
                
        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NeuralGPU_RemoteConnectSeqSeq(i_source_host, source.i0, source.n,
                                            i_target_host, target.i0, target.n)

    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NeuralGPU_RemoteConnectSeqGroup(i_source_host, source.i0,
                                                  source.n, i_target_host,
                                                  target_arr_pt, len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NeuralGPU_RemoteConnectGroupSeq(i_source_host, source_arr_pt,
                                                  len(source_list),
                                                  i_target_host, target.i0,
                                                  target.n)
        else:
            ret = NeuralGPU_RemoteConnectGroupGroup(i_source_host,
                                                    source_arr_pt,
                                                    len(source_list),
                                                    i_target_host,
                                                    target_arr_pt,
                                                    len(target_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def SetStatus(nodes, params, val=None):
    "Set neuron group parameters or variables using dictionaries"
    if val != None:
         SetNeuronStatus(nodes, params, val)
    elif type(params)==dict:
        for param_name in params:
            SetNeuronStatus(nodes, param_name, params[param_name])
    elif (type(params)==list)  | (type(params) is tuple):
        if len(params) != len(nodes):
            raise ValueError("List should have the same size as nodes")
        for param_dict in params:
            if type(param_dict)!=dict:
                raise ValueError("Type of list elements should be dict")
            for param_name in param_dict:
                SetNeuronStatus(nodes, param_name, param_dict[param_name])
    else:
        raise ValueError("Wrong argument in SetStatus")       
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    


def GetStatus(nodes, var_name=None):
    "Get neuron group scalar or array variable or parameter"
    if var_name != None:
         return GetNeuronStatus(nodes, var_name)
    if type(nodes)==NodeSeq:
        nodes = nodes.ToList()
    if (type(nodes)!=list) & (type(nodes)!=tuple):
        nodes = [nodes]
    dict_list = []
    for i_node in nodes:
        status_dict = {}
        name_list = GetScalVarNames(i_node) + GetScalParamNames(i_node) + GetPortVarNames(i_node) + GetPortParamNames(i_node) + GetArrayVarNames(i_node) + GetArrayParamNames(i_node)
        for var_name in name_list:
            val = GetNeuronStatus([i_node], var_name)[0]
            status_dict[var_name] = val
        dict_list.append(status_dict)
    return dict_list



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

NeuralGPU_GetSeqSeqConnections = _neuralgpu.NeuralGPU_GetSeqSeqConnections
NeuralGPU_GetSeqSeqConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, c_int_p)
NeuralGPU_GetSeqSeqConnections.restype = c_int_p

NeuralGPU_GetSeqGroupConnections = _neuralgpu.NeuralGPU_GetSeqGroupConnections
NeuralGPU_GetSeqGroupConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                             c_void_p, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NeuralGPU_GetSeqGroupConnections.restype = c_int_p

NeuralGPU_GetGroupSeqConnections = _neuralgpu.NeuralGPU_GetGroupSeqConnections
NeuralGPU_GetGroupSeqConnections.argtypes = (c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NeuralGPU_GetGroupSeqConnections.restype = c_int_p

NeuralGPU_GetGroupGroupConnections = _neuralgpu.NeuralGPU_GetGroupGroupConnections
NeuralGPU_GetGroupGroupConnections.argtypes = (c_void_p, ctypes.c_int,
                                               c_void_p, ctypes.c_int,
                                               ctypes.c_int, c_int_p)
NeuralGPU_GetGroupGroupConnections.restype = c_int_p

def GetConnections(source=None, target=None, syn_type=0): 
    "Get connections between two node groups"
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
    
    n_conn = ctypes.c_int(0)
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        conn_arr = NeuralGPU_GetSeqSeqConnections(source.i0, source.n,
                                                  target.i0, target.n, syn_type,
                                                  ctypes.byref(n_conn))
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            conn_arr = NeuralGPU_GetSeqGroupConnections(source.i0, source.n,
                                                        target_arr_pt,
                                                        len(target),
                                                        syn_type,
                                                        c_types.byref(n_conn))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            conn_arr = NeuralGPU_GetGroupSeqConnections(source_arr_pt,
                                                        len(source),
                                                        target.i0, target.n,
                                                        syn_type,
                                                        ctypes.byref(n_conn))
        else:
            conn_arr = NeuralGPU_GetGroupGroupConnections(source_arr_pt,
                                                          len(source),
                                                          target_arr_pt,
                                                          len(target),
                                                          syn_type,
                                                          ctypes.byref(n_conn))

    conn_list = []
    for i_conn in range(n_conn.value):
        conn_id = [conn_arr[i_conn*3], conn_arr[i_conn*3 + 1],
                   conn_arr[i_conn*3 + 2]]
        conn_list.append(conn_id)
        
    ret = conn_list

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    

