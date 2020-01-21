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
def Create(model_name, n_nodes=1, n_ports=1):
    "Create a neuron group"
    c_model_name = ctypes.create_string_buffer(str.encode(model_name), len(model_name)+1)
    i_node =NeuralGPU_Create(c_model_name, ctypes.c_int(n_nodes), ctypes.c_int(n_ports))
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
NeuralGPU_SetNeuronScalParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronScalParam.restype = ctypes.c_int
def SetNeuronScalParam(i_node, n_nodes, param_name, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetNeuronScalParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes), c_param_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronArrayParam = _neuralgpu.NeuralGPU_SetNeuronArrayParam
NeuralGPU_SetNeuronArrayParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuralGPU_SetNeuronArrayParam.restype = ctypes.c_int
def SetNeuronArrayParam(i_node, n_nodes, param_name, params_list):
    "Set neuron array parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    array_size = len(params_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronArrayParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes), c_param_name,
                                       array_float_type(*params_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtScalParam = _neuralgpu.NeuralGPU_SetNeuronPtScalParam
NeuralGPU_SetNeuronPtScalParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronPtScalParam.restype = ctypes.c_int
def SetNeuronPtScalParam(node_pt, n_nodes, param_name, val):
    "Set neuron list scalar parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    ret = NeuralGPU_SetNeuronPtScalParam(c_void_p(node_pt),
                                          ctypes.c_int(n_nodes), c_param_name,
                                          ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtArrayParam = _neuralgpu.NeuralGPU_SetNeuronPtArrayParam
NeuralGPU_SetNeuronPtArrayParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuralGPU_SetNeuronPtArrayParam.restype = ctypes.c_int
def SetNeuronPtArrayParam(node_pt, n_nodes, param_name, params_list):
    "Set neuron list array parameter value"
    c_param_name = ctypes.create_string_buffer(str.encode(param_name), len(param_name)+1)
    array_size = len(params_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronPtArrayParam(ctypes.c_void_p(node_pt),
                                         ctypes.c_int(n_nodes),
                                         c_param_name,
                                         array_float_type(*params_list),
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
def SetNeuronScalVar(i_node, n_nodes, var_name, val):
    "Set neuron scalar variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = NeuralGPU_SetNeuronScalVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronArrayVar = _neuralgpu.NeuralGPU_SetNeuronArrayVar
NeuralGPU_SetNeuronArrayVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuralGPU_SetNeuronArrayVar.restype = ctypes.c_int
def SetNeuronArrayVar(i_node, n_nodes, var_name, vars_list):
    "Set neuron array variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    array_size = len(vars_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronArrayVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_nodes), c_var_name,
                                       array_float_type(*vars_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtScalVar = _neuralgpu.NeuralGPU_SetNeuronPtScalVar
NeuralGPU_SetNeuronPtScalVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuralGPU_SetNeuronPtScalVar.restype = ctypes.c_int
def SetNeuronPtScalVar(node_pt, n_nodes, var_name, val):
    "Set neuron list scalar variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    ret = NeuralGPU_SetNeuronPtScalVar(c_void_p(node_pt),
                                          ctypes.c_int(n_nodes), c_var_name,
                                          ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuralGPU_SetNeuronPtArrayVar = _neuralgpu.NeuralGPU_SetNeuronPtArrayVar
NeuralGPU_SetNeuronPtArrayVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuralGPU_SetNeuronPtArrayVar.restype = ctypes.c_int
def SetNeuronPtArrayVar(node_pt, n_nodes, var_name, vars_list):
    "Set neuron list array variable value"
    c_var_name = ctypes.create_string_buffer(str.encode(var_name), len(var_name)+1)
    array_size = len(vars_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuralGPU_SetNeuronPtArrayVar(ctypes.c_void_p(node_pt),
                                         ctypes.c_int(n_nodes),
                                         c_var_name,
                                         array_float_type(*vars_list),
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


def SetNeuronParam(nodes, param_name, val):
    "Set neuron group scalar or array parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_param_name = ctypes.create_string_buffer(str.encode(param_name),
                                               len(param_name)+1)

    if type(nodes)==NodeSeq:
        if IsNeuronScalParam(nodes.i0, param_name):
            SetNeuronScalParam(nodes.i0, nodes.n, param_name, val)
        elif (IsNeuronPortParam(nodes.i0, param_name) |
              IsNeuronArrayParam(nodes.i0, param_name)):
            SetNeuronArrayParam(nodes.i0, nodes.n, param_name, val)
        else:
            raise ValueError("Unknown neuron parameter")
    else:
        node_arr = (ctypes.c_int * len(nodes))(*nodes) 
        node_arr_pt = ctypes.cast(node_arr, ctypes.c_void_p)    

        if IsNeuronScalParam(nodes[0], param_name):
            NeuralGPU_SetNeuronPtScalParam(node_arr_pt, len(nodes),
                                           c_param_name, val)
        elif (IsNeuronPortParam(nodes[0], param_name) |
              IsNeuronArrayParam(nodes.i0, param_name)):
            array_size = len(val)
            array_float_type = ctypes.c_float * array_size
            NeuralGPU_SetNeuronPtArrayParam(node_arr_pt, len(nodes),
                                           c_param_name,
                                           array_float_type(*val),
                                           ctypes.c_int(array_size))
        else:
            raise ValueError("Unknown neuron parameter")

        
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
        node_arr = (ctypes.c_int * len(nodes))(*nodes)
        node_arr_pt = ctypes.cast(node_arr, ctypes.c_void_p)    
        
        if IsNeuronScalParam(nodes[0], var_name):
            NeuralGPU_SetNeuronPtScalParam(node_arr_pt, len(nodes),
                                           c_var_name, val)
        elif (IsNeuronPortParam(nodes[0], var_name) |
              IsNeuronArrayParam(nodes.i0, var_name)):
            array_size = len(val)
            array_float_type = ctypes.c_float * array_size
            NeuralGPU_SetNeuronPtArrayParam(node_arr_pt, len(nodes),
                                            c_var_name,
                                            array_float_type(*val),
                                            ctypes.c_int(array_size))
        elif IsNeuronScalVar(nodes[0], var_name):
            NeuralGPU_SetNeuronPtScalVar(node_arr_pt, len(nodes),
                                         c_var_name, val)
        elif (IsNeuronPortVar(nodes[0], var_name) |
              IsNeuronArrayVar(nodes.i0, var_name)):
            array_size = len(val)
            array_float_type = ctypes.c_float * array_size
            NeuralGPU_SetNeuronPtArrayVar(node_arr_pt, len(nodes),
                                          c_var_name,
                                          array_float_type(*val),
                                          ctypes.c_int(array_size))
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
    



