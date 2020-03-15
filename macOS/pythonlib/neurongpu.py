""" Python interface for NeuronGPU"""
import sys, platform
import ctypes, ctypes.util
import os
import unicodedata

print('-----------------------------------------------------------------')
print('NeuronGPU')
print('A GPU-MPI library for simulation of large-scale networks')
print(' of spiking neurons')
print('Homepage: https://github.com/golosio/NeuronGPU') 
print('Author: B. Golosio, University of Cagliari')
print('email: golosio@unica.it')
print('-----------------------------------------------------------------')

lib_path="/usr/local/lib/libneurongpu.so"
_neurongpu=ctypes.CDLL(lib_path)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_char_p = ctypes.POINTER(ctypes.c_char)
c_void_p = ctypes.c_void_p

class NodeSeq(object):
    def __init__(self, i0, n=1):
        if i0 == None:
            i0 = 0
            n = -1
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


class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

class SynGroup(object):
    def __init__(self, i_syn_group):
        self.i_syn_group = i_syn_group

def to_byte_str(s):
    if type(s)==str:
        return s.encode('ascii')
    elif type(s)==bytes:
        return s
    else:
        raise ValueError("Variable cannot be converted to string")

def to_def_str(s):
    if (sys.version_info >= (3, 0)):
        return s.decode("utf-8")
    else:
        return s

def waitenter(val):
    if (sys.version_info >= (3, 0)):
        return input(val)
    else:
        return raw_input(val)
    
conn_rule_name = ("one_to_one", "all_to_all", "fixed_total_number",
                  "fixed_indegree", "fixed_outdegree")
    
NeuronGPU_GetErrorMessage = _neurongpu.NeuronGPU_GetErrorMessage
NeuronGPU_GetErrorMessage.restype = ctypes.POINTER(ctypes.c_char)
def GetErrorMessage():
    "Get error message from NeuronGPU exception"
    message = ctypes.cast(NeuronGPU_GetErrorMessage(), ctypes.c_char_p).value
    return message
 
NeuronGPU_GetErrorCode = _neurongpu.NeuronGPU_GetErrorCode
NeuronGPU_GetErrorCode.restype = ctypes.c_ubyte
def GetErrorCode():
    "Get error code from NeuronGPU exception"
    return NeuronGPU_GetErrorCode()
 
NeuronGPU_SetOnException = _neurongpu.NeuronGPU_SetOnException
NeuronGPU_SetOnException.argtypes = (ctypes.c_int,)
def SetOnException(on_exception):
    "Define whether handle exceptions (1) or exit (0) in case of errors"
    return NeuronGPU_SetOnException(ctypes.c_int(on_exception))

SetOnException(1)

NeuronGPU_SetRandomSeed = _neurongpu.NeuronGPU_SetRandomSeed
NeuronGPU_SetRandomSeed.argtypes = (ctypes.c_ulonglong,)
NeuronGPU_SetRandomSeed.restype = ctypes.c_int
def SetRandomSeed(seed):
    "Set seed for random number generation"
    ret = NeuronGPU_SetRandomSeed(ctypes.c_ulonglong(seed))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetTimeResolution = _neurongpu.NeuronGPU_SetTimeResolution
NeuronGPU_SetTimeResolution.argtypes = (ctypes.c_float,)
NeuronGPU_SetTimeResolution.restype = ctypes.c_int
def SetTimeResolution(time_res):
    "Set time resolution in ms"
    ret = NeuronGPU_SetTimeResolution(ctypes.c_float(time_res))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetTimeResolution = _neurongpu.NeuronGPU_GetTimeResolution
NeuronGPU_GetTimeResolution.restype = ctypes.c_float
def GetTimeResolution():
    "Get time resolution in ms"
    ret = NeuronGPU_GetTimeResolution()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetMaxSpikeBufferSize = _neurongpu.NeuronGPU_SetMaxSpikeBufferSize
NeuronGPU_SetMaxSpikeBufferSize.argtypes = (ctypes.c_int,)
NeuronGPU_SetMaxSpikeBufferSize.restype = ctypes.c_int
def SetMaxSpikeBufferSize(max_size):
    "Set maximum size of spike buffer per node"
    ret = NeuronGPU_SetMaxSpikeBufferSize(ctypes.c_int(max_size))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetMaxSpikeBufferSize = _neurongpu.NeuronGPU_GetMaxSpikeBufferSize
NeuronGPU_GetMaxSpikeBufferSize.restype = ctypes.c_int
def GetMaxSpikeBufferSize():
    "Get maximum size of spike buffer per node"
    ret = NeuronGPU_GetMaxSpikeBufferSize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetSimTime = _neurongpu.NeuronGPU_SetSimTime
NeuronGPU_SetSimTime.argtypes = (ctypes.c_float,)
NeuronGPU_SetSimTime.restype = ctypes.c_int
def SetSimTime(sim_time):
    "Set neural activity simulated time in ms"
    ret = NeuronGPU_SetSimTime(ctypes.c_float(sim_time))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_Create = _neurongpu.NeuronGPU_Create
NeuronGPU_Create.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int)
NeuronGPU_Create.restype = ctypes.c_int
def Create(model_name, n_node=1, n_ports=1, status_dict=None):
    "Create a neuron group"
    if (type(status_dict)==dict):
        node_group = Create(model_name, n_node, n_ports)
        SetStatus(node_group, status_dict)
        return node_group
        
    elif status_dict!=None:
        raise ValueError("Wrong argument in Create")
    
    c_model_name = ctypes.create_string_buffer(to_byte_str(model_name), len(model_name)+1)
    i_node =NeuronGPU_Create(c_model_name, ctypes.c_int(n_node), ctypes.c_int(n_ports))
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_CreatePoissonGenerator = _neurongpu.NeuronGPU_CreatePoissonGenerator
NeuronGPU_CreatePoissonGenerator.argtypes = (ctypes.c_int, ctypes.c_float)
NeuronGPU_CreatePoissonGenerator.restype = ctypes.c_int
def CreatePoissonGenerator(n_node, rate):
    "Create a poisson-distributed spike generator"
    i_node = NeuronGPU_CreatePoissonGenerator(ctypes.c_int(n_node), ctypes.c_float(rate)) 
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_CreateRecord = _neurongpu.NeuronGPU_CreateRecord
NeuronGPU_CreateRecord.argtypes = (c_char_p, ctypes.POINTER(c_char_p), c_int_p, c_int_p, ctypes.c_int)
NeuronGPU_CreateRecord.restype = ctypes.c_int
def CreateRecord(file_name, var_name_list, i_node_list, i_port_list):
    "Create a record of neuron variables"
    n_node = len(i_node_list)
    c_file_name = ctypes.create_string_buffer(to_byte_str(file_name), len(file_name)+1)    
    array_int_type = ctypes.c_int * n_node
    array_char_pt_type = c_char_p * n_node
    c_var_name_list=[]
    for i in range(n_node):
        c_var_name = ctypes.create_string_buffer(to_byte_str(var_name_list[i]), len(var_name_list[i])+1)
        c_var_name_list.append(c_var_name)

    ret = NeuronGPU_CreateRecord(c_file_name,
                                 array_char_pt_type(*c_var_name_list),
                                 array_int_type(*i_node_list),
                                 array_int_type(*i_port_list),
                                 ctypes.c_int(n_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetRecordDataRows = _neurongpu.NeuronGPU_GetRecordDataRows
NeuronGPU_GetRecordDataRows.argtypes = (ctypes.c_int,)
NeuronGPU_GetRecordDataRows.restype = ctypes.c_int
def GetRecordDataRows(i_record):
    "Get record n. of rows"
    ret = NeuronGPU_GetRecordDataRows(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetRecordDataColumns = _neurongpu.NeuronGPU_GetRecordDataColumns
NeuronGPU_GetRecordDataColumns.argtypes = (ctypes.c_int,)
NeuronGPU_GetRecordDataColumns.restype = ctypes.c_int
def GetRecordDataColumns(i_record):
    "Get record n. of columns"
    ret = NeuronGPU_GetRecordDataColumns(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetRecordData = _neurongpu.NeuronGPU_GetRecordData
NeuronGPU_GetRecordData.argtypes = (ctypes.c_int,)
NeuronGPU_GetRecordData.restype = ctypes.POINTER(c_float_p)
def GetRecordData(i_record):
    "Get record data"
    data_arr_pt = NeuronGPU_GetRecordData(ctypes.c_int(i_record))
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


NeuronGPU_SetNeuronScalParam = _neurongpu.NeuronGPU_SetNeuronScalParam
NeuronGPU_SetNeuronScalParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NeuronGPU_SetNeuronScalParam.restype = ctypes.c_int
def SetNeuronScalParam(i_node, n_node, param_name, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NeuronGPU_SetNeuronScalParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronArrayParam = _neurongpu.NeuronGPU_SetNeuronArrayParam
NeuronGPU_SetNeuronArrayParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuronGPU_SetNeuronArrayParam.restype = ctypes.c_int
def SetNeuronArrayParam(i_node, n_node, param_name, param_list):
    "Set neuron array parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuronGPU_SetNeuronArrayParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       array_float_type(*param_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronPtScalParam = _neurongpu.NeuronGPU_SetNeuronPtScalParam
NeuronGPU_SetNeuronPtScalParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuronGPU_SetNeuronPtScalParam.restype = ctypes.c_int
def SetNeuronPtScalParam(nodes, param_name, val):
    "Set neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    ret = NeuronGPU_SetNeuronPtScalParam(node_pt,
                                         ctypes.c_int(n_node), c_param_name,
                                         ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronPtArrayParam = _neurongpu.NeuronGPU_SetNeuronPtArrayParam
NeuronGPU_SetNeuronPtArrayParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuronGPU_SetNeuronPtArrayParam.restype = ctypes.c_int
def SetNeuronPtArrayParam(nodes, param_name, param_list):
    "Set neuron list array parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuronGPU_SetNeuronPtArrayParam(node_pt,
                                          ctypes.c_int(n_node),
                                          c_param_name,
                                          array_float_type(*param_list),
                                          ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_IsNeuronScalParam = _neurongpu.NeuronGPU_IsNeuronScalParam
NeuronGPU_IsNeuronScalParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronScalParam.restype = ctypes.c_int
def IsNeuronScalParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NeuronGPU_IsNeuronScalParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_IsNeuronPortParam = _neurongpu.NeuronGPU_IsNeuronPortParam
NeuronGPU_IsNeuronPortParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronPortParam.restype = ctypes.c_int
def IsNeuronPortParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_IsNeuronPortParam(ctypes.c_int(i_node), c_param_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_IsNeuronArrayParam = _neurongpu.NeuronGPU_IsNeuronArrayParam
NeuronGPU_IsNeuronArrayParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronArrayParam.restype = ctypes.c_int
def IsNeuronArrayParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_IsNeuronArrayParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_SetNeuronScalVar = _neurongpu.NeuronGPU_SetNeuronScalVar
NeuronGPU_SetNeuronScalVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NeuronGPU_SetNeuronScalVar.restype = ctypes.c_int
def SetNeuronScalVar(i_node, n_node, var_name, val):
    "Set neuron scalar variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = NeuronGPU_SetNeuronScalVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronArrayVar = _neurongpu.NeuronGPU_SetNeuronArrayVar
NeuronGPU_SetNeuronArrayVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NeuronGPU_SetNeuronArrayVar.restype = ctypes.c_int
def SetNeuronArrayVar(i_node, n_node, var_name, var_list):
    "Set neuron array variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuronGPU_SetNeuronArrayVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       array_float_type(*var_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronPtScalVar = _neurongpu.NeuronGPU_SetNeuronPtScalVar
NeuronGPU_SetNeuronPtScalVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NeuronGPU_SetNeuronPtScalVar.restype = ctypes.c_int
def SetNeuronPtScalVar(nodes, var_name, val):
    "Set neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    ret = NeuronGPU_SetNeuronPtScalVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetNeuronPtArrayVar = _neurongpu.NeuronGPU_SetNeuronPtArrayVar
NeuronGPU_SetNeuronPtArrayVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NeuronGPU_SetNeuronPtArrayVar.restype = ctypes.c_int
def SetNeuronPtArrayVar(nodes, var_name, var_list):
    "Set neuron list array variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                             len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NeuronGPU_SetNeuronPtArrayVar(node_pt,
                                        ctypes.c_int(n_node),
                                        c_var_name,
                                        array_float_type(*var_list),
                                        ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_IsNeuronScalVar = _neurongpu.NeuronGPU_IsNeuronScalVar
NeuronGPU_IsNeuronScalVar.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronScalVar.restype = ctypes.c_int
def IsNeuronScalVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    ret = (NeuronGPU_IsNeuronScalVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_IsNeuronPortVar = _neurongpu.NeuronGPU_IsNeuronPortVar
NeuronGPU_IsNeuronPortVar.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronPortVar.restype = ctypes.c_int
def IsNeuronPortVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = (NeuronGPU_IsNeuronPortVar(ctypes.c_int(i_node), c_var_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_IsNeuronArrayVar = _neurongpu.NeuronGPU_IsNeuronArrayVar
NeuronGPU_IsNeuronArrayVar.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsNeuronArrayVar.restype = ctypes.c_int
def IsNeuronArrayVar(i_node, var_name):
    "Check name of neuron array variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = (NeuronGPU_IsNeuronArrayVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetNeuronParamSize = _neurongpu.NeuronGPU_GetNeuronParamSize
NeuronGPU_GetNeuronParamSize.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_GetNeuronParamSize.restype = ctypes.c_int
def GetNeuronParamSize(i_node, param_name):
    "Get neuron parameter array size"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NeuronGPU_GetNeuronParamSize(ctypes.c_int(i_node), c_param_name) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetNeuronParam = _neurongpu.NeuronGPU_GetNeuronParam
NeuronGPU_GetNeuronParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NeuronGPU_GetNeuronParam.restype = c_float_p
def GetNeuronParam(i_node, n_node, param_name):
    "Get neuron parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_pt = NeuronGPU_GetNeuronParam(ctypes.c_int(i_node),
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


NeuronGPU_GetNeuronPtParam = _neurongpu.NeuronGPU_GetNeuronPtParam
NeuronGPU_GetNeuronPtParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NeuronGPU_GetNeuronPtParam.restype = c_float_p
def GetNeuronPtParam(nodes, param_name):
    "Get neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NeuronGPU_GetNeuronPtParam(node_pt,
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

NeuronGPU_GetArrayParam = _neurongpu.NeuronGPU_GetArrayParam
NeuronGPU_GetArrayParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_GetArrayParam.restype = c_float_p
def GetArrayParam(i_node, n_node, param_name):
    "Get neuron array parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NeuronGPU_GetArrayParam(ctypes.c_int(i_node1), c_param_name)
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
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NeuronGPU_GetArrayParam(ctypes.c_int(i_node), c_param_name)
        array_size = GetNeuronParamSize(i_node, param_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NeuronGPU_GetNeuronVarSize = _neurongpu.NeuronGPU_GetNeuronVarSize
NeuronGPU_GetNeuronVarSize.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_GetNeuronVarSize.restype = ctypes.c_int
def GetNeuronVarSize(i_node, var_name):
    "Get neuron variable array size"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = NeuronGPU_GetNeuronVarSize(ctypes.c_int(i_node), c_var_name)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_GetNeuronVar = _neurongpu.NeuronGPU_GetNeuronVar
NeuronGPU_GetNeuronVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NeuronGPU_GetNeuronVar.restype = c_float_p
def GetNeuronVar(i_node, n_node, var_name):
    "Get neuron variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_pt = NeuronGPU_GetNeuronVar(ctypes.c_int(i_node),
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


NeuronGPU_GetNeuronPtVar = _neurongpu.NeuronGPU_GetNeuronPtVar
NeuronGPU_GetNeuronPtVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NeuronGPU_GetNeuronPtVar.restype = c_float_p
def GetNeuronPtVar(nodes, var_name):
    "Get neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NeuronGPU_GetNeuronPtVar(node_pt,
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

NeuronGPU_GetArrayVar = _neurongpu.NeuronGPU_GetArrayVar
NeuronGPU_GetArrayVar.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_GetArrayVar.restype = c_float_p
def GetArrayVar(i_node, n_node, var_name):
    "Get neuron array variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NeuronGPU_GetArrayVar(ctypes.c_int(i_node1), c_var_name)
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
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NeuronGPU_GetArrayVar(ctypes.c_int(i_node), c_var_name)
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
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
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


NeuronGPU_GetNScalVar = _neurongpu.NeuronGPU_GetNScalVar
NeuronGPU_GetNScalVar.argtypes = (ctypes.c_int,)
NeuronGPU_GetNScalVar.restype = ctypes.c_int
def GetNScalVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuronGPU_GetNScalVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetScalVarNames = _neurongpu.NeuronGPU_GetScalVarNames
NeuronGPU_GetScalVarNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetScalVarNames.restype = ctypes.POINTER(c_char_p)
def GetScalVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNScalVar(i_node)
    var_name_pp = ctypes.cast(NeuronGPU_GetScalVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list

NeuronGPU_GetNPortVar = _neurongpu.NeuronGPU_GetNPortVar
NeuronGPU_GetNPortVar.argtypes = (ctypes.c_int,)
NeuronGPU_GetNPortVar.restype = ctypes.c_int
def GetNPortVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuronGPU_GetNPortVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetPortVarNames = _neurongpu.NeuronGPU_GetPortVarNames
NeuronGPU_GetPortVarNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetPortVarNames.restype = ctypes.POINTER(c_char_p)
def GetPortVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNPortVar(i_node)
    var_name_pp = ctypes.cast(NeuronGPU_GetPortVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list


NeuronGPU_GetNScalParam = _neurongpu.NeuronGPU_GetNScalParam
NeuronGPU_GetNScalParam.argtypes = (ctypes.c_int,)
NeuronGPU_GetNScalParam.restype = ctypes.c_int
def GetNScalParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuronGPU_GetNScalParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetScalParamNames = _neurongpu.NeuronGPU_GetScalParamNames
NeuronGPU_GetScalParamNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetScalParamNames.restype = ctypes.POINTER(c_char_p)
def GetScalParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNScalParam(i_node)
    param_name_pp = ctypes.cast(NeuronGPU_GetScalParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list

NeuronGPU_GetNPortParam = _neurongpu.NeuronGPU_GetNPortParam
NeuronGPU_GetNPortParam.argtypes = (ctypes.c_int,)
NeuronGPU_GetNPortParam.restype = ctypes.c_int
def GetNPortParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuronGPU_GetNPortParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetPortParamNames = _neurongpu.NeuronGPU_GetPortParamNames
NeuronGPU_GetPortParamNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetPortParamNames.restype = ctypes.POINTER(c_char_p)
def GetPortParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNPortParam(i_node)
    param_name_pp = ctypes.cast(NeuronGPU_GetPortParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NeuronGPU_GetNArrayParam = _neurongpu.NeuronGPU_GetNArrayParam
NeuronGPU_GetNArrayParam.argtypes = (ctypes.c_int,)
NeuronGPU_GetNArrayParam.restype = ctypes.c_int
def GetNArrayParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NeuronGPU_GetNArrayParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetArrayParamNames = _neurongpu.NeuronGPU_GetArrayParamNames
NeuronGPU_GetArrayParamNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetArrayParamNames.restype = ctypes.POINTER(c_char_p)
def GetArrayParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNArrayParam(i_node)
    param_name_pp = ctypes.cast(NeuronGPU_GetArrayParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NeuronGPU_GetNArrayVar = _neurongpu.NeuronGPU_GetNArrayVar
NeuronGPU_GetNArrayVar.argtypes = (ctypes.c_int,)
NeuronGPU_GetNArrayVar.restype = ctypes.c_int
def GetNArrayVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NeuronGPU_GetNArrayVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_GetArrayVarNames = _neurongpu.NeuronGPU_GetArrayVarNames
NeuronGPU_GetArrayVarNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetArrayVarNames.restype = ctypes.POINTER(c_char_p)
def GetArrayVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNArrayVar(i_node)
    var_name_pp = ctypes.cast(NeuronGPU_GetArrayVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list




def SetNeuronStatus(nodes, var_name, val):
    "Set neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
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


NeuronGPU_Calibrate = _neurongpu.NeuronGPU_Calibrate
NeuronGPU_Calibrate.restype = ctypes.c_int
def Calibrate():
    "Calibrate simulation"
    ret = NeuronGPU_Calibrate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_Simulate = _neurongpu.NeuronGPU_Simulate
NeuronGPU_Simulate.restype = ctypes.c_int
def Simulate(sim_time=1000.0):
    "Simulate neural activity"
    SetSimTime(sim_time)
    ret = NeuronGPU_Simulate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_ConnectMpiInit = _neurongpu.NeuronGPU_ConnectMpiInit
NeuronGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(c_char_p))
NeuronGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = c_char_p * argc
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(to_byte_str(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    ret = NeuronGPU_ConnectMpiInit(ctypes.c_int(argc),
                                   array_char_pt_type(*c_var_name_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_MpiId = _neurongpu.NeuronGPU_MpiId
NeuronGPU_MpiId.restype = ctypes.c_int
def MpiId():
    "Get MPI Id"
    ret = NeuronGPU_MpiId()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_MpiNp = _neurongpu.NeuronGPU_MpiNp
NeuronGPU_MpiNp.restype = ctypes.c_int
def MpiNp():
    "Get MPI Np"
    ret = NeuronGPU_MpiNp()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_ProcMaster = _neurongpu.NeuronGPU_ProcMaster
NeuronGPU_ProcMaster.restype = ctypes.c_int
def ProcMaster():
    "Get MPI ProcMaster"
    ret = NeuronGPU_ProcMaster()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_MpiFinalize = _neurongpu.NeuronGPU_MpiFinalize
NeuronGPU_MpiFinalize.restype = ctypes.c_int
def MpiFinalize():
    "Finalize MPI"
    ret = NeuronGPU_MpiFinalize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_RandomInt = _neurongpu.NeuronGPU_RandomInt
NeuronGPU_RandomInt.argtypes = (ctypes.c_size_t,)
NeuronGPU_RandomInt.restype = ctypes.POINTER(ctypes.c_uint)
def RandomInt(n):
    "Generate n random integers in CUDA memory"
    ret = NeuronGPU_RandomInt(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_RandomUniform = _neurongpu.NeuronGPU_RandomUniform
NeuronGPU_RandomUniform.argtypes = (ctypes.c_size_t,)
NeuronGPU_RandomUniform.restype = c_float_p
def RandomUniform(n):
    "Generate n random floats with uniform distribution in (0,1) in CUDA memory"
    ret = NeuronGPU_RandomUniform(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_RandomNormal = _neurongpu.NeuronGPU_RandomNormal
NeuronGPU_RandomNormal.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float)
NeuronGPU_RandomNormal.restype = c_float_p
def RandomNormal(n, mean, stddev):
    "Generate n random floats with normal distribution in CUDA memory"
    ret = NeuronGPU_RandomNormal(ctypes.c_size_t(n), ctypes.c_float(mean),
                                 ctypes.c_float(stddev))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_RandomNormalClipped = _neurongpu.NeuronGPU_RandomNormalClipped
NeuronGPU_RandomNormalClipped.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                          ctypes.c_float)
NeuronGPU_RandomNormalClipped.restype = c_float_p
def RandomNormalClipped(n, mean, stddev, vmin, vmax):
    "Generate n random floats with normal clipped distribution in CUDA memory"
    ret = NeuronGPU_RandomNormalClipped(ctypes.c_size_t(n),
                                        ctypes.c_float(mean),
                                        ctypes.c_float(stddev),
                                        ctypes.c_float(vmin),
                                        ctypes.c_float(vmax))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_ConnectMpiInit = _neurongpu.NeuronGPU_ConnectMpiInit
NeuronGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(c_char_p))
NeuronGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    array_char_pt_type = c_char_p * argc
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(to_byte_str(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    ret = NeuronGPU_ConnectMpiInit(ctypes.c_int(argc),
                                   array_char_pt_type(*c_var_name_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_Connect = _neurongpu.NeuronGPU_Connect
NeuronGPU_Connect.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_float, ctypes.c_float)
NeuronGPU_Connect.restype = ctypes.c_int
def SingleConnect(i_source_node, i_target_node, i_port, weight, delay):
    "Connect two nodes"
    ret = NeuronGPU_Connect(ctypes.c_int(i_source_node),
                            ctypes.c_int(i_target_node),
                            ctypes.c_ubyte(i_port), ctypes.c_float(weight),
                            ctypes.c_float(delay))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_ConnSpecInit = _neurongpu.NeuronGPU_ConnSpecInit
NeuronGPU_ConnSpecInit.restype = ctypes.c_int
def ConnSpecInit():
    "Initialize connection rules specification"
    ret = NeuronGPU_ConnSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SetConnSpecParam = _neurongpu.NeuronGPU_SetConnSpecParam
NeuronGPU_SetConnSpecParam.argtypes = (c_char_p, ctypes.c_int)
NeuronGPU_SetConnSpecParam.restype = ctypes.c_int
def SetConnSpecParam(param_name, val):
    "Set connection parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NeuronGPU_SetConnSpecParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_ConnSpecIsParam = _neurongpu.NeuronGPU_ConnSpecIsParam
NeuronGPU_ConnSpecIsParam.argtypes = (c_char_p,)
NeuronGPU_ConnSpecIsParam.restype = ctypes.c_int
def ConnSpecIsParam(param_name):
    "Check name of connection parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_ConnSpecIsParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SynSpecInit = _neurongpu.NeuronGPU_SynSpecInit
NeuronGPU_SynSpecInit.restype = ctypes.c_int
def SynSpecInit():
    "Initializa synapse specification"
    ret = NeuronGPU_SynSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_SetSynSpecIntParam = _neurongpu.NeuronGPU_SetSynSpecIntParam
NeuronGPU_SetSynSpecIntParam.argtypes = (c_char_p, ctypes.c_int)
NeuronGPU_SetSynSpecIntParam.restype = ctypes.c_int
def SetSynSpecIntParam(param_name, val):
    "Set synapse int parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NeuronGPU_SetSynSpecIntParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_SetSynSpecFloatParam = _neurongpu.NeuronGPU_SetSynSpecFloatParam
NeuronGPU_SetSynSpecFloatParam.argtypes = (c_char_p, ctypes.c_float)
NeuronGPU_SetSynSpecFloatParam.restype = ctypes.c_int
def SetSynSpecFloatParam(param_name, val):
    "Set synapse float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NeuronGPU_SetSynSpecFloatParam(c_param_name, ctypes.c_float(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NeuronGPU_SetSynSpecFloatPtParam = _neurongpu.NeuronGPU_SetSynSpecFloatPtParam
NeuronGPU_SetSynSpecFloatPtParam.argtypes = (c_char_p, ctypes.c_void_p)
NeuronGPU_SetSynSpecFloatPtParam.restype = ctypes.c_int
def SetSynSpecFloatPtParam(param_name, arr):
    "Set synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    if (type(arr) is list)  | (type(arr) is tuple):
        arr = (ctypes.c_float * len(arr))(*arr) 
    arr_pt = ctypes.cast(arr, ctypes.c_void_p)    
    ret = NeuronGPU_SetSynSpecFloatPtParam(c_param_name, arr_pt)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SynSpecIsIntParam = _neurongpu.NeuronGPU_SynSpecIsIntParam
NeuronGPU_SynSpecIsIntParam.argtypes = (c_char_p,)
NeuronGPU_SynSpecIsIntParam.restype = ctypes.c_int
def SynSpecIsIntParam(param_name):
    "Check name of synapse int parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_SynSpecIsIntParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SynSpecIsFloatParam = _neurongpu.NeuronGPU_SynSpecIsFloatParam
NeuronGPU_SynSpecIsFloatParam.argtypes = (c_char_p,)
NeuronGPU_SynSpecIsFloatParam.restype = ctypes.c_int
def SynSpecIsFloatParam(param_name):
    "Check name of synapse float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_SynSpecIsFloatParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_SynSpecIsFloatPtParam = _neurongpu.NeuronGPU_SynSpecIsFloatPtParam
NeuronGPU_SynSpecIsFloatPtParam.argtypes = (c_char_p,)
NeuronGPU_SynSpecIsFloatPtParam.restype = ctypes.c_int
def SynSpecIsFloatPtParam(param_name):
    "Check name of synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NeuronGPU_SynSpecIsFloatPtParam(c_param_name) != 0)
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
    

    

NeuronGPU_ConnectSeqSeq = _neurongpu.NeuronGPU_ConnectSeqSeq
NeuronGPU_ConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int)
NeuronGPU_ConnectSeqSeq.restype = ctypes.c_int

NeuronGPU_ConnectSeqGroup = _neurongpu.NeuronGPU_ConnectSeqGroup
NeuronGPU_ConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_int)
NeuronGPU_ConnectSeqGroup.restype = ctypes.c_int

NeuronGPU_ConnectGroupSeq = _neurongpu.NeuronGPU_ConnectGroupSeq
NeuronGPU_ConnectGroupSeq.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int)
NeuronGPU_ConnectGroupSeq.restype = ctypes.c_int

NeuronGPU_ConnectGroupGroup = _neurongpu.NeuronGPU_ConnectGroupGroup
NeuronGPU_ConnectGroupGroup.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int)
NeuronGPU_ConnectGroupGroup.restype = ctypes.c_int

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
            val = syn_dict[param_name]
            if ((param_name=="synapse_group") & (type(val)==SynGroup)):
                val = val.i_syn_group
            SetSynSpecIntParam(param_name, val)
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
        ret = NeuronGPU_ConnectSeqSeq(source.i0, source.n, target.i0, target.n)
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NeuronGPU_ConnectSeqGroup(source.i0, source.n, target_arr_pt,
                                            len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NeuronGPU_ConnectGroupSeq(source_arr_pt, len(source),
                                            target.i0, target.n)
        else:
            ret = NeuronGPU_ConnectGroupGroup(source_arr_pt, len(source),
                                              target_arr_pt, len(target))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NeuronGPU_RemoteConnectSeqSeq = _neurongpu.NeuronGPU_RemoteConnectSeqSeq
NeuronGPU_RemoteConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int)
NeuronGPU_RemoteConnectSeqSeq.restype = ctypes.c_int

NeuronGPU_RemoteConnectSeqGroup = _neurongpu.NeuronGPU_RemoteConnectSeqGroup
NeuronGPU_RemoteConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int)
NeuronGPU_RemoteConnectSeqGroup.restype = ctypes.c_int

NeuronGPU_RemoteConnectGroupSeq = _neurongpu.NeuronGPU_RemoteConnectGroupSeq
NeuronGPU_RemoteConnectGroupSeq.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int)
NeuronGPU_RemoteConnectGroupSeq.restype = ctypes.c_int

NeuronGPU_RemoteConnectGroupGroup = _neurongpu.NeuronGPU_RemoteConnectGroupGroup
NeuronGPU_RemoteConnectGroupGroup.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                              ctypes.c_int, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_int)
NeuronGPU_RemoteConnectGroupGroup.restype = ctypes.c_int

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
        ret = NeuronGPU_RemoteConnectSeqSeq(i_source_host, source.i0, source.n,
                                            i_target_host, target.i0, target.n)

    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NeuronGPU_RemoteConnectSeqGroup(i_source_host, source.i0,
                                                  source.n, i_target_host,
                                                  target_arr_pt, len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NeuronGPU_RemoteConnectGroupSeq(i_source_host, source_arr_pt,
                                                  len(source),
                                                  i_target_host, target.i0,
                                                  target.n)
        else:
            ret = NeuronGPU_RemoteConnectGroupGroup(i_source_host,
                                                    source_arr_pt,
                                                    len(source),
                                                    i_target_host,
                                                    target_arr_pt,
                                                    len(target))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def SetStatus(gen_object, params, val=None):
    "Set neuron or synapse group parameters or variables using dictionaries"
    if type(gen_object)==SynGroup:
        return SetSynGroupStatus(gen_object, params, val)
    nodes = gen_object    
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
    

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

NeuronGPU_GetSeqSeqConnections = _neurongpu.NeuronGPU_GetSeqSeqConnections
NeuronGPU_GetSeqSeqConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, c_int_p)
NeuronGPU_GetSeqSeqConnections.restype = c_int_p

NeuronGPU_GetSeqGroupConnections = _neurongpu.NeuronGPU_GetSeqGroupConnections
NeuronGPU_GetSeqGroupConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                             c_void_p, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NeuronGPU_GetSeqGroupConnections.restype = c_int_p

NeuronGPU_GetGroupSeqConnections = _neurongpu.NeuronGPU_GetGroupSeqConnections
NeuronGPU_GetGroupSeqConnections.argtypes = (c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NeuronGPU_GetGroupSeqConnections.restype = c_int_p

NeuronGPU_GetGroupGroupConnections = _neurongpu.NeuronGPU_GetGroupGroupConnections
NeuronGPU_GetGroupGroupConnections.argtypes = (c_void_p, ctypes.c_int,
                                               c_void_p, ctypes.c_int,
                                               ctypes.c_int, c_int_p)
NeuronGPU_GetGroupGroupConnections.restype = c_int_p

def GetConnections(source=None, target=None, syn_group=-1): 
    "Get connections between two node groups"
    if source==None:
        source = NodeSeq(None)
    if target==None:
        target = NodeSeq(None)
    if (type(source)==int):
        source = [source]
    if (type(target)==int):
        target = [target]
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
    
    n_conn = ctypes.c_int(0)
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        conn_arr = NeuronGPU_GetSeqSeqConnections(source.i0, source.n,
                                                  target.i0, target.n,
                                                  syn_group,
                                                  ctypes.byref(n_conn))
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            conn_arr = NeuronGPU_GetSeqGroupConnections(source.i0, source.n,
                                                        target_arr_pt,
                                                        len(target),
                                                        syn_group,
                                                        ctypes.byref(n_conn))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            conn_arr = NeuronGPU_GetGroupSeqConnections(source_arr_pt,
                                                        len(source),
                                                        target.i0, target.n,
                                                        syn_group,
                                                        ctypes.byref(n_conn))
        else:
            conn_arr = NeuronGPU_GetGroupGroupConnections(source_arr_pt,
                                                          len(source),
                                                          target_arr_pt,
                                                          len(target),
                                                          syn_group,
                                                          ctypes.byref(n_conn))

    conn_list = []
    for i_conn in range(n_conn.value):
        conn_id = ConnectionId(conn_arr[i_conn*3], conn_arr[i_conn*3 + 1],
                   conn_arr[i_conn*3 + 2])
        conn_list.append(conn_id)
        
    ret = conn_list

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

 
NeuronGPU_GetConnectionStatus = _neurongpu.NeuronGPU_GetConnectionStatus
NeuronGPU_GetConnectionStatus.argtypes = (ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, c_int_p,
                                         c_char_p, c_char_p,
                                         c_float_p, c_float_p)
NeuronGPU_GetConnectionStatus.restype = ctypes.c_int

def GetConnectionStatus(conn_id):
    i_source = conn_id.i_source
    i_group = conn_id.i_group
    i_conn = conn_id.i_conn
    
    i_target = ctypes.c_int(0)
    i_port = ctypes.c_char()
    i_syn = ctypes.c_char()
    delay = ctypes.c_float(0.0)
    weight = ctypes.c_float(0.0)

    NeuronGPU_GetConnectionStatus(i_source, i_group, i_conn,
                                  ctypes.byref(i_target),
                                  ctypes.byref(i_port),
                                  ctypes.byref(i_syn),
                                  ctypes.byref(delay),
                                  ctypes.byref(weight))
    i_target = i_target.value
    i_port = ord(i_port.value)
    i_syn = ord(i_syn.value)
    delay = delay.value
    weight = weight.value
    conn_status_dict = {"source":i_source, "target":i_target, "port":i_port,
                        "syn":i_syn, "delay":delay, "weight":weight}

    return conn_status_dict


def GetStatus(gen_object, var_key=None):
    "Get neuron group, connection or synapse group status"
    if type(gen_object)==SynGroup:
        return GetSynGroupStatus(gen_object, var_key)
    
    if type(gen_object)==NodeSeq:
        gen_object = gen_object.ToList()
    if (type(gen_object)==list) | (type(gen_object)==tuple):
        status_list = []
        for gen_elem in gen_object:
            elem_dict = GetStatus(gen_elem, var_key)
            status_list.append(elem_dict)
        return status_list
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetStatus(gen_object, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        if (type(gen_object)==ConnectionId):
            status_dict = GetConnectionStatus(gen_object)
        elif (type(gen_object)==int):
            i_node = gen_object
            status_dict = {}
            name_list = GetScalVarNames(i_node) + GetScalParamNames(i_node) \
                        + GetPortVarNames(i_node) + GetPortParamNames(i_node) \
                        + GetArrayVarNames(i_node) + GetArrayParamNames(i_node)
            for var_name in name_list:
                val = GetStatus(i_node, var_name)
                status_dict[var_name] = val
        else:
            raise ValueError("Unknown object type in GetStatus")
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if (type(gen_object)==ConnectionId):
            status_dict = GetConnectionStatus(gen_object)
            return status_dict[var_key]
        elif (type(gen_object)==int):
            i_node = gen_object
            return GetNeuronStatus([i_node], var_key)[0]
        else:
            raise ValueError("Unknown object type in GetStatus")
        
    else:
        raise ValueError("Unknown key type in GetStatus", type(var_key))



NeuronGPU_CreateSynGroup = _neurongpu.NeuronGPU_CreateSynGroup
NeuronGPU_CreateSynGroup.argtypes = (c_char_p,)
NeuronGPU_CreateSynGroup.restype = ctypes.c_int
def CreateSynGroup(model_name, status_dict=None):
    "Create a synapse group"
    if (type(status_dict)==dict):
        syn_group = CreateSynGroup(model_name)
        SetStatus(syn_group, status_dict)
        return syn_group
    elif status_dict!=None:
        raise ValueError("Wrong argument in CreateSynGroup")

    c_model_name = ctypes.create_string_buffer(to_byte_str(model_name), \
                                               len(model_name)+1)
    i_syn_group = NeuronGPU_CreateSynGroup(c_model_name) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return SynGroup(i_syn_group)

  
NeuronGPU_GetSynGroupNParam = _neurongpu.NeuronGPU_GetSynGroupNParam
NeuronGPU_GetSynGroupNParam.argtypes = (ctypes.c_int,)
NeuronGPU_GetSynGroupNParam.restype = ctypes.c_int
def GetSynGroupNParam(syn_group):
    "Get number of synapse parameters for a given synapse group"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupNParam")
    i_syn_group = syn_group.i_syn_group
    
    ret = NeuronGPU_GetSynGroupNParam(ctypes.c_int(i_syn_group))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NeuronGPU_GetSynGroupParamNames = _neurongpu.NeuronGPU_GetSynGroupParamNames
NeuronGPU_GetSynGroupParamNames.argtypes = (ctypes.c_int,)
NeuronGPU_GetSynGroupParamNames.restype = ctypes.POINTER(c_char_p)
def GetSynGroupParamNames(syn_group):
    "Get list of synapse group parameter names"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParamNames")
    i_syn_group = syn_group.i_syn_group

    n_param = GetSynGroupNParam(syn_group)
    param_name_pp = ctypes.cast(NeuronGPU_GetSynGroupParamNames(
        ctypes.c_int(i_syn_group)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NeuronGPU_IsSynGroupParam = _neurongpu.NeuronGPU_IsSynGroupParam
NeuronGPU_IsSynGroupParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_IsSynGroupParam.restype = ctypes.c_int
def IsSynGroupParam(syn_group, param_name):
    "Check name of synapse group parameter"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in IsSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NeuronGPU_IsSynGroupParam(ctypes.c_int(i_syn_group), \
                                     c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    
NeuronGPU_GetSynGroupParam = _neurongpu.NeuronGPU_GetSynGroupParam
NeuronGPU_GetSynGroupParam.argtypes = (ctypes.c_int, c_char_p)
NeuronGPU_GetSynGroupParam.restype = ctypes.c_float
def GetSynGroupParam(syn_group, param_name):
    "Get synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)

    ret = NeuronGPU_GetSynGroupParam(ctypes.c_int(i_syn_group),
                                         c_param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NeuronGPU_SetSynGroupParam = _neurongpu.NeuronGPU_SetSynGroupParam
NeuronGPU_SetSynGroupParam.argtypes = (ctypes.c_int, c_char_p,
                                       ctypes.c_float)
NeuronGPU_SetSynGroupParam.restype = ctypes.c_int
def SetSynGroupParam(syn_group, param_name, val):
    "Set synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NeuronGPU_SetSynGroupParam(ctypes.c_int(i_syn_group),
                                         c_param_name, ctypes.c_float(val))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetSynGroupStatus(syn_group, var_key=None):
    "Get synapse group status"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupStatus")
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetSynGroupStatus(syn_group, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        status_dict = {}
        name_list = GetSynGroupParamNames(syn_group)
        for param_name in name_list:
            val = GetSynGroupStatus(syn_group, param_name)
            status_dict[param_name] = val
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
            return GetSynGroupParam(syn_group, var_key)        
    else:
        raise ValueError("Unknown key type in GetSynGroupStatus", type(var_key))

def SetSynGroupStatus(syn_group, params, val=None):
    "Set synapse group parameters using dictionaries"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupStatus")
    if ((type(params)==dict) & (val==None)):
        for param_name in params:
            SetSynGroupStatus(syn_group, param_name, params[param_name])
    elif (type(params)==str):
            return SetSynGroupParam(syn_group, params, val)        
    else:
        raise ValueError("Wrong argument in SetSynGroupStatus")       
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())

