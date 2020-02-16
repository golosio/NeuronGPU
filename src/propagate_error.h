#ifndef PROPAGATEERRORH
#define PROPAGATEERRORH

#define BEGIN_ERR_PROP			   \
  checkNeuronGPUInstance();		   \
  NeuronGPU_instance->SetErrorFlag(false); \
  NeuronGPU_instance->SetErrorMessage(""); \
  NeuronGPU_instance->SetErrorCode(0);	   \
  try

#define END_ERR_PROP							 \
  catch (ngpu_exception &e){                                             \
    NeuronGPU_instance->SetErrorFlag(true);				 \
    NeuronGPU_instance->SetErrorMessage(e.what());			 \
    NeuronGPU_instance->SetErrorCode(2);				 \
  }								         \
  catch (std::bad_alloc&) {						 \
    NeuronGPU_instance->SetErrorFlag(true);			         \
    NeuronGPU_instance->SetErrorMessage("Memory allocation error.");     \
    NeuronGPU_instance->SetErrorCode(1);			         \
  }									 \
  catch (...) {                                                          \
    NeuronGPU_instance->SetErrorFlag(true);				 \
    NeuronGPU_instance->SetErrorMessage("Error in NeuronGPU function."); \
    NeuronGPU_instance->SetErrorCode(255);				 \
  }                                                                      \
  if (NeuronGPU_instance->OnException() == ON_EXCEPTION_EXIT) {          \
    std::cerr << NeuronGPU_instance->GetErrorMessage();                  \
    exit(NeuronGPU_instance->GetErrorCode());                            \
  }

#endif
