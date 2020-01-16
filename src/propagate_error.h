#ifndef PROPAGATEERRORH
#define PROPAGATEERRORH

#define BEGIN_ERR_PROP			   \
  checkNeuralGPUInstance();		   \
  NeuralGPU_instance->SetErrorFlag(false); \
  NeuralGPU_instance->SetErrorMessage(""); \
  NeuralGPU_instance->SetErrorCode(0);	   \
  try

#define END_ERR_PROP							 \
  catch (ngpu_exception &e){                                             \
    NeuralGPU_instance->SetErrorFlag(true);				 \
    NeuralGPU_instance->SetErrorMessage(e.what());			 \
    NeuralGPU_instance->SetErrorCode(2);				 \
  }								         \
  catch (bad_alloc&) {                                                   \
    NeuralGPU_instance->SetErrorFlag(true);			         \
    NeuralGPU_instance->SetErrorMessage("Memory allocation error.");     \
    NeuralGPU_instance->SetErrorCode(1);			         \
  }									 \
  catch (...) {                                                          \
    NeuralGPU_instance->SetErrorFlag(true);				 \
    NeuralGPU_instance->SetErrorMessage("Error in NeuralGPU function."); \
    NeuralGPU_instance->SetErrorCode(255);				 \
  }                                                                      \
  if (NeuralGPU_instance->OnException() == ON_EXCEPTION_EXIT) {          \
    std::cerr << NeuralGPU_instance->GetErrorMessage();                  \
    exit(NeuralGPU_instance->GetErrorCode());                            \
  }

#endif
