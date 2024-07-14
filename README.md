# SPDTensor

In PyTorch, DTensor typically requires creating multiple processes that run the same code concurrently. However, in some cases, you might want to avoid starting multiple processes in your main code and instead use this feature directly, similar to Jax Sharding.