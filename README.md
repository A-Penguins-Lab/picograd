## PicoGrad  
A minimal implementation of PyTorch's autograd with full GPU compatibility 

PicoGrad is inspired from tinygrad and micrograd. I am building a scaled down version of tinygrad and autograd annd a scaled up version of micrograd to support GPUs and multidimensional tensors. 

It is a lot of fun, the primary objective behind this project is to document all my findings on building with GPUs and post in a blog. 

Once fully built, picograd will have: 
1. A better name
2. Support for GPUs using triton/CUDA (yet to be researched)
3. autograd like functionality

## Is there anything that sets this apart? 
Absolutely not, and its always best to use pytorch or tinygrad if we want reliable functionality. This is a toy project that is purely meant for learning purposes. 

## To-Do List  
- [x] Set up the project structure  
- [ ] Set up a blog  
- [ ] Build the Tensor object  
- [ ] Write unit tests  
