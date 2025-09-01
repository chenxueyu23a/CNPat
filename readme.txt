The llama_emb contains the 4096-dimensional llama embeddings used in the experiment. 
You can also run the following code to preprocess the original data. CNPat contains the original patent information.

(1) Generate llama embedding:

Process all patents by running demo-allpat.py, which generates embeddings with a length of 4096.

(2) Dimensionality reduction code (LLaMA2-whitening):

Run whitening-test.py, which produces embeddings with a length of 512.
