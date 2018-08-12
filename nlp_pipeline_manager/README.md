# NLP Pipe Manager 

**nlpipe.py**

This file contains a parent class for file I/O to disk. This allows any pipeline using this as the parent class to 
save all attributes to disk in a pickle file for later use.

**nlp_preprocessor.py**

This is an implementation of the main pipeline. It allows a user to put in bits and pieces, which it then chains
together so the user only needs to call `fit` and `transform` to get vectorized NLP data.

**pipeline_demo.ipynb**

A notebook that shows the pipeline in action.

**save_pipeline.mdl**

A saved model file from the pipeline_demo. This demonstrates the I/O capabilites of the pipeline

**supervised_nlp.py**

This file implements a class for using the nlp preprocessor along with a model to do prediction (classification
or regression). It assumes that the user will be providing an SkLearn model to work with. Simplifies the user API
to just `fit`, `predict`, and `score`.

**topic_modeling_nlp.py**

This file implements a class for doing topic modeling with the nlp_preprocess. The user must provide a 
topic modeling method like `TruncatedSVD` or `LatentDirichletAllocation` from SkLearn. This simplifies the user API
to just `fit`, `transform`, and `print_topics`.
