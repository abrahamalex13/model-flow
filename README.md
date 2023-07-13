# ModelFlow

Develop predictive models with one configurable workflow. 
The workflow configuration should be understandable to _anyone_ interested.

## Entrypoints

Pass raw data through feature-transforms pipeline: `python -m src.features.main`

With transformed data, estimate models, using modules in `src.models`

## Configuration Files

**_Configure_ the data sample.**
Data are real, while models are intangible abstractions.
- From what (storage) source do we extract data?
- What filters subset the training data?
- What filters subset new test data?
- What outcome do we seek to predict? 
How should its missing values be interpreted?
- What data columns constitute subject _attributes_ -- 
column-concatenated to predictions, for 
subject identifiers and exploratory analysis?

**_Configure_ the feature transforms pipeline, 
which yields model-ready inputs from raw values.**
After declaring preset transformers, 
proceed by feature, declaring each one's transforms. 
With feature-then-transforms flow, intent is analyst-friendly thought process.
(Under the hood, details convert to transform-then-features structure, 
according to transformer APIs.)
- Uniquely name the pipeline. Arbitrarily many could transform a dataset.
- Name transform functions (transformers) with default arguments.
- Declare model features. For each,
    - What data type?
    - Which transforms apply?

**_Configure_ subdirectories which house feature transform pipeline artifacts.
Expect several models will co-exist.**
The subdirectory structure must partition:
- Two pipelines predicting different outcomes
- Two pipelines predicting same outcome, but trained from different samples
- Two pipelines predicting same outcome & trained from same samples, but
following different transform procedures
