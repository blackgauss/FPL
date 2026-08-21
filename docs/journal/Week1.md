## August 19, 2026

I have decided to play! Now I need to bootstrap a team selection for week 1.
I want my season to an ML project where I get to build an ML system that can do well
at FPL. 

But I need something quick. Here are some guiding principles I'd like to set though

- Rapid Experimentation Loop
-- This means need to be able to quickly test new ideas and see their performance.
- Well documented
- Good SWE hygiene
- Should build towards a professional ML system.

For languages I want to use Python ML suite and then JAX for computing in python. Then I also want to look into
building things with rust. I am not sure what yet but we will look into it.

The computers we have available are an Nvidia DGX spark, and some consumer
lenovo thinkpads with 8 to 16gb of ram.

I think a place i'd like to start is by setting up some config files. 

I want to use 
- https://github.com/olbauday/FPL-Core-Insights 
- https://github.com/vaastav/Fantasy-Premier-League
- FPL API

for data and we should set up how to access these in config or download them or something. Well for historical data we can download once.
We need to git ignore where we will store data. Also I want the real ingestion to take in filepaths not hardcoded.

So we should go look into those sources and document schema and so on.

Then we need to build a way to rapidly do queries on this data and I think polars would be a good framework for this.
It should be able to handle stuff well. I want to use it for data prep and creating training data.

Then we need a pipeline that takes prepared training data and outputs model parameters that can be deployed into inference for evaluation or scoring.


