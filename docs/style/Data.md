I need to get my thoughts down on how data will be read and written in this project. I think the broad idea will be to equip a high powered query engine such as polars and then to save files in parquet.

We will only make contracts for specific jobs but I don't think we should make abstract Python interfaces for queries like a get player method. In as much as possible I think things should be close to the files and python just an orchastrator because of performance. 

So what will downstream tasks be like?

Well we will need to support two main kinds of tasks.

(1) Exploratory Data Analysis 
- This is for when we are gathering information during project development.

(2) Production Jobs
- Things like data transformations in order to make feature stores
- Things like preparing training data from feature stores to feed into model training

I hate to add a time dependent reference here but the current work does not really help much in this direction.

We need to do a full analysis from the POV mentioned above.