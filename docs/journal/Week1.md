# August 19, 2026

## Getting started
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

# August 20, 2026
## Team Week 1 Selection

### Outline
Okay assuming we have set up some basic data tooling let's think through what our team selection must satisfy. We want a way to evaluate the "value" of a proposed team selection.  It is too short sighted for the value to be the week 1 expected points. This team needs to be good for as long as we have it within the transfer rules. Remember we are given one free transfer per game week and this is essential to maximizing season outcomes. We also need to do a really good job with selecting captains and vice captains (2x points). Also the bench priorities.

So we need some way to have this logic in an evaluation gym. I think something basic like make a model, simulate performance over a horizon of time probably with MCMC and then this is how we can estimate scores. From these scores we need a way to assign values. Well the value for a given week should be a function of the distribution. We will need to manage risk and return (variance and returns).

Another thing is, this is a competive game. We compare about beating other high performing players. It does not matter so much to choose a 90% percentile team if it loses against other higher performers. But we also don't want to choose outlier teams which may be likely to fail on a few tail end bets. 

### Data
Before we train anything we should understand what is the fundamental unit of data. Ideally it is a row that captures an observation/measurement. The important things for leakage are the date / timestamp. We need to be very clear about what will cause leakage and should have a validation step for this before anything. Things that test leakage with some statistical tests in addition to data level validation.

Then we want our funadmental unit of data to work for any training data preparation step. So from raw events we can then featurize, then from features we can prepare training data.

What data should we use to get the initial priors on performance? Some data depends on the curent place in season etc. So I think a simple way is to try and predict the thing you want to do. So if the task is to make a first week team, we should get really good at predicting first week team from historical and contextual data.

#### Fresh Data

It is super important that we have up-to-date information about injuries and probabilities of player as well as prices otherwise our team selections wont be possible in real life. For this reason it is essential we connect to the FPL api or any other data sources for live updates.

I think it's good instead of keeping this in the data set to keep a separate collection of filters that can be used to filter data sets on the fly based on injuries, not player, transfers, no longer in league etc.

I think it will pay off greatly to have a lot of validation tests to make sure training data is fresh with matching clubs, status, price etc.

### Model Objective
It is also worth noting that I will be in a head to head league which means I could do well on average and lose every single week. We want a team which is likely to not lose head to heads against other probable teams. We should try and see if we can do some competitive analysis on what types of teams we should beat.

### Searching Over Teams
Once I have a way to estimate the value of a team at a given week I want to be able to simulate head to heads and then we need some way for a model to learn good ways to choose a team. We ideally want to make this easier than choosing over players if that makes sense. We should have a filtering stage where we don't consider a handful players expected to never feature. I dont mind as a first approximation getting a basket of teams that we think are good and reasonable and simulating them against each other.

I think we can do an initial player scoring where we give the predictions for every player. Then only keep a certain amount of players within a certain threshold. We want to keep enough for position, team, price diversity. 

So with our subset of considered players we build a collection of potential wihtin-budget teams with them. With N players (and some over counting on positions) there are like N choose 15 teams. We can be more specific by counting over positions but we want to get the search space to a large enough number where there is complexity but small enough to be tractable.

Then once we are there we can simulate H2H across these teams to assign value to a team. This value be an aggregate so we also want to know what this team's weaknesses are. They could be black box but also like maybe its weak to fixtures or we want to see this and so on.

At some point we also have to implement the captain mechanic as well as free transfers. We only need to plan for a few weeks because there will be a point where we can change our whole team in the season.

Also what ever infra i design for this shouldnt be specific to this algorihtm but it should be a harness to support the experimentation and improvement of the above that can be reused later.

Something I expect to be an issue with this is that if we use expected points then we are averaging over time. This means that many teams will perform similarly on average and will make distinguishing betweens difficult. This means we need a way to capture variance, especially do to contextual factors. We should be at least giving an estimate of some of the moments of the distirbutions for player points. I wonder if we could use something like a t-digest or something to store estimates of cdfs for players. I think t-digest is a good way to go. We will likely want to use quantiles in many many computations.

### Using Captains and Transfer
We will also need to take into account our weekly free transfer as well as who to captain. Each of these probably needs its own discussion. I think it will be good to have seperate optimizers for these. Since candidate teams will change week to week with difference in captain, vice captain, and transfers it will probably be wise to abstract the team type and player types. And then this way we can change captains and so on and save metadata and so on.

There's some care to be taken in how to do this so it makes sense but i also think it will make downstream algorithms more readable and so on.

#### Captain


#### Transfers


### Squad Priority
I wonder how to model substitutions. Probably an easy way is to just have decaying weights in order of priority and also maybe have a simple probability of player. But these are probably same thing. what we really what is probability of playing and also probability of being in my first team and maybe even probability of being captain?

### Observability
Another important lesson from industry work is that ML projects be extremely observable across their entire lifecycle. We need some kind of way for each model to capture what its training data was, what it's experiment lifecycle was like, etc but this is a lot of data and we need to get a minimal working thing first. mainly anything needed for documenting experiments almost like a model registry or something.

## Some SWE Considerations
### Domain Modeling
As I vibe-code this 0-1 software. I am reminded of domain modeling and its importance. We did not start with a domain model but we should have. I have since added some basic entities. I think for these we should be clear about what data an entity has that is immutable and what is mutable. And also strongly type and make clear what is public or private.

Okay so what we want from our domain models is really to make the expression of programs easier to reason about. We don't actually want the computation to happen rigidily within the domain model. We want performance. So we should be able to have translations or converters into things like arrays or row-based formats from our models so we can read data to abstraction. But I think it'd be even cooler if we can just use the abstract programs to create recipes that can be applied directly to the raw data so we dont have to move data through. It's really the ease of expression of algorithms that the domain model will provide.

I guess ultimately we want something like a tensor flow where we can specify computation graphs quickly that are compatible with our domain model so all of its rigid structure is carried by the domain model and doesnt have to be reimplemented everytime. It doesn't have to reinvent the wheel just be compatible with the wheel.

We want to design in our model but execute in a well maintained computational framework (JAX, numpy, scikitlearn, etc).

Anything rigid or not subject to change in the FPL rules should be reflected in the domain model. But it also needs to have its respective implementation for the array translation and so on.

### How to specify algorithms
#### Multi Week Planning
We want our models to be able to plan and reason through the consequences of choosing a certain squad. It reminds me a bit of the AlphaGo planning situation but we dont need that example. I mean more just like a way to say okay, if we choose this squad what can happen? Like what transfer paths and captain paths are available. We need a way to incorporate this into the Squad score when comparing.

We need to set up a gym first. At this point, it is assumed our domain model is made. We should be able to taken chosen squad at a certain game week and replay how what happened in the next few weeks. This will allows us to test our baseline model predictions and how htey interact with the unlearned features of captains, transfers, and who plays. Once we have this gym it will act as our evaluation on actual squads. Later we can model probability of playing and so on and then ad captain and transfer models.

I wonder if it is work abstracting the gym to make it very rigid and explicity what it needs to run an eval. I guess its better to say what an eval is. Observability will come back here because that is what will tell us more qualitative information about our model and its performance which will allow rapid iteration.


### Model Ensembles
We will need player models for each position. These will be responsible for per player predictions (something like global model + correction per player). It is very important we account for matchup and covariance within squad. Our player models cannot assume I.I.D. because it will cause catastrophic collapse in our teams if we have destructive interference. ALso we want ot be able to capitilize on team momentum. We should really do an analysis (which our infra shoould support) on which fields are most correlated among players and then determine what we should do to account for this. We need an analysis and experiments folder where we will have our investigations document.

What we need is some kind of feature that lets a model know how to adjust. There is probably a common one that doesn't need as much player-player interaciton that can be captured in team matchup. So when we make a squad we need to take into account the covariance among teamate as well as the fact that some players are competing against eachother. Probably not a big deal for attackers but mixing attackers and defneders likely has diffferent outcome. How do we account for this in features?

# August 21, 2026
It is now only a few hours until the first match. We have some skeleton of tooling but the code has gotten a bit disorganized in the final push so I am request a refactor to stay aligned with our design goals

## On Model Performance
I think it is good break down performance into different dimensions. For example it is well known to consider ranking seperately from calibration. If we are giving player scores, its one type of performance to correctly order (rank) players by their score. But the model could be miscalibrated. But a good ranking model can still be useful for selecting a top candidate when we don't need to know exact points.

In the spiring of having observability we should choose some metrics which evaluate the ranking and calibration respectively.

## On Comparing two models
Given a fixed data set [t | X | y] with t timestamps, X features, and y label we want to standardize the evaluation setup. The timestamps are important because they establish a causal ordering on data. That means you should not use anything from the future to predict the past (leakage). The labels are used to understand the ranking and calibration. But the features shouldn't matter so much if it is clear that the function being passed can properly ingest it. Because we are comparing functions. Would it be fun to make a model evalation framework in rust specifically for FPL? But why rust? Well actually we should understand where Python introduces delay. I think it's mainly that we want 0 cost abstraction. Also I think the baseline should at least be whatever the best linear function of X to approximate y is.

This reminds me that not only our modeling should be observable but also our infa. We need to know where time time is being spent. is it on data prep, model training, inference and comparison, etc? 