I would like the code to be designed with very clear architechtures. I prefer functional paradigms in as much as possible favoring composability when possible. Make sure to make the interfaces clear.

We also want to have good black box tests which do not depend on implementations. Commit messages should follow conventional commits.

There should be clear documentation for any PRs and features and so on

Use YAML for config so it's language agnostic. Use protobufs when needed. In as much as possible we want to separate config and code and deployment stuff.

I think abstraction can be good for making easy to use interfaces but we need to be aware of when abstraction will accrue complexity overhead. In as much as possible we don't want costly abstracts. Both for runtime but also for if an abstraction needs to be changed then the cost of rewriting it by how many lines need to be changed for a feature or extension etc. 

We should design pieces that depend on each other to be as agnostic to internal implementations as possible. That is, model training should not depend on how we implemented dat prep to much. Model training should work for "a" method of data prep. This keeps things flexible and reduces the time for performance optimizations to be implemented.