# Small rooms classifier data

~1.5k manually labeled room impulse responses from the RealRIRs datasets. Can be used for training a "small rooms" classifier. Format is as follows:

```
[
  [
	  dataset: Name of dataset that contains the IR,
	  ir_name: Name of the IR that has been labeled (as used with __getitem__),
	  channel: Channel of the IR that has been labeled,
	  is_small_room: Does the IR sound like a "small room"? (Not like a church, etc.), 
  ],
  ...
]
```

Note that all of the labeling was done by me, so will be highly subjective.