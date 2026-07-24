Value is complex orientation
Confidence is magnitude

following the [[cope]] pattern of A * B + C
gives the product of the confidences, and the modular sum of the values
Normalization rescales confidence to be a relative quantity; need to pick how this works.
How should relative confidence work?
Divide by the softmaximum to scale down? Bias towards eliminating the low confidence values.
Activation function design to control magnitude?

# RMS Norm
Is pretty good actually for this. Relative confidence only