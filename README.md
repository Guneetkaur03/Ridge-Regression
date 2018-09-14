# Ridge-Regression
It is popular Regularization Regression Technique
>Learns weights and bias using Ordinary Least Sqaure criterion but adds a penalty for large variations in W parameters.
>Once the parameters are learned, the Ridge Regression prediction formula is the same as ordinary least sqaures.

This Regression technique is used to create parsimonious models with large number of features. Usually used in two scenarios
- When number of predictor variables exceeds number of observations.
- When there is correlation between predictor variables (Multicollinearity)

###
Additional term lamda * sum of squares of coeffiecients(parameters) is added to cost function and aim is to minimize the value of parameters close to zero. Lamda here is the regularization parameter.

> There is a tradeoff between bias and variance. Bias and variance depends on the value of lamda. when lamda is high, sum of sqaures of
> coefficient increases, coefficients tends towards zero as a result of which hypothesis is a straight line and there are chances of 
> underfitting. 
