# Learning rate decay

## Definition  

* slowly reduce the learning rate over time

## Reason

* afford taking larger steps during the initial steps of learning
* take smaller steps as learning approaches convergence

## Formulation

$\alpha = \frac{1}{1 + decay\_rate * epoch\_num} * \alpha_0$

## Hyperparameters

* $decay\_rate$
* $\alpha_0$

## Variations
* exponential decay: $\alpha=0.95^{epoch\_num} * \alpha_0$
* constant decay: $\alpha=\frac{k}{\sqrt(epoch\_num)} * \alpha_0$ or $\alpha=\frac{k}{\sqrt(t)} * \alpha_0$
* discrete staircase decay
