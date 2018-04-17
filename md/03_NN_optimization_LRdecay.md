# Learning rate decay

## Definition  

* slowly reduce the learning rate over time

## Why

* afford taking larger steps during the initial steps of learning
* take smaller steps as learning approaches convergence

## Formulation

$\alpha = \frac{1}{1 + decayrate \ \ast \ epochnum} \ \ast \ \alpha_0$

## Hyperparameters

* $decayrate$
* $\alpha_0$

## Variations
* exponential decay

    $\alpha=0.95^{\ epochnum\ }\ \ast \ \alpha_0$

* constant decay

    $\alpha=\frac{k}{\sqrt{epochnum}} \ \ast \ \alpha_0$

    $\alpha=\frac{k}{\sqrt{t}} \ \ast \ \alpha_0$

* discrete staircase decay
