# Intuition

Bert-like models are trained to predict a masked token from its contextual representation. If this token is highly predictible under the trained model, the entropy will be low, otherwise entropy will be high.
For the trained model, we expect the input speech to be more predictible for good english than bad english.

# More formally

Let $\textbf{x}$ be the input speech and $\textbf{u}={u_{1}, u_{2}, ..., u_{n}}$ the encoded input speech by the bert-like speech model, where each $u_{i} \in \mathbb{R}^{d}$ with $d$ the embedding size.
In order to compute the overall entropy of the whole sequence $\textbf{u}$, each vector is masked at time:
- $\textbf{u}^{1}={u_{\tiny MASK}, u_{2}, u_{3}, ..., u_{n}}$
- $\textbf{u}^{2}={u_{1}, u_{\tiny MASK}, u_{3}, ..., u_{n}}$
- $\textbf{u}^{3}={u_{1}, u_{2}, u_{\tiny MASK}, ..., u_{n}}$
- ...
- $\textbf{u}^{n}={u_{1}, u_{2}, ..., u_{\tiny MASK}}$

where $u_{\tiny MASK}$ is the masked token contextual representation.

The probability distribution over the vocabulary for the $u_{\tiny MASK}$ of the $\textbf{u}^{i}$ sequence can be computed as:

```math
X^{\tiny mask}_{\textbf{u}^{i}} = \frac{u_{\tiny MASK}\; \cdot W}{\sum u_{\tiny MASK}\; \cdot W}
```

where $W \in \mathbb{R}^{d*V}$, with $V$ the vocabulary size.

The entropy for the $u_{\tiny MASK}$ is defined as:

```math
X^{\tiny mask}_{\textbf{u}^{i}} = \sum\limits_{p\in X^{\tiny mask}} p \times log\;p
```

The overall entropy of the input speech can be compute as:

```math
H(\textbf{x}) = \frac{1}{n} \sum\limits_{i=1}^{n} H(X^{\tiny mask}_{\textbf{u}^{i}})
```

