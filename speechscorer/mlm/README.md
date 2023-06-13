# Intuition

Bert-like models are trained to predict a masked token from its contextual representation. If this token is highly predictible under the trained model, the entropy will be low, otherwise entropy will be high.
For the trained model, we expect the input speech to be more predictible for good english than bad english.

# More formally

Let $\textbf{x}$ be the input speech and $\textbf{u}={u_{1}, u_{2}, ..., u_{n}}$ the encoded input speech by the bert-like speech model, where each $u_{i} \in \mathbb{R}^{d}$ with $d$ the embedding size.
In order to compute the overall entropy of the whole sequence $\textbf{u}$, each vector is masked at time:
- $\textbf{u}^{1}={u_{\tiny mask}, u_{2}, u_{3}, ..., u_{n}}$
- $\textbf{u}^{2}={u_{1}, u_{\tiny mask}, u_{3}, ..., u_{n}}$
- $\textbf{u}^{3}={u_{1}, u_{2}, u_{\tiny mask}, ..., u_{n}}$
- ...
- $\textbf{u}^{n}={u_{1}, u_{2}, ..., u_{\tiny mask}}$

where $u_{\tiny mask}$ is the masked token contextual representation.

Let $\textbf{u}^{i}_{\tiny mask}$ be the masked token of the $u^{i}$ sequence.

The probability distribution over the vocabulary for $\textbf{u}^{i}_{\tiny mask}$ masked token is:

```math
X^{\textbf{u}^{i}_{\tiny mask}} = P(\cdot|\textbf{u}^{i}_{\tiny mask})
```

where $X^{\textbf{u}^{i}_{\tiny mask}} \in \mathbb{R}^V$ is a vector of probability over the vocabulary $V$.

The entropy for the $\textbf{u}^{i}_{\tiny mask}$ is defined as:

```math
H(X^{\textbf{u}^{i}_{\tiny mask}}) = \sum\limits_{p\in X^{\textbf{u}^{i}_{\tiny mask}}} p \times log\;p
```

The overall entropy of the input speech can be compute as:

```math
H(\textbf{x}) = \frac{1}{n} \sum\limits_{i=1}^{n} H(X^{\textbf{u}^{i}_{\tiny mask}})
```

