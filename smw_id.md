# Sherman-Morrison-Woodbury identity


## A special case
A special case of the Sherman-Morrison-Woodbury identity is the following

$$
    (I + UV^T)^{-1} = I - U(I + V^TU)^{-1}V^T.
$$

The derivation is simple. Assume the inverse have the form $I + UZV^T$, then we have that

\begin{align}
    I 
    &= (I + UV^T)(I + UZV^T)\\
    & = I + (UV^T + UZV^T + UV^TUZV^T)\\
    &= I + U(I + Z + V^TUZ)V^T.
\end{align}

From the fact the most right expression needs to be equal to $I$ we must have that

$$
    Z = (I + V^TU)^{-1}(-I) = -(I + V^TU)^{-1}.
$$

Which confirms the identity.

## The full SMW identity
In the general case the Sherman-Morrison-Woodbury identity have the form

$$
(A + USV^T)^{-1} = A^{-1} - A^{-1}U(S^{-1} + V^TA^{-1}U)^{-1}V^TA^{-1}.
$$

The derivation here is to set $B = A + USV^T$ so that $B^{-1} = (I + A^{-1}USV^T)^{-1}A^{-1}$. Now set $W = A^{-1}U$ and $Z = SV$ and use the special from the previous section.  

An alternatively one can a similar approach to the previous section with a guess on the inverse have the form $A^{-1} + A^{-1}UZV^TA^{-1}$ and then compute what $Z$ makes that the inverse.

## Code

```julia:./code/woodbury
using SymEGRSSMatrices
@show randn(2)
```

\output{./code/woodbury}
