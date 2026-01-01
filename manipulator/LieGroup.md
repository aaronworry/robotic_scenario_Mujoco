## BCH approximating
$$
\exp{(\Delta \phi^{\wedge})}\exp{(\phi^{\wedge})} \approx \exp{((J_l^{-1}(\phi)\Delta \phi + \phi)^{\wedge})} \\
\exp{(\phi^{\wedge})}\exp{(\Delta \phi^{\wedge})} \approx \exp{((J_r^{-1}(\phi)\Delta \phi + \phi)^{\wedge})}
$$
where $J_r(\phi) = J_l(-\phi)$.

## Derivative of Exp3
$$
J_r(r) = 
\frac{\sin{||r||}}{||r||}           I_3
- \frac{1-\cos{||r||}}{||r||^2}     \left[ r \right]_{\times}
+ \frac{1}{||r||^2} (1-\frac{\sin{||r||}}{||r||}) r r^T
$$
where $r = \theta a$ describe the rotation axis, and $a$ is an unit vector 

## Derivative of Log3

This function is the right derivative of $log3$, that is, for $R \in SO(3)$ and $\textbf{x} \in \mathfrak{so}(3)$, it provides the linear approximation:

$$
\log_3(R \oplus \textbf{x}) = \log_3(R \exp_3(\textbf{x})) \approx \log_3(R) + \text{Jlog3}(R) \textbf{x}
$$

Equivalently, $\text{Jlog3}$ is the right Jacobian of $\log_3$:
$$
\text{Jlog3}(R) = \frac{\partial \log_3(R \oplus \textbf{x})}{\partial \textbf{x}} |_{\textbf{x}=\textbf{0}} = \frac{\partial \log_3(R \exp{\textbf{x}^{\wedge}})}{\partial \textbf{x}} |_{\textbf{x}=\textbf{0}} = J_r^{-1}|_{3\times 3}
$$

Note that this is the right Jacobian: $\text{Jlog3}(R) : T_{R} SO(3) \to T_{\log_6(R)} \mathfrak{so}(3)$. 
(By convention, calculations in Pinocchio always perform right differentiation, i.e., Jacobians are in local coordinates (also known as body coordinates), unless otherwise specified.)

If we denote by $\theta = \log_3(R)$ and $\log = \log_3(R, \theta)$, then $\text{Jlog} = \text{Jlog}_3(R)$ can be calculated as:

$$
\begin{array}{ll}
\text{Jlog} & = \frac{\theta \sin(\theta)}{2 (1 - \cos(\theta))} I_3
           + \frac{1}{2} \widehat{\log}
           + \left(\frac{1}{\theta^2} - \frac{\sin(\theta)}{2\theta(1-\cos(\theta))}\right) \log \log^T \\ 
& = I_3 + \frac{1}{2} \widehat{\log} +  \left(\frac{1}{\theta^2} - \frac{1 + \cos \theta}{2 \theta \sin \theta}\right){\widehat{\log}}^2 \\
\end{array}
$$
where $\widehat{v}$ denotes the skew-symmetric matrix obtained from the 3D vector $v$.
The inputs must be such that $\theta = \Vert \log \Vert$.

\param[in] theta the angle value. 
\param[in] log the output of log3.
\param[out] Jlog the jacobian

## Derivative of Log6

This function is the right derivative of log6, that is, for $M \in SE(3)$ and $\xi \in \mathfrak{se}(3)$, it provides the linear approximation:

$$
\log_6(M \oplus \xi) = \log_6(M \exp_6(\xi)) \approx \log_6(M) + \text{Jlog6}(M) \xi
$$

Equivalently, $\text{Jlog6}$ is the right Jacobian of $\log_6$:

$$
\text{Jlog6}(M) = \frac{\partial \log_6(M \oplus \xi)}{\partial \xi} |_{\xi=\textbf{0}} = \frac{\partial \log_6(M \exp{\xi^{\wedge}})}{\partial \xi} |_{\xi=\textbf{0}}
$$

Note that this is the right Jacobian: $\text{Jlog6}(M) : T_{M} SE(3) \to T_{\log_6(M)} \mathfrak{se}(3)$.
(By convention, calculations in Pinocchio always perform right differentiation, i.e., Jacobians are in local coordinates (also known as body coordinates), unless otherwise specified.)

Internally, it is calculated using the following formulas:

$$
\text{Jlog6}(M) =
\left(\begin{array}{cc}
\text{Jlog3}(R) & J * \text{Jlog3}(R) \\
           0     &     \text{Jlog3}(R) \\
\end{array}\right)
$$

where

$$
M =
\left(\begin{array}{cc}
\exp(\mathbf{r}) & \mathbf{p} \\
            0     & 1          \\
\end{array}\right)
$$

$$
J=\frac{1}{2}[\mathbf{p}]_{\times} + \beta'(||r||)\frac{\mathbf{r}^T\mathbf{p}}{||r||}\mathbf{r}\mathbf{r}^T - (||r||\beta'(||r||) + 2 \beta(||r||)) \mathbf{p}\mathbf{r}^T + \mathbf{r}^T\mathbf{p}\beta(||r||)I_3 + \beta(||r||)\mathbf{r}\mathbf{p}^T
$$

and

$$
\beta(x)=\left(\frac{1}{x^2} - \frac{\sin x}{2x(1-\cos x)}\right)
$$

For $(A,B) \in SE(3)^2$, let $M_1(A, B) = A B$ and $m_1 = \log_6(M_1)$. Then, we have the following partial (right) Jacobians:

- $ \frac{\partial m_1}{\partial A} = \text{Jlog6}(M_1) Ad_B^{-1} $
- $ \frac{\partial m_1}{\partial B} = 	\text{Jlog6}(M_1) $

Let $A \in SE(3)$, $M_2(A) = A^{-1}$ and $m_2 =\log_6(M_2)$. Then, we have the following partial (right) Jacobian:

- $ \frac{\partial m_2}{\partial A} = - \text{Jlog6}(M_2) Ad_A $
