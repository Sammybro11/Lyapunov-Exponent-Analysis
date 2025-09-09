# Analysing Chaos Using Lyapunov Exponents

This project is an analysis on different kinds of 
chaotic systems and analysis on the chaotic nature 
observed in them using Lyapunov Exponent.

### Implementation
**Installation**
```
git clone https://github.com/Sammybro11/Chaos-using-Lyapunov 
cd Chaos-using-Lyapunov
```
**Python Virtual Environment**
```
python3 -m venv env
source env/bin/activate
```
**Installing required Packages**
```
pip install -r requirements.txt
```

## Physics and Analysis

### Hénon Maps

We start with a relatively simple example called a
[Hénon Map](https://en.wikipedia.org/wiki/H%C3%A9non_map). 
The Hénon map is a discrete-time dynamical system. It is one of the most studied examples of dynamical systems that exhibit chaotic behavior. 

The Hénon map takes a point ($x_n$, $y_n$) in the plane and maps it to a new point:
```math
\begin{cases}
x_{n+1} &= 1 - ax^2_n + y_n \\
y_n+1 &= bx_n
\end{cases}
```



