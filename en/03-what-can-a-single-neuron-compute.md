# What Can a Single Neuron Compute?

Before analyzing deep neural networks containing dozens of interconnected layers and millions of parameters, we must zoom in on an individual neuron. Exploring what a single neuron computes reveals the foundational types of models it can express, the geometry of its decision spaces, and the mathematical framework required to train it.

---

## 1. The Anatomy and Computation of a Single Neuron

A neural network consists of layers of interconnected computational nodes. When isolated, every individual neuron executes a two-stage computation to map input values to a single numerical output:

1. **Linear Weighted Summation:** Every incoming input $x_i$ is multiplied by its dedicated connection weight $w_i$. These products are summed together along with an additive **bias** term $b$.
2. **Activation Mapping:** The aggregated scalar sum is passed through an **activation function** $f(\cdot)$ to generate the final scalar output $y$.

![Gemini visual 1](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-01.svg)

### The Role of the Bias ($b$)
The bias term acts as an intrinsic intercept. If all input variables are zero ($x_1 = x_2 = \dots = x_n = 0$), the weighted sum $\sum w_i x_i$ collapses to $0$. Without a bias, the neuron would be strictly constrained to pass through the origin. Adding $b$ permits the neuron to evaluate a non-zero value at the origin:
$$\text{Input Sum} = b + \sum_{i=1}^n w_i x_i$$
Mathematically, the bias operates like a weight attached to a constant input of $1$, which is why it is depicted as a distinct incoming parameter arrow into the processing unit.

---

## 2. The Regression Neuron: Linear Activation

When the objective is **regression** (predicting a continuous scalar output), the neuron applies a **linear activation function**:

![Gemini visual 2](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-02.svg)

### Why Use the Simple Identity Function $y = x$?
Instead of choosing an arbitrary linear transformation $f(x) = mx + c$, deep learning uses the identity activation $y = x$. The parameters $w_1$ and $b$ already provide complete mathematical freedom to represent any slope and any vertical intercept:
$$y = m(w_1 x_1 + b) + c = (m w_1) x_1 + (m b + c) = \tilde{w}_1 x_1 + \tilde{b}$$
Introducing extra scaling or shifting parameters inside the activation function adds redundancy without increasing expressive power.

### Multidimensional Input Generalization
If the neuron receives $n$ independent inputs $(x_1, x_2, \dots, x_n)$ with a linear activation:
$$y = b + \sum_{i=1}^n w_i x_i = b + \vec{w} \cdot \vec{x}$$
The neuron evaluates an $n$-dimensional linear mapping producing a continuous 1D scalar output. In two dimensions ($n=2$), the function represents a flat tilted plane $y = w_1 x_1 + w_2 x_2 + b$; in higher dimensions ($n \ge 3$), it defines a linear hyperplane over the input feature space.

---

## 3. The Classification Neuron: Step Activation & Decision Boundaries

When the task is **classification** (assigning inputs to discrete categories such as class $0$ vs. class $1$), the linear sum is routed through a discontinuous **Heaviside step activation function**:

$$x = b + \sum_{i=1}^n w_i x_i, \quad y = \begin{cases} 0 & \text{if } x \le 0 \\ 1 & \text{if } x > 0 \end{cases}$$

Consider a neuron processing two inputs ($x_1, x_2$) with parameters $w_1 = 2$, $w_2 = -1$, and $b = 1$:

![Gemini visual 3](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-03.svg)

### The Geometry of the Decision Boundary
The boundary separating regions where the neuron outputs $0$ from where it outputs $1$ occurs precisely where the activation argument equals zero:
$$x_1 w_1 + x_2 w_2 + b = 0$$
Solving for $x_2$ yields the explicit linear equation of the boundary line in the $(x_1, x_2)$ coordinate plane:
$$x_2 = \left( -\frac{w_1}{w_2} \right) x_1 + \left( -\frac{b}{w_2} \right)$$
For our concrete parameters ($w_1 = 2, w_2 = -1, b = 1$):
$$2x_1 - x_2 + 1 = 0 \implies x_2 = 2x_1 + 1$$
Every point lying on one side of this line generates an internal sum $x \le 0$ resulting in label $0$, while every point on the opposing side produces $x > 0$ resulting in label $1$.

---

## 4. Dimensional Progression of Decision Boundaries

The geometric nature of the classification decision boundary scales systematically with the dimensionality of the input vector $\vec{x}$:

![Gemini visual 4](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-04.svg)

* **1-Dimensional Inputs ($x_1$):** The feature space is a 1D line; the decision boundary is a single **point** $x_1 = -\frac{b}{w_1}$.
* **2-Dimensional Inputs ($x_1, x_2$):** The feature space is a 2D plane; the boundary is a **line** $w_1 x_1 + w_2 x_2 + b = 0$.
* **3-Dimensional Inputs ($x_1, x_2, x_3$):** The feature space is a 3D volume; the boundary is a **plane** $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$.
* **$n$-Dimensional Inputs ($\vec{x} \in \mathbb{R}^n$):** The boundary is an **$(n-1)$-dimensional flat affine hyperplane** that bisects $n$-dimensional space into two half-spaces.

---

## 5. How to Train a Neuron: Data, Model Selection & Loss Functions

### Supervised Datasets
Supervised machine learning operates on a dataset composed of empirical input-output examples:
$$\mathcal{D} = \big\{ (\vec{x}^{(1)}, y^{(1)}), \; (\vec{x}^{(2)}, y^{(2)}), \; \dots, \; (\vec{x}^{(N)}, y^{(N)}) \big\}$$
where each example consists of an input feature vector $\vec{x}$ and the corresponding ground-truth target output $y$.

### Model Topology Determination Rules
The physical format and dimensional structure of the dataset dictate the neuron's configuration:

![Gemini visual 5](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-05.svg)

### The Loss Function: Quantifying Model Error
Training is the search for parameter values $(w_1, w_2, \dots, w_n, b)$ that make the neuron's predictions match the training data as closely as possible.

To evaluate how poorly the model performs with its current parameters, we establish an objective **Loss Function** $L(w_1, \dots, w_n, b)$:
* In **Regression**, the error for each point is the vertical distance (residual) between the predicted value and the ground truth:
  $$\text{Residual} = y^{(i)} - \text{model}(\vec{x}^{(i)})$$
* In **Classification**, the model incurs error whenever a sample point falls on the incorrect side of the decision boundary.

![Gemini visual 6](../assets/images/gemini-videos-01-03/inline-svg/lesson-03-06.svg)

By computing the squared difference $(y - \text{model}(\vec{x}))^2$ for every training sample and summing these penalties, the loss function compresses the model's global error across the entire dataset into a single scalar value. 

The central goal of deep learning is to iteratively adjust the weights and bias in the direction that lowers this loss value—a procedure executed through **Gradient Descent**.

---

## 6. Summary Comparison

| Attribute | Regression Neuron | Classification Neuron |
| :--- | :--- | :--- |
| **Activation Function** | Linear Identity ($y = x$) | Step Function ($0$ if $x \le 0$, $1$ if $x > 0$) |
| **Output Type** | Continuous real number ($y \in \mathbb{R}$) | Discrete binary category ($y \in \{0, 1\}$) |
| **Output Geometry** | Continuous slope / response surface | Discrete partitioned regions ($y = 0$ vs. $y = 1$) |
| **Boundary Structure** | None (evaluates a continuous plane) | $(n-1)$-dimensional hyperplane ($b + \sum w_i x_i = 0$) |
| **Trainable Parameters** | $n$ weights $+ 1$ bias ($n + 1$ total) | $n$ weights $+ 1$ bias ($n + 1$ total) |
| **Error Formulation** | Distance from point to line/surface ($y - \hat{y}$) | Misclassified points placed in the wrong region |
| **Optimization Goal** | Minimize Sum of Squared Errors: $\sum (y - \hat{y})^2$ | Minimize classification loss / boundary violations |
