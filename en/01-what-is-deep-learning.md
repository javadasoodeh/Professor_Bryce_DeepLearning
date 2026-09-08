# Deep Learning: Foundations and Core Principles

**Source:** [What is Deep Learning? (DL 01)](https://www.youtube.com/watch?v=DrhJLHiia7g)

Deep learning powers modern technologies such as facial recognition for device unlocking, automated speech understanding in virtual assistants, and personalized recommendation systems. 

**Definition:** **Deep learning** is the application of **neural networks** and **differentiable programming** to perform **machine learning**.

---

## 1. What is Machine Learning?

Machine learning (ML) exists at the intersection of **Artificial Intelligence (AI)** and **Data Science**:
* **Data Science:** The practice of organizing, analyzing, and extracting utility from data.
* **Artificial Intelligence:** Solving computational problems that normally require human intelligence.
* **Machine Learning:** Automatically making intelligent inferences directly from data.

![Gemini visual 1](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-01.svg)

### Paradigm Shift from Classical Software Engineering
* **Classical Computer Science:** The engineer manually writes explicit algorithmic logic to compute outputs from given inputs.
* **Machine Learning:** Data examples define both the inputs and expected outputs. The machine learning algorithm automatically infers the underlying functional mapping between them.

---

## 2. Core Problem Types in Machine Learning

Most machine learning tasks fall into two fundamental categories:

1. **Regression:** Inferring a function $f(x) = y$ mapping continuous inputs to continuous outputs. In linear regression, this corresponds to determining the line of best fit through data points.
2. **Classification:** Assigning a discrete category label to each input. In binary classification, this corresponds to establishing a decision boundary separating inputs labeled $0$ from inputs labeled $1$.

![Gemini visual 2](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-02.svg)

### Combined Example: Object Recognition
Complex problems frequently combine both regression and classification. In object recognition:
* **Input:** Raw image represented as a 2D grid of pixel intensities.
* **Regression Component:** Predicting continuous spatial coordinates $(x, y, \text{width}, \text{height})$ for the bounding box around each detected item.
* **Classification Component:** Assigning a discrete categorical label (e.g., `"orange"`, `"apple"`, `"basket"`) to each detected object.

![Gemini visual 3](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-03.svg)

---

## 3. Biological Inspiration to Artificial Neural Networks

Artificial neural networks draw conceptual inspiration from biological neurons in nervous systems:
* **Biological Neurons:** Cells receive incoming signals across dendrites. If the aggregate excitation is sufficient, the cell fires an action potential down its axon, releasing neurotransmitters across synapses to neighboring neurons. Connections can be excitatory (encouraging firing) or inhibitory (suppressing firing).
* **Artificial Neural Networks:** Nodes represent individual neurons, and directed edges represent synaptic connections. Every edge has a numerical **weight** $w$:
  * **Positive Weight ($w > 0$):** Excitatory connection.
  * **Negative Weight ($w < 0$):** Inhibitory connection.
  * **Magnitude ($|w|$):** Strength of the connection.

![Gemini visual 4](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-04.svg)

---

## 4. Single Artificial Neuron Computation

A single artificial neuron computes a weighted sum of its incoming inputs and evaluates whether this sum meets an internal activation threshold:

![Gemini visual 5](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-05.svg)

### Worked Calculation:
1. **Weighted Summation:**
   $$\text{Input Sum} = (1 \times 2) + (0 \times 1) + (1 \times -1) = 2 + 0 - 1 = 1$$
2. **Threshold Evaluation:**
   $$\text{Threshold} = 0.5$$
   Because $1 \ge 0.5$, the condition is met and the neuron emits an activation output of **$1$**.

### Divergence from Neuroscience
Deep learning departs strictly from biological realism. While neurobiology explores the biochemical complexities of synaptic plasticity, deep learning treats the artificial neuron as a pragmatic mathematical mechanism optimized for functional data approximation.

---

## 5. Training via Gradient Descent & The Activation Problem

To train an artificial neural network on labeled datasets, parameters (weights) are iteratively adjusted using **Gradient Descent**:
* From vector calculus, the gradient ($\nabla$) of a function points in the direction of **steepest increase**.
* Moving in the direction opposite to the gradient ($-\nabla$) identifies the path of **steepest decrease**, enabling the minimization of a loss/error function.

![Gemini visual 6](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-06.svg)

### The Limitation of Step Functions
A thresholded neuron functions as a discontinuous step function:
$$\text{Output}(x) = \begin{cases} 0 & \text{if } x < 0.5 \\ 1 & \text{if } x \ge 0.5 \end{cases}$$

The derivative of this step function is uninformative for optimization:
* Everywhere $x \ne 0.5$, the slope is flat: $\frac{d}{dx}\text{Output}(x) = 0$.
* At the threshold boundary ($x = 0.5$), the function is discontinuous and the derivative is undefined.
* With zero derivatives everywhere, gradient descent cannot determine which direction to adjust weights.

### The Solution: Smooth Sigmoid Activation
To enable gradient-based training, the step function is replaced by a continuous, differentiable S-shaped curve known as the **Sigmoid function**:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

![Gemini visual 7](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-07.svg)

Because the sigmoid activation is smooth and continuous, non-zero derivatives can be propagated across network layers using the chain rule (backpropagation), enabling Stochastic Gradient Descent (SGD).

---

## 6. Differentiable Programming

Deep learning generalizes beyond standard networks of interconnected linear nodes to the broader paradigm of **Differentiable Programming**:

* A neuron evaluates a simple parameterized subroutine: it computes a weighted dot product followed by a non-linear activation.
* Under differentiable programming, any general programmatic routine $F(x, \theta)$ parameterized by $\theta$ on inputs $x$ can serve as a component in a learning system, provided derivatives can be evaluated through it:
  * **$\nabla_\theta F$ (Gradient with respect to parameters):** Guides optimization algorithms during training to calibrate $\theta$ and minimize prediction errors.
  * **$\nabla_x F$ (Gradient with respect to inputs):** Allows gradient signals to flow backward through $F$ into preceding modules, enabling end-to-end integration within larger architectures.

![Gemini visual 8](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-08.svg)

---

## 7. Summary & Conceptual Roadmap

![Gemini visual 9](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-09.svg)

* **Core Foundation:** Simple neural network formulations, supervised learning targets (regression vs. classification), and gradient calculus.
* **Scaling Up:** Multi-layer deep architectures, backpropagation algorithms, and production software frameworks.
* **Evaluation & Criticism:** Understanding model bounds, systemic biases, data constraints, and failure modes across real-world systems.
