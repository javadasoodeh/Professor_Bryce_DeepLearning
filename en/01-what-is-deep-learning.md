**Deep Learning · DL 01**

# What Is Deep Learning?

Hi, welcome to deep learning. I'm Bryce, and I'm excited to help you learn about one of the hottest topics in computer science. The starting question is simple: what is deep learning, and why should you be interested?

Deep learning is all around us every day. The algorithms that recognize your face when you unlock a device, understand your speech when you talk to a digital assistant, or recommend content on your favorite platform are all based on deep learning.

A useful definition for this course is:

> **Deep learning is the use of neural networks and differentiable programming to perform machine learning.**

That definition immediately raises more questions: what are neural networks, and what is machine learning?

![Whiteboard overview of the deep learning lesson](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-09.svg)
The lesson connects machine learning, neural networks, activation functions, gradient descent, and differentiable programming.

## Machine learning

If we categorize machine learning, it lies at the intersection of **artificial intelligence** and **data science**. Data science is about organizing, analyzing, and making good use of data. Artificial intelligence is about solving problems computationally that seem to require intelligence when solved by humans.

So machine learning is about making intelligent inferences automatically from data.

![Machine learning drawn as the overlap of artificial intelligence and data science](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-01.svg)
Machine learning is placed at the overlap of artificial intelligence and data science.

This requires a perspective shift from how we normally do things in computer science. Usually, we design algorithms to solve a problem directly. In machine learning, instead, the data examples define the inputs and outputs of the problem, and we implement algorithms whose job is to infer the solution to the problem from the data set.

## Regression and classification

Broadly speaking, most machine-learning problems can be broken down into two categories: **regression** and **classification**.

![Regression example on the whiteboard](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-02.svg)
**Regression:** infer a function that maps continuous inputs to continuous outputs. The simplest example shown is linear regression: choose the line that best describes the relationship between the inputs and outputs for the points in the data set.

![Classification example on the whiteboard](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-02.svg)
**Classification:** assign a discrete label to each input point. The simple example is a decision boundary separating the region labeled 0 from the region labeled 1.

Both regression and classification can become much more complicated than these two-dimensional examples, and many interesting problems combine aspects of both.

### Object recognition combines both ideas

A classic deep-learning task is object recognition. The input is an image represented as a grid of pixels. For each object in the image, the output includes a bounding box described by continuous coordinates, as well as a label saying what the object is in one of several discrete categories.

![Fruit objects shown with bounding boxes and labels](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-03.svg)
The bounding box contributes continuous coordinates, while “orange” and “apple” are discrete labels.

We will start with simple examples to learn the ropes, then move quickly toward the much more exciting kinds of problems that deep learning can solve.

## Neural networks: the computational model

Deep learning solves these kinds of problems using neural networks and differentiable programming. A neural network is a computational model based very loosely on the behavior of neurons in the brain.

Neurons activate and send electrical signals that are sensed by other neurons, which may then activate in turn. Some connections are stronger than others; some pairs of neurons tend to activate together, while others have an inhibitory relationship.

![A neural network graph next to a biological neuron drawing](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-04.svg)
The biological idea is translated into a graph: nodes represent neurons and directed edges represent connections.

This inspires a mathematical model in which each directed edge has a numerical **weight** indicating the strength of the connection. A positive weight means the second neuron is excited by the first; a negative weight means it is inhibited.

Each neuron can sum the weighted inputs from its neighbors to determine whether it activates.

![Single neuron with three incoming weighted connections](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-05.svg)
The example neuron has three incoming activations. The weights are 2, 1, and −1, and the activation threshold shown inside the neuron is ≥ 0.5.

The top neighbor is active with activation 1 and weight 2, so it has a stronger positive influence. The middle neighbor has activation 0 and weight 1. The bottom neighbor is active with activation 1 and weight −1, so when it is active it makes the neuron less likely to activate.

To determine whether the neuron activates, multiply each neighbor’s activation by the corresponding weight and add the results:

$$
(1)(2) + (0)(1) + (1)(-1) = 2 + 0 - 1 = 1
$$

Then compare the sum with the activation threshold. Since $1 > 0.5$, the neuron activates and outputs 1.

![Weighted neuron after the sum is evaluated](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-05.svg)
The worked result on the board is $2 + 0 − 1 = 1$, producing an output of 1.

## Where deep learning breaks away from neuroscience

This is the point where deep learning breaks away from neuroscience. There is a great deal of interesting research about how neurons really activate and how the connections between them change, but that research is not what we will use here. Instead, the neural network is treated as a practical tool for performing machine learning.

To use it that way, we need a method for training a neural network from examples in a data set. The key idea behind that training is **gradient descent**.

From calculus, a function’s gradient points in the direction of steepest increase. That directional information can therefore help us minimize the function by moving in the opposite direction.

![Gradient descent drawn on a bowl shaped function](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-06.svg)
The gradient provides directional information; at the minimum, the drawing marks $∇F = 0$.

We will go much more deeply into this soon. For now, the important point is that because training needs gradients, we need to be able to differentiate the neural network’s activations.

## The activation problem: the step function

Think of this neuron’s activation as a function of its input sum. With the threshold at 0.5, the function outputs 0 everywhere below 0.5. At 0.5 it jumps upward and starts outputting 1. So the neuron’s activation is described by a **step function**.

![Step activation function being drawn](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-07.svg)
First the threshold location and lower level are established.

![Completed step activation function](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-07.svg)
The completed activation jumps from 0 to 1 at the threshold.

> **The problem:** the derivative of this step function is incredibly uninformative. It is zero everywhere the function is flat and undefined at the one point where the function is discontinuous.

If gradients are going to train the neural network, we need an activation function with nicer derivatives.

## A smooth alternative: the sigmoid

The first example of a better activation function is a smooth approximation to the step function, known as a **sigmoid**.

![Sigmoid curve replacing the hard step](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-07.svg)
The hard jump is replaced by a smooth transition.

![Sigmoid curve with its formula](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-07.svg)
The board writes the sigmoid as $σ(x) = 1/(1 + e^{−x})$.

$$
σ(x) = 1 / (1 + e^{−x})
$$

In the next few lectures, we will develop the tools to differentiate activation functions and pass those derivatives around through the network. That will set us up to use **stochastic gradient descent** to train a neural network on regression and classification problems.

## Differentiable programming

Many of the ideas used to train a neural network also apply more broadly. A neuron can be viewed as computing a very simple program: take a weighted sum, then apply an activation function. We can imagine more interesting programs filling the same role.

This leads to **differentiable programming**. Suppose we can write a function operating on numerical inputs and tuned by numerical parameters, and we can calculate derivatives of that function. Then that function can also be incorporated into a deep-learning model.

![Differentiable programming diagram with F of x and theta](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-08.svg)
A program computes $F(x, θ)$. The board notes that $∇_{θ}F$ lets us train $F$, while $∇_{x}F$ lets us use $F$ inside a larger deep-learning model.

## Where the course goes from here

We will begin with the simplest possible neural networks so that we can dig into the basics of machine learning and the mathematics behind stochastic gradient descent. Then we will gradually add complexity, building up to deep neural networks and general differentiable programs.

Along the way, we will get lots of practice with powerful deep-learning libraries, and we will extensively discuss the limitations and downsides both of particular models and of deep learning in general.

**By the end of the semester, you should be prepared to design, apply, evaluate, and criticize deep-learning models for a wide variety of exciting real-world problems.**

![Completed whiteboard at the end of the lesson](../assets/images/gemini-videos-01-03/inline-svg/lesson-01-09.svg)
The completed board ties together the full path: machine learning → neural networks → differentiable activations and gradients → differentiable programming.

---

Source: [https://www.youtube.com/watch?v=DrhJLHiia7g](https://www.youtube.com/watch?v=DrhJLHiia7g)
