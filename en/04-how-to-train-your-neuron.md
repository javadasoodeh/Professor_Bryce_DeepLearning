**One Neuron**

# How to train your neuron

A single neuron first forms a weighted sum of its inputs, adds a bias, and then applies an activation function. With a step function as the activation, it can act as a binary classifier. With a linear activation, it can act as a regressor. The next question is how to choose the weights and bias so that the model fits the data.

The model is trained by measuring its loss on a data set and then using the gradient of that loss to change the parameters. The loss depends explicitly on the neuron’s weights and bias.

<p align="center" dir="ltr"><font size="5">L(w<sub>1</sub>,…,w<sub>n</sub>,b) = Σ<sub>j=1</sub><sup>N</sup> ( y<sub>j</sub> − f(x⃗<sub>j</sub>) )²</font></p>

In computations it is common to divide this by the number of data points and use mean squared error. For the derivation, it is cleaner to work with the sum of squared errors so that a factor of 1/N does not have to be carried through every derivative.

![Whiteboard recap of one neuron, regression, classification, and loss](../assets/images/lesson-04/en/lesson-04-01.png)

*The board begins with the one-neuron computation, the regression and classification cases, and the squared-error loss written as a function of the model parameters.*

## A small regression example

Start with a four-point, one-dimensional regression data set. The current neuron has weight w<sub>1</sub>=2 and bias b=−2, so its output is the straight line

<p align="center" dir="ltr"><font size="5">f(x) = 2x − 2.</font></p>

At the four inputs x=1,2,3,4, the model therefore predicts 0,2,4,6. The desired outputs are 1,3,2,4. The pointwise errors y−f(x) are 1,1,−2,−2.

<div dir="ltr">

| x | target y | prediction f(x) | error y−f(x) |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 1 |
| 2 | 3 | 2 | 1 |
| 3 | 2 | 4 | −2 |
| 4 | 4 | 6 | −2 |

</div>

<p align="center" dir="ltr"><font size="5">L(2,−2) = (1−0)² + (3−2)² + (2−4)² + (4−6)² = 10.</font></p>

![Regression loss calculation on the whiteboard](../assets/images/lesson-04/en/lesson-04-02.png)

*The four residuals are written directly into the sum of squared errors; their squares add to 10.*

The goal is to minimize this loss. Calculus tells us that the gradient points in the direction of steepest increase, so the direction that decreases the loss is the negative gradient. Because this neuron has two parameters, the gradient has two components: one partial derivative with respect to w<sub>1</sub>, and one with respect to b.

## Deriving the gradient in the general case

Instead of deriving only this particular example, take the loss for an arbitrary neuron and an arbitrary data set. There is one partial derivative for every model parameter. For an arbitrary weight w<sub>i</sub>, differentiation passes through the summation, and the chain rule is applied to each squared term.

<p align="center" dir="ltr"><font size="5">∂L/∂w<sub>i</sub> = −2 Σ<sub>j=1</sub><sup>N</sup> ( y<sub>j</sub> − f(x⃗<sub>j</sub>) ) · ∂f(x⃗<sub>j</sub>)/∂w<sub>i</sub>.</font></p>

![Chain rule applied to the squared error loss](../assets/images/lesson-04/en/lesson-04-03.png)

*The square contributes the factor of 2; differentiating the inside contributes a minus sign because y<sub>j</sub> is constant and the model output is subtracted.*

The activation function cannot yet be differentiated in one universal way because different neurons can use different activations. But every neuron begins with the same weighted sum. For data point j, write the pre-activation weighted sum as x<sub>j</sub> (without the vector mark):

<p align="center" dir="ltr"><font size="5">x<sub>j</sub> = b + Σ<sub>k=1</sub><sup>n</sup> w<sub>k</sub>x<sub>j,k</sub>.</font></p>

Only one term in that sum contains w<sub>i</sub>. Therefore

<p align="center" dir="ltr"><font size="5">∂x<sub>j</sub>/∂w<sub>i</sub> = x<sub>j,i</sub>, and ∂x<sub>j</sub>/∂b = 1.</font></p>

![Derivative of weighted sum with respect to a weight](../assets/images/lesson-04/en/lesson-04-04.png)

*The derivative of the weighted sum with respect to weight w<sub>i</sub> leaves exactly the corresponding input component x<sub>j,i</sub>; the bias derivative is 1.*

The repeated quantity y<sub>j</sub>−f(x⃗<sub>j</sub>) is exactly the unsquared difference already computed in the loss calculation. Name it the error e<sub>j</sub>. Then

<p align="center" dir="ltr"><font size="5">e<sub>j</sub> = y<sub>j</sub> − f(x⃗<sub>j</sub>), so ∂L/∂w<sub>i</sub> = −2 Σ e<sub>j</sub> · ∂f(x⃗<sub>j</sub>)/∂w<sub>i</sub>.</font></p>

![Error term introduced in the gradient formula](../assets/images/lesson-04/en/lesson-04-05.png)

*The board labels the target-minus-prediction difference as e<sub>j</sub>, tying the gradient calculation directly back to the residuals used in the loss.*

## Applying the gradient to the linear regressor

For the linear activation, the derivative of the activation is 1. The chain rule therefore leaves only the corresponding input component when differentiating with respect to a weight. For w<sub>1</sub>:

<p align="center" dir="ltr"><font size="5">∂L/∂w<sub>1</sub> = −2 Σ e<sub>j</sub>x<sub>j,1</sub> = −2[1·1 + 1·2 − 2·3 − 2·4] = 22.</font></p>

For the bias, the weighted-sum derivative is 1:

<p align="center" dir="ltr"><font size="5">∂L/∂b = −2 Σ e<sub>j</sub> = −2[1 + 1 − 2 − 2] = 4.</font></p>

So the gradient at w<sub>1</sub>=2, b=−2 is

<p align="center" dir="ltr"><font size="5">∇L = [22, 4]<sup>T</sup>.</font></p>

![Numerical regression derivatives and gradient](../assets/images/lesson-04/en/lesson-04-06.png)

*The board evaluates both partial derivatives numerically and combines them into the gradient vector [22,4].*

## What the gradient means geometrically

The loss is a function of the model parameters. Here its two inputs are w<sub>1</sub> and b. Every point in the (w<sub>1</sub>,b) plane corresponds to a different line the neuron could compute. For example, w<sub>1</sub>=−3 and b=2 would define a different linear neuron. Our current point is (2,−2), where the loss is 10.

Imagine the loss value rising out of this parameter plane as a surface. At any point, the gradient points in the direction in which that surface rises most steeply. To decrease the loss, move in the opposite direction.

![Parameter plane with weight and bias axes](../assets/images/lesson-04/en/lesson-04-07.png)

*The parameter plane is drawn with w<sub>1</sub> horizontally and b vertically; each point stands for one possible parameterization of the neuron.*

Take a small step by subtracting a multiple of the gradient:

<p align="center" dir="ltr"><font size="5">[w<sub>1</sub>, b]<sup>T</sup> ← [w<sub>1</sub>, b]<sup>T</sup> − η∇L.</font></p>

The multiplier η is the step size. With η=0.01, subtract 0.22 from the weight and 0.04 from the bias:

<p align="center" dir="ltr"><font size="5">w<sub>1</sub>: 2 → 1.78, b: −2 → −2.04.</font></p>

![Gradient descent step with eta 0.01](../assets/images/lesson-04/en/lesson-04-08.png)

*The update direction is drawn on the parameter plane and η=0.01 is written beneath the numerical gradient.*

With the new parameters, evaluate the loss again, calculate another gradient, and take another step. Repeating this process improves the model.

The loss surface for this example is much steeper in the weight direction than in the bias direction. That matches the model: changing the weight alters the slope and can change the errors quickly across the whole x-range, while changing the bias only nudges the line up or down.

![Contour lines of regression loss surface](../assets/images/lesson-04/en/lesson-04-09.png)

*Elliptical contour lines are sketched around the low-loss region, elongated to show very different steepness in the two parameter directions.*

After the first step, the bias is slightly smaller, so the intercept shifts down. The weight decreases much more, so the line’s slope is reduced. Visually, the fitted line is shifted slightly downward and tilted to the right, which better matches the four data points.

## Why the step-function classifier cannot be trained this way

Trying to apply the same gradient calculation to the step activation runs into an immediate problem: the derivative of a step function is zero almost everywhere. That makes the gradient carry essentially no useful information about how the parameters should move.

![Transition from regression to classification](../assets/images/lesson-04/en/lesson-04-10.png)

*The lesson returns to the classification half of the board after completing the regression gradient-descent picture.*

The solution is to replace the hard step by a smooth approximation: the sigmoid function.

<p align="center" dir="ltr"><font size="5">σ(x) = 1 / (1 + e<sup>−x</sup>).</font></p>

When x is large and positive, e<sup>−x</sup> is tiny, so σ(x) approaches 1. When x is very negative, e<sup>−x</sup> is huge, the denominator grows without bound, and σ(x) approaches 0. Between the two extremes there is a smooth transition.

![Sigmoid formula and smooth S-shaped activation](../assets/images/lesson-04/en/lesson-04-11.png)

*The hard step is replaced on the board by a smooth S-shaped curve, with σ(x)=1/(1+e<sup>−x</sup>) written above it.*

## Classification with the sigmoid

A classification boundary can still be drawn where the weighted sum changes from negative to positive. But close to that boundary, the sigmoid output is around 0.5; only far from the boundary does it become close to 0 or 1. That is sensible: points lying almost exactly on the boundary should not receive the same confidence as points well inside one side.

Use the neuron with w<sub>1</sub>=1, w<sub>2</sub>=1, and b=−4. The six data points on the board are:

<div dir="ltr">

| x<sub>1</sub> | x<sub>2</sub> | target y | weighted sum x<sub>1</sub>+x<sub>2</sub>−4 | σ(weighted sum) |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0 | −1 | 0.27 |
| 2 | 1 | 0 | −1 | 0.27 |
| 2 | 3 | 1 | 1 | 0.73 |
| 3 | 2 | 1 | 1 | 0.73 |
| 4 | 1 | 1 | 1 | 0.73 |
| 4 | 2 | 1 | 2 | 0.88 |

</div>

With the original hard step, these weighted sums would produce predictions 0,0,1,1,1,1, so every point except the one on the wrong side of the drawn dividing line is classified as shown. With the sigmoid, those same weighted sums become graded outputs: for −1,

<p align="center" dir="ltr"><font size="5">σ(−1)=1/(1+e)≈0.27;</font></p>

for +1,

<p align="center" dir="ltr"><font size="5">σ(1)=1/(1+e<sup>−1</sup>)≈0.73;</font></p>

and for +2,

<p align="center" dir="ltr"><font size="5">σ(2)≈0.88.</font></p>

![Sigmoid classification boundary and data](../assets/images/lesson-04/en/lesson-04-12.png)

*The same linear boundary remains, but the smooth activation changes how strongly points near and far from that boundary are scored.*

![Classification table with sigmoid outputs](../assets/images/lesson-04/en/lesson-04-13.png)

*The right-hand table is filled with sigmoid outputs such as 0.27, 0.73, and 0.88 for the six data points.*

The loss can still be formed from squared differences between the target values and these predictions. The immediate need, however, is the gradient, so the sigmoid itself must be differentiated.

## Derivative of the sigmoid

Start from σ(x)=1/(1+e<sup>−x</sup>). Using the quotient rule, the derivative is

<p align="center" dir="ltr"><font size="5">dσ/dx = e<sup>−x</sup> / (1+e<sup>−x</sup>)².</font></p>

This expression can be rearranged into a particularly useful form:

<p align="center" dir="ltr"><font size="5">σ′(x) = σ(x)(1−σ(x)).</font></p>

The algebra for showing the equality is left as a check, but this identity is what makes the next gradient expression compact.

![Sigmoid outputs before differentiation](../assets/images/lesson-04/en/lesson-04-14.png)

*The numerical sigmoid outputs are in place just before the derivative is derived.*

![Derivative of sigmoid on the whiteboard](../assets/images/lesson-04/en/lesson-04-15.png)

*The quotient-rule derivative is written and identified with σ(x)(1−σ(x)).*

## Gradient for the sigmoid classifier

The same general squared-error derivative is used. Consider one weight first; the other weight and the bias follow the same pattern:

<p align="center" dir="ltr"><font size="5">∂L/∂w<sub>i</sub> = −2 Σ<sub>j=1</sub><sup>N</sup> e<sub>j</sub> · ∂f(x⃗<sub>j</sub>)/∂w<sub>i</sub>.</font></p>

Now f is the sigmoid applied to the weighted sum. Passing the chain rule through the sigmoid gives the sigmoid derivative evaluated at that weighted sum, multiplied by the derivative of the weighted sum. The latter is again just the corresponding input component x<sub>j,i</sub>.

<p align="center" dir="ltr"><font size="5">∂L/∂w<sub>i</sub> = −2 Σ<sub>j=1</sub><sup>N</sup> e<sub>j</sub> σ(x<sub>j</sub>)(1−σ(x<sub>j</sub>)) x<sub>j,i</sub>.</font></p>

![Beginning of sigmoid-classifier gradient derivation](../assets/images/lesson-04/en/lesson-04-16.png)

*The classifier derivative begins with exactly the same error term and chain-rule structure used for regression.*

![Completed partial derivative for classifier weight](../assets/images/lesson-04/en/lesson-04-17.png)

*The finished expression multiplies each error by σ(x<sub>j</sub>)(1−σ(x<sub>j</sub>)) and the input component associated with the chosen weight.*

For w<sub>1</sub>, use the first coordinate of each data point; for w<sub>2</sub>, use the second coordinate. For the bias, the weighted-sum derivative is 1, so the x<sub>j,i</sub> factor disappears. Putting the three partial derivatives together gives the gradient vector for (w<sub>1</sub>,w<sub>2</sub>,b).

Again, the gradient points toward increasing loss. Subtract a small multiple of it from the current parameters, recompute the gradient using the updated weights and bias, take another step, and continue.

![Final classifier gradient discussion](../assets/images/lesson-04/en/lesson-04-18.png)

*The completed classification derivative remains on the board while the update process is tied back to the same gradient-descent loop used for regression.*

## The same idea in higher dimensions

Nothing essential in the derivation depends on having one regression input or two classification inputs. If the input has more coordinates, the neuron simply has more weights. The formula for each partial derivative stays the same; it is applied once for every weight to build the full gradient vector.

For regression, this produces linear regression using a single neuron trained by gradient descent. For classification, replacing the step activation by a sigmoid produces logistic regression—again using only a single neuron and gradient descent.

![Final whiteboard at end of lesson](../assets/images/lesson-04/en/lesson-04-19.png)

*The final board contains the regression update, the loss-surface sketch, the sigmoid activation, its derivative, and the classifier-gradient formula side by side.*

---

[Source: https://www.youtube.com/watch?v=HU91yMSTU0Y](https://www.youtube.com/watch?v=HU91yMSTU0Y)
