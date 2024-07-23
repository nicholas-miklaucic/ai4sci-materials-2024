#import "@preview/touying:0.4.0": *

// Themes: default, simple, metropolis, dewdrop, university, aqua
#let s = themes.dewdrop.register(aspect-ratio: "16-9", navigation: none)
#let s = (s.methods.info)(
  self: s,
  title: [Introduction to Machine Learning in Python],
  author: [Presented by Nicholas Miklaucic],
  date: datetime.today(),
)
#let (init, slides, touying-outline, alert) = utils.methods(s)
#show: init

#import "@preview/cetz:0.2.2": canvas, plot


#set text(size: 18pt, font: "Source Sans Pro", weight: "regular")
#set par(justify: false)
#show quote: set text(size: 18pt, weight: "regular")
#show quote: set par(first-line-indent: 2em)
#show quote: set block(above: 1em, below: 0em)
#set quote(block: true)
#show figure.caption: set text(size: 14pt, weight: "regular")
#show heading.where(level: 3): set text(weight: "semibold", fill: blue)

#let (slide, empty-slide, title-slide, focus-slide) = utils.slides(s)

#show: slides.with(slide-level: 1)


= Introduction: What is Machine Learning?

== Machine Learning (ML)
Machine learning (ML) is when a computer does something through some kind of experience, not just a
pre-programmed algorithm.. More formally:

#quote([A computer program is said to learn from experience E with respect to some class of tasks T,
and performance measure P, if its performance at tasks in T, as measured by P, improves with
experience E.], attribution: [Tom Mitchell, *Machine Learning*])

== Supervised Learning

- Prediction of some _target_, often $y$, from labeled training data, often $X$.
There are two broad kinds of supervised learning:
- _Classification_ predicts one of a set of discrete _classes_, like diagnosing a disease.
- _Regression_ predicts a numerical output, like predicting the price a house will sell at.

#grid(columns: (1fr, 1fr),
[#figure(image("images/house.png", height: 53%), caption:"Zillow thinks this house is worth $4
million.")], [#figure(image("images/dog.jpg", height: 53%), 
caption:"A ML model could classify this image as a picture of a dog.")])

== Supervised Learning
Consider this an unofficial exercise 0: what are other examples of classification and regression?
What examples might appear in materials science?

= Evaluating ML Models
== Evaluating ML Models
If we have a model making predictions, how do we assess how good our prediction is?

Let's call what our model predicts $hat(y)$ ("y-hat"), and we have the actual $y$ to compare it
with.



#grid(columns: (2fr, 3fr),
[== Classification
We can plot $hat(y)$ vs. $y$ in a _confusion matrix_.
This is basically all of the important information we need to evaluate the model.
Sometimes we use the raw counts, sometimes we normalize by row like here.

The most common metric is _accuracy_: the percent of correct predictions. 
(This is the fraction of values on the main diagonal.)

There are many other metrics that can be better when classes are imbalanced. For now, we'll stick
with accuracy, but looking at the full confusion matrix gives a lot more information.],
[#figure(image("images/confmat.png", height: 100%), 
caption:"Reproduced from Li, Dong, Yang, & Hu 2021")])


#grid(columns: (4fr, 3fr),[== Regression
We can also plot the predicted vs. actual values, as shown here.

There are a couple common metrics you will see:

*Mean Squared Error (MSE)*: $"mean"((y - hat(y))^2)$

*Root Mean Squared Error (RMSE aka L2)*: $sqrt("mean"((y - hat(y))^2))$

*Mean Absolute Error (MAE aka L1)*: $"mean"(|y - hat(y)|)$

$R^2$: $display(1 - "Var"(y - hat(y))/("Var"(y)))$

MSE is in units of the target squared.
RMSE and MAE are in the same units as the target: they're different versions of "average" error.
$R^2$ is unitless: it measures the percent of the variance in the target we can explain with the
inputs, so 0 is the same as no information and 1 is perfect.
],
[#figure(image("images/piezo_scatter.png", height: 100%), 
caption:"Formation energy prediction")])

== Validation
We want to know how well our model will do on _new_ inputs. If we can compute accuracy or MAE, then
we already know those answers! Scientists don't want AI to tell them what they already know.

Moreover, performance on training data isn't predictive of future performance. Models can _overfit_,
finding patterns that don't hold up.

We want to see how our model will do on data it hasn't seen before. How can we do that?

== Hold-out Set

We can split our data into sets. Let's say we have 100 _samples_.

If we train our model on 80 of those samples, and then test it on the remaining 20, we have a better
estimate of our model's future performance.

If we have a ton of data, this works well. If we don't have enough data, we face a tricky dilemma:
- If we use too much data for validation, our model won't have enough training data, and our performance will be much lower than it would be had we trained on the whole set.
- If we use too little data for validation, the estimates will be unreliable, and randomness in how our validation set is chosen will end up dominating the results.

== Cross-Validation

We can get the best of both worlds by training multiple models.

If we split our 100 samples into groups of 20, called _folds_, we can train 5 models. Each model is
trained on the other 80 samples and then predicts for a single fold. At the end, we have predictions
for each sample, none of which were tainted by the model being able to train on the correct answer.

Now our tradeoff is purely computing power. The ideal is _leave-one-out cross-validation_ (LOO-CV),
where we train a model for every data point. LOO-CV is the best estimate of future performance, but
it requires orders of magnitude more computing power. Normally, we use CV with 5 or 10 folds as a
more reasonable baseline.

= Example: Random Forest
== Decision Tree
#align(center, image("images/decision_tree.jpg", height: 90%))

== Random Forest
#align(center, image("images/random_forest.png", height: 90%))

= A Practical Demonstration

= Neural Networks

== Gradient Descent
Let's imagine fitting an equation like $a x^2 + b x + c$ to some data.

We can define a _loss function_ that quantifies the error of our model.

Then, using calculus, we can figure out how changing each parameter a little bit will affect the
loss. Using that information, we can change the parameters to decrease the loss. Repeating this
enough times will let us find good values of parameters, hopefully.

(Calculating gradients is what PyTorch does, so we won't have to worry about any calculus!)

== Gradient Descent
Imagine trying to get down from a mountain in a blizzard, so you can't see at all in front of you.
You would probably just try to go downhill, and that's what we do here. We aren't guaranteed to find
the best model, but this works well in practice. (Interestingly, this is not at all how humans
learn!)

#align(center, image("images/grad-descent.png", height: 60%))

== Everything's a Number If You Try Hard Enough
What we just described only works when we have a smooth function that maps numbers to numbers and a
quantitative loss function. 

How can we make a function that takes in non-numeric input, like a chemical element? A simple method
is _one-hot encoding_: we have columns for each element that are 1 or 0 depending on whether the
input is that element or not.

$
f("H") &= (1, 0, dots, 0) \
f("He") &= (0, 1, dots, 0) \
dots.v
$

== Embeddings
A solution that often works better is to learn a list of numbers (a _vector_) for each potential
input. This lets us model how different inputs might be similar.

$
f("H") &= (1.2, 4.3, dots, -2.3) \
f("He") &= (1.3, -3.1, dots, 0.3) \
dots.v
$

In a language model, maybe the embeddings for _green_ and _verde_ are similar, so the model can
apply what it learns in one language for another language.

== Embeddings
#align(center, image("images/embeddings.png", height: 90%))

== Prelude: Lines
We want a flexible kind of model that can represent all kinds of functions.

For computational reasons, we want this to be built out of simple building blocks.

What's a simple example of a relationship between some $x$ and $y$? A line!

$ y = a x + b $
== Composing Neurons
What amazing things can this model do? I'm so excited to find out!
#align(center, image("images/neural_network.png", height: 80%))

== Composing Neurons
This is still just a line. We just made an overly complicated line.
#align(center, image("images/sad-pikachu.gif", height: 80%))

== Beyond Linearity 

To prevent this network from just being a fancy line, we need to use some function that isn't just
multiplication or addition. Basically any function will do.

This is called an _activation function_, and people have tried a *lot* of them. Let's use a simple
one: the _rectified linear unit_, or ReLU:

$ "ReLU"(x) = max(x, 0) $

Above 0, this just returns the input, but below 0 it returns 0.

Let's add this function to the inner neurons:

== Beyond Linearity
#align(center, image("images/neural_network_relu.png", height: 90%))

== Example

With enough neurons, these networks are extremely flexible. In fact, you can show that they can
approximate _any_ smooth function with enough neurons.

https://www.desmos.com/calculator/jtcieaiwa7

= Specialized Neural Networks
== Specialized Neural Networks
In theory, with enough neurons even a simple network will work.

In practice, some ways of connecting neurons are better than others for specific tasks. The work of
an AI researcher is often trying to find good ways of building networks for a specific task.

Remember: no matter how complex diagrams get, it's just a big equation with sliders. The training
process is the same.

== Convolutional Neural Network
Neurons respond in the same way to a pattern regardless of where it appears in an image.

#align(center, image("images/cnn.png", height: 80%))

== Transformer
The T in ChatGPT, so you might have heard of this one!

In a sequence, compares each element to the others and uses a weighted average of the other values
depending on how much each element matches. A neural network is used to do the comparison.

This is called _attention_, and it's very useful for certain kinds of tasks. Here, pronoun shouldn't change depending on the activity: ideally, it would just be estimated from the name.


#quote([  
  *Nicholas* slipped spectacularly on the ice. *He* faceplanted into the earth.  
])


== Graph Neural Network
When we say _graph_, we mean a network of nodes and edges. I'll refer to the chart kind of graph as
a _plot._

In a graph neural network, we define functions using neural networks that control how information is
communicated between nodes. In the basic formulation, each node sends a _message_ to its neighbors
and then each node is updated using those messages.

#align(center, image("images/molecule.png", height: 50%))


== Graph Neural Network

https://distill.pub/2021/gnn-intro/

= A Practical Demonstration: the Sequel