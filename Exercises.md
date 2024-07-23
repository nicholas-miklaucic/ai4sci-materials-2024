# Exercises

## Exercise 0 (optional)

- Think of 3 examples of classification that haven't been mentioned.
- Think of 3 examples of regression that haven't been mentioned.

## Exercise 1: Perfecting Penguin Prediction

This exercise uses the Traditional Machine Learning notebook. Run all of the
cells using the "Run All" button or the option in the Kernel menu.

Go to the cell that runs the regression model, near the bottom. The relevant
section is

```python
pipe = make_pipeline(
    transform,
    RandomForestRegressor()
)
```

This got an R2 score of about 0.759, with some variation due to randomness.
Let's see if you can do better than me!

Replace the `RandomForestRegressor()` with a different regression model in
scikit-learn. Don't worry about the math behind every potential model: they all
have the same interface, so you can just replace `RandomForestRegressor()` with
a different model and it should work fine.

[The main scikit-learn page](https://scikit-learn.org/stable/index.html) has a
"Regression" section that displays a few common algorithms. Maybe try one of
those. You'll just need to import the model from the correct submodule and then
replace `RandomForestRegressor()`. For example, if I went to the
[Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#)
section of the User Guide and wanted to use `SVR`, which is under the
"Regression" section, I could see that it comes from `sklearn.svm` and then
replace the above code with

```python
from sklearn.svm import SVR
pipe = make_pipeline(
    transform,
    SVR()
)
```

What model did you end up using? What R2 score, roughly, do you get? Did you
change any of the hyperparameters?

## Exercise 2: Analyzing Appropriate Activations

Open and run the Neural Networks notebook. (Runtime > Run All or Kernel > Run
All in the menu.)

Go to the section where we define our hyperparameters:

```python
hidden_size = 128
act = F.relu
```

As mentioned in the morning, there are many different activation functions
researchers have developed, with different advantages and disadvantages.

### 2a: Home Cooking

First, let's try making our own activation function! Define a function that
takes in numeric inputs and gives numeric outputs. It should have some kind of
_nonlinearity_: it can't just be multiplication or addition by a constant. We'll
define a function for our activation:

```python
def my_activation(x):
   """The newest activation function that will take the world by storm!"""
   return x ** 2

# then, replace act = nn.ReLU() with:
act = my_activation
```

I set it to compute the square of the input. Replace `x ** 2` with your own
code. If you want to use math functions you might see on your calculator or in
school, use `torch.` to get the right function. For example, to get the sine of
the inputs, do `torch.sin(x)`.

Now run that cell and then run the training loop. Give the R2 of your model.
Does your model perform better or worse than the version with ReLU activations?

### 2b: Good Artists Copy, Great Artists Steal

People have spent a lot of time coming up with activation functions that work
well for many neural networks, with lots of complex math and fancy stuff.
PyTorch implements many popular activation functions so we don't have to do it
ourself.

Go to the
[PyTorch documentation section on activation functions.](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

Most of these functions can be used as a drop-in replacement for ReLU without
any changes. (Do **not** use `MultiHeadAttention` or `GLU`. They won't work.)

Let's start by using `nn.ELU()`. The page is
[here](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU).
Note how it looks similar to ReLU, but it's smoother and goes a little below 0.

Replace the activation function with `nn.ELU()`:

```python
act = nn.ELU()
```

Now try a different function. Pick your favorite from the list (or try several!)
and use it to replace ELU or ReLU. What function did you pick, and what R2 score
did you get? Check the documentation page for the function. Can you see a graph?
Does it look similar to ReLU or different?

## Exercise 3: Complex Chemistry Classification

If your changes from Exercise 2 ended up having a really bad impact, undo them
or re-download the notebook. Instead of regressing `K_VRH`, the bulk modulus,
directly, we're going to group the values into "high" and "low" and treat it as
a classification problem.

1. There's no longer any need to normalize `y_t`. Where we define `y_aff`,
   replace it with a transform that won't do anything:

```py
# y_aff = AffineTransform(torch.tensor(y.mean()).float(), torch.tensor(y.std()).float())
y_aff = AffineTransform(0, 1)
```

Re-run the cells.

2. In the next cell, add code to convert `y_t` to Trues and Falses.

```py
y_t = torch.from_numpy(y).float().reshape(-1, 1)

# [YOUR CODE HERE]

ds = TensorDataset(X_aff.inv(X_t), y_aff.inv(y_t))
```

You want to figure out how to make it so `y_t` is True whenever the value is
higher than the mean and False otherwise. PyTorch tensors support the same kinds
of elementwise operations you've seen with numpy and pandas: you should be able
to do this in a single line, but it's OK if you find a different way.

3. To make some of the later code work, you'll need to go from True/False to
   1/0. Do this by putting `y_t = y_t.float()` after the code you just added,
   before the line that starts with `ds = `.

4. To train a neural network as a classifier, we need to use _cross-entropy
   loss_. The math here isn't super important for us right now (feel free to ask
   about it if you're interested!) The gist is that cross-entropy loss is a way
   of assessing classification that depends on the predicted probabilities, not
   just the most likely outcome. So a small change in the parameters will have a
   meaningful impact on cross-entropy, whereas a small change may not change any
   prediction and will therefore leave accuracy unchanged.

   Replace the loss function with `F.binary_cross_entropy_with_logits`. Our
   network will output a _logit_, which is the log-odds if that means anything
   to you. Each probability has a corresponding logit, and unlike probabililties
   logits don't have to be between 0 and 1. This means that we don't need to
   change our model outputs: we were originally predicting the exact bulk
   modulus, and now we predict the log-odds that the bulk modulus is high.

5. Update the training loop to show accuracy instead of the regression metrics.
   Feel free to show a confusion matrix or other plots.

6. Once you get it working, feel free to mess around with the hyperparameters.
   What's the highest accuracy you can get?

_Hint_: To go from logits to normal probabilities, use `F.sigmoid(logits)`. To
go from logits to the predicted class, use `logits > 0`.

## Exercise 4: Connecting Cooler Convolutions

Now open the Graph Neural Networks notebook and run all the cells.

Scroll to the cell that has this code (use Ctrl+F or Cmd+F if that helps):

```py
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool


Conv = GraphConv
```

PyTorch Geometric has many different layers that implement the same rough idea:
each node gets a chance to 'talk' to its neighbors.

The documentation on these layers is
[here](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers).

Anything that starts with `Conv` should be workable. You'll need to import it.
For example, here I've imported `GCNConv` in addition to `GraphConv`, and now
I'll use it.

```
from torch_geometric.nn import GCNConv, GraphConv

Conv = GCNConv
```

Replace `GCNConv` with a different convolution layer, if you prefer, and then
run the subsequent cells to train it. What performance do you get? Is it better
or worse than `GraphConv`?

If one doesn't work, try another.

The math behind these layers can be pretty intimidating, so don't worry too much
if you look at the documentation page for a layer and don't understand it. Of
course, feel free to ask about anything, but the goal here is to see how you
might use a pre-existing model from the literature without having to read a lot
of papers first.

## Exercise 5: Extracting Elemental Embeddings

If your change from earlier made the model worse, undo your change by setting
`Conv = GraphConv`.

Look at the `__init__` method of the `GCN` module. Change the code so that the
elements have a 2-dimensional embedding. You may need to look at the
documentation of the `nn.Embedding` layer to see how to change the dimension of
the embedding.

You will also need to update the `conv1` layer so it accepts 2-dimensional
inputs.

Re-run the training code. Does your model do better or worse? Why do you think
that might be?

Once that's done, go to the cell at the bottom that plots a heatmap. With only 2
dimensions, we can simply plot the embeddings directly instead of needing to do
a correlation. Replace that code with

```python
embed_df = pd.DataFrame(model.embed.weight.cpu().detach().numpy(), index=dataset.atoms(), columns=['x', 'y'])
sns.scatterplot(embed_df, x='x', y='y', s=5)
embed_df.head()
```

You should see a scatterplot and then it will show the exact numbers in a
DataFrame.

Your task is to figure out how to modify our scatterplot to show the symbol of
each element. You can do this by using the `plt.text` command. For example, to
add text at location (0, 1), we could write `plt.text(0, 1, "Hi Mom!")`. Using a
for-loop, add text for every element at the right (x, y) location as seen in the
DataFrame.

When that's done, maybe bring up a periodic table at [ptable.com](ptable.com).
Do you see any patterns in the embeddings? Has your model learned some kind of
chemically meaningful information? Don't worry if nothing pops out, as long as
you have the plot.
