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

## Exercise 2: Complex Chemistry Classification

Now open and run the Neural Networks notebook. Instead of regressing `K_VRH`,
the bulk modulus, directly, we're going to group the values into "high" and
"low" and treat it as a classification problem.

1. There's no longer any need to normalize `y_t`, so where the notebook has

```python
ds = TensorDataset(X_aff.inv(X_t), y_aff.inv(y_t))
```

remove the `y_aff.inv`.

2. Above that line, convert `y_t` to Trues and Falses, such that a value is True
   if the bulk modulus is above the mean bulk modulus and False otherwise.
   PyTorch tensors support the same kinds of elementwise operations you've seen
   with numpy and pandas: you should be able to do this in a single line.

3. To make some of the later code work, you'll need to go from True/False to
   1/0. Do this by putting `y_t = y_t.float()` after the code you just added.

4. To train a neural network as a classifier, we need to use _cross-entropy
   loss_. The math here isn't super important for us right now (feel free to ask
   about it if you're interested!) The upshot is that cross-entropy loss is a
   way of assessing classification that depends on the predicted probabilities,
   not just the most likely outcome. So a small change in the parameters will
   have a meaningful impact on cross-entropy, whereas a small change may not
   change any prediction and will therefore leave accuracy unchanged.

   Replace the loss function with `F.binary_cross_entropy_with_logits`. Our
   network will output a _logit_, which is the log-odds: each probability has a
   corresponding logit, and unlike probabililties logits don't have to be
   between 0 and 1. This means that we don't need to change our model outputs:
   we were originally predicting the exact bulk modulus, and now we predict the
   log-odds that the bulk modulus is high.

5. Update the training loop to show accuracy instead of the regression metrics.
   Feel free to show a confusion matrix or other plots.

6. Once you get it working, feel free to mess around with the hyperparameters.
   What's the highest accuracy you can get?

_Hint_: To go from logits to normal probabilities, use `F.sigmoid(logits)`. To
go from logits to the predicted class, use `logits > 0`.
