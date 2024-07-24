# Exercise 0

Classification: detecting harmful prompts to a chatbot, predicting the winner of
a basketball game, and detecting AI-generated imagery.

Regression: predicting where an object will be in the next frame of a video,
forecasting the temperature in 48 hours, estimating how many transcripts of a
given gene sequence will appear in RNA sequencing.

# Exercise 1

Linear regression beats random forest regression on this dataset:

So does `sklearn.ensemble.GradientBoostingRegressor()`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
################# SOLUTION CODE
from sklearn.linear_model import LinearRegression
#################
from sklearn.metrics import r2_score

X = df[['bill_length_mm', 'bill_depth_mm', 'species']]
y = df['body_mass_g']

transform = ColumnTransformer([
    ('scaler', StandardScaler(), ['bill_length_mm', 'bill_depth_mm']),
    ('onehot', OneHotEncoder(), ['species'])
])
pipe = make_pipeline(
    transform,
    ################ SOLUTION CODE
    LinearRegression()
    ################
)

yhat = cross_val_predict(pipe, X, y, cv=10)

print(f'Root Mean Squared Error (RMSE):\t{np.sqrt(((y - yhat) ** 2).mean()):.4f}')
print(f'Mean Absolute Error (MAE):\t{np.abs(y - yhat).mean():.4f}')
print(f'Explained Variance (R^2):\t{r2_score(y, yhat):.2%}')
```

```
Root Mean Squared Error (RMSE):	338.0812
Mean Absolute Error (MAE):	269.1809
Explained Variance (R^2):	82.32%
```

# Exercise 2

Any function is fine here: the point is to try it out.

## 2a

```python
lr = 1e-3  # 1 * 10^-3 0.001
num_epochs = 100

hidden_size = 64
###################### SOLUTION CODE
def my_activation(x):
   """The newest activation function that will take the world by storm!"""
   return x / (1 + x ** 2)

# act = nn.ReLU()
act = my_activation
#####################
loss = F.mse_loss
```

This gets comparable but slightly worse performance on average.

## 2b

The Mish activation works better than ReLU for me.

```python
lr = 1e-3  # 1 * 10^-3 0.001
num_epochs = 100

hidden_size = 64
###################### SOLUTION CODE
act = nn.Mish()
#####################
loss = F.mse_loss
```

# Exercise 3

Change the `y_aff = AffineTransform(...)` line to read:

```py
# y_aff = AffineTransform(torch.tensor(y.mean()).float(), torch.tensor(y.std()).float())
y_aff = AffineTransform(0, 1)
```

In the next cell:

```py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

X_t = torch.from_numpy(X.values).float()
y_t = torch.from_numpy(y).float().reshape(-1, 1)

##############################
# NEW SOLUTION CODE
y_t = y_t > y_t.mean()
y_t = y_t.float()
#############################

ds = TensorDataset(X_aff.inv(X_t), y_aff.inv(y_t))
train_ds, val_ds = random_split(ds, [1 - val_frac, val_frac], generator=torch.random.manual_seed(123))
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=len(val_ds))
val_X, val_y = next(iter(val_dl))
```

Then, after the model definition, in the cell with hyperparameters:

```py
lr = 1e-3  # 1 * 10^-3 0.001
num_epochs = 100

hidden_size = 64
act = nn.ReLU()

############################
# DIFFERENT CODE
loss = F.binary_cross_entropy_with_logits
############################
```

Then, in the cell which has the training loop, right after that:

```py
###################
### SOLUTION_CODE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
###################

############################
### SOLUTION CODE
val_preds = (model(val_X) > 0).cpu().detach().numpy()
############################
val_true = val_y.cpu().float().detach().numpy()

print()
############################
### SOLUTION CODE
print(f'Accuracy: {accuracy_score(val_true, val_preds):.2%}')
############################
```

# Exercise 4

Using `GATv2Conv` as one of many options:

```python
from torch import nn
import torch.nn.functional as F
################ SOLUTION CODE
from torch_geometric.nn import GATv2Conv, GraphConv
################
from torch_geometric.nn import global_mean_pool

############## SOLUTION CODE
Conv = GATv2Conv
##############

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, act=F.relu, dropout_rate: float = 0.3):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)

# code continues
```

This does worse than `GraphConv`.

# Exercise 5

Updating the embedding dimension to 2 requires changing the `self.embed` and
`self.conv1` lines:

```python
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, act=F.relu, dropout_rate: float = 0.3):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        ############# SOLUTION CODE
        self.embed = nn.Embedding(len(dataset.atoms()), embedding_dim=2)
        self.conv1 = Conv(2, hidden_channels)
        ############# SOLUTION CODE
        # rest of code
```

The last cell, which plots the embeddings, can be done as such:

```python
embed_df = pd.DataFrame(model.embed.weight.cpu().detach().numpy(), index=dataset.atoms(), columns=['x', 'y'])
sns.scatterplot(embed_df, x='x', y='y', s=5)
for i in range(len(dataset.atoms())):
    plt.text(embed_df['x'][i], embed_df['y'][i], dataset.atoms()[i])
```

The solution in class using

```py
embed_df = pd.DataFrame(model.embed.weight.cpu().detach().numpy(), index=dataset.atoms(), columns=['x', 'y'])
sns.scatterplot(embed_df, x='x', y='y', s=5)
for index, row in df.iterrows():
    plt.text(row['x'], row['y'], index)
```

also works.

I didn't find anything particularly interesting in my embedding plot.
