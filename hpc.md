### Login

### Copying Data files
/home vs. /work

```bash
scp -r -P 222 Downloads/GNN_training_data miklaucn@login.rci.sc.edu:/work/miklaucn
```
You'll need to do the 2FA on your phone once you enter the password.

### Requesting a Node
timeshare

```bash
srun -n 2 -N 1 --time=01:00:00 --pty bash -i
```

### Setting up the Virtual Environment

```bash
module load python3/anaconda/2023.9
module load cuda/12.1
conda create --name alignn python=3.10
source activate alignn
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install alignn

cd /work/miklaucn
git clone https://github.com/usnistgov/alignn.git
python alignn/alignn/train.py -h
```
