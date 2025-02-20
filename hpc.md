# HPC Environment Configuration Instructions

Any time you see `[brackets]`, that means you need to replace that with
something.

## Login

```bash
ssh -P 222 [your-username]@login-theia.rc.sc.edu
```

You may need to now do a 2FA prompt on your phone.

You'll know you've succeeded when you see a prompt that looks like

```bash
rcstud123@login001 ~
```

Clusters have many distinct nodes: the login node is where you authenticate and
request a node to do the actual computing power on. It is **_not_** for running
jobs, just for connecting you to a different node to work in.

To request an interactive development node with a GPU, run

```bash
idev_gpu
```

Then, once there's a freed-up node, you'll be sent to a different session. Your
prompt will now look like

```bash
[rcstud12@node030 ~]$
```

This is where you can do development.

### Running the Container

The Deep Graph Library (DGL) used for GNNs can't be compiled directly on the
cluster. Instead, a special _container_ has been prepared that has this already
loaded.

First, we load the necessary modules: these are pieces of software we can load
on the cluster. CUDA is what makes the graphics card work, and Singularity is
the software that makes the container work.

```bash
module load singularity/4.1.4 cuda/12.1
```

Now we open the container.

```bash
singularity run --nv /home/camp/sing/dgl_24.01-py3.sif
```

The prompt should now read

```bash
Singularity>
```

### Loading Training Data

Normally, you would copy files from your computer using the `scp` command:

```bash
scp -r -P 222 [my/folder] [your-username]@login-theia.rc.sc.edu:~/[destination]
```

This will give you a 2FA prompt.

To circumvent any difficulties, we can instead download our training data from
the Internet directly.

```bash
wget https://github.com/nicholas-miklaucic/ai4sci-materials-2024/raw/main/alignn/GNN_training_data.zip
unzip GNN_training_data.zip
```

### Downloading ALIGNN

We will use the training scripts in the GitHub repository, so we want to
download the newest version of ALIGNN directly.

```bash
git clone https://github.com/usnistgov/alignn
```

We should now have two folders in our home directory: an `alignn/` folder, and a
`GNN_training_data` folder. You can check yourself by running `ls`.

Normally we would work inside a virtual environment using
`conda create -n [name] python=3.10; conda activate [name]`, as you can see in
the ALIGNN README. We're already working inside of a container, however, so
we'll be skipping that step.

Go into the `alignn/` directory by executing `cd alignn`. Install the necessary
packages by running `pip install -e .`: this installs the current directory and
its dependencies as an **e**ditable package.

### Executing ALIGNN

To work around a bug, we need to set an environment variable:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Training ALIGNN

Now we can run ALIGNN using a config file and training data from the folders we
downloaded:

```bash
python alignn/train_alignn.py --config_name=/home/[your_user]/GNN_training_data/ALIGNN_heat_capacity_data/config_example.json --root_dir=/home/[your_user]/GNN_training_data/ALIGNN_heat_capacity_data/ --output_dir=/home/[your_user]/results/heat
```

You should start seeing the loss go down as it trains.

### Submitting a Batch Job

We don't want to have to babysit our jobs. If something will take a long time,
ideally we can just start it and then come back when it's done.

We can take these commands and move them to a file we can ask the HPC to run.

Copy the necessary scripts into your home directory:

```bash
cp /home/miklaucn/run_alignn.sh ~
cp /home/miklaucn/alignn.slurm ~
```

We'll make two directories to store our logs and results. `cd`, without any
arguments, goes to the home directory.

```bash
cd
mkdir jobs
mkdir results
```

Now, we need to edit these files to replace `miklaucn` with your username:

```bash
nano ~/run_alignn.sh
nano ~/alignn.slurm
```

To keep our home directory clean, we can put the logs and results in separate
folders:

When that's done, we can ask the HPC to run our job:

```bash
sbatch alignn.slurm
```

To check how our job is doing, we can see the queue with

```bash
squeue
```

If you see that the status is `R`, that means it's running. It might take a few
minutes.

Ideally, eventually you'll see that `squeue` no longer shows your `alignn` job.
That means it should have worked. You can see what the logs were by first
looking at your `logs/` directory:

```bash
ll logs/
```

If you've submitted multiple times, find the most recent log name. Then do

```bash
cat logs/log-[LOG_ID].node[NODE_ID].out
```

To see any errors:

```
cat logs/log-[LOG_ID].node[NODE_ID].err
```

### Copying Results

At this point, you may be interested in getting the results off the HPC so you
can visualize and analyze them more comfortably.

`scp` and `rsync` should work, as above, but I needed a workaround.

First, zip the results folder:

```bash
zip -r res results/*
```

Now you should have a `res.zip` file.

Then

```bash
curl bashupload.com -T res.zip
```

When it's done, go to the URL that it prints out in your browser. Download that
results file.
