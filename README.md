## Installation

Recommended environment to run code is to use Ubuntu 22.04.1 LTS with a conda environment

# Install miniconda

See the [official Miniconda page](https://conda.io/en/latest/miniconda.html).

# Setup a Conda environment

Create a new Conda environment named `cad`.

```bash
$ conda create --name cad python=3.8
```

# Activate the environment

To activate your conda environment

```bash
$ conda activate cad
```

# Install required python packages

Make sure your your in the path of the project: 

```bash
$ cd /path/to/assigment/directory
```

Now you can install the required python packages

```bash
$ pip install -r requirements.txt
```

## Data

LIDAR data is actually just a set of positions estimated from a separate scan-matching (ICP), 
so we can insert it into our solver as another position measurement, just as we do for GNSS. 
However, the LIDAR frame is not the same as the frame shared by the
IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
known extrinsic calibration rotation matrix R_li and translation vector t_i_li.

## Explore Data

```bash
$ python plot.py
```
## Compilation & Run


```bash
$ python ekf.py
```
## What to do

- Complete the missing TODO parts in ekf.py
- Make use of the tools in plot.py to visualize your results (estimated state vs ground truth)
- Write report of max two pages discussing the results you get

## Submission 

- Your submission must include the report, the implementation and the data, all in one zip file
- Students can work in groups of max two but each student must submit individually in Moodle

## Evaluation

Your work will be evaluated based on the following:

- Quality of code your deliver
- Quality of state estimation results compared to ground truth 
- Discussion of your results and quality of your report

