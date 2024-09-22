# MLOps course

## Build docker image with jupyter lab and miniconda

```shell
docker build -t jupyter-lab-anaconda .
```

## Start Jupyter lab

```shell
docker run --rm -i -t -v $(pwd):/home/jovyan/work/mlops -p 8888:8888 jupyter-lab-anaconda
```
