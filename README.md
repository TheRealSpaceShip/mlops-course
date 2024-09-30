# MLOps course

## Build docker image with jupyter lab and miniconda

```shell
docker build -t jupyter-lab-miniconda .
```

## Start Jupyter lab

```shell
docker run --rm -i -t -v $(pwd):/home/jovyan/work -p 8888:8888 jupyter-lab-miniconda
```
