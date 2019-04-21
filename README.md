# Stable NALU

### Install

```bash
python3 setup.py develop
```

### Run experiments

Here are 4 experiments, they correspond to the experiments in the NALU paper.

```
python3 experiments/simple_function_static.py --help # 4.1 (static)
python3 experiments/simple_function_recurrent.py --help # 4.1 (recurrent)
python3 experiments/sequential_mnist.py --help # 4.2
python3 experiments/number_translation.py --help # 4.3
```

Example, where NALU rarely works:

```
python3 experiments/simple_function_static.py --layer-type NALU --verbose
```

Example, with an improved NALU:

```
python3 experiments/simple_function_static.py --layer-type RegualizedLinearNALU --nalu-bias --nalu-gate regualized --verbose
```

The `--verbose` logs network internal measures to the tensorboard. You can access the tensorboard with:

```
tensorboard --logdir tensorboard
```

You might see issues if you rerun an experiment without deleting the previuse experiment results from `./tensorboard`.

### Internal tests

```bash
python3 setup.py test
```