# Lazy Mode Prototype

## Compile
```
python setup.py develop
```

## Run Example

```
python examples/simple_model.py --lazy --inference
```

## Trace Lazy Aten Functions
```
PYTORCH_LAZY_TRACE=trace_file python examples/simple_model.py --lazy --inference
```
