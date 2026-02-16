## Misc errors

### Data processing error

Comment out Line 299-306 in `.venv/lib/python3.10/site-packages/tensorflow_datasets/core/dataset_builder.py` to avoid the `AttributeError: 'MultiplexedPath' object has no attribute 'parts'` error (seems an issue with running python3.10; using `tensorflow_datasets==4.9.2` fixes this issue but disabling gcs does not work any more somehow)

### Quantization error in training

If using quantization in training, might need to modify Line 474 in `.venv/lib64/python3.10/site-packages/bitsandbytes/autograd/_functions.py` to `return output.clone` from `return output` ([related issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/736)).

### torch.compile + quantization

Quantization does not work well with torch.compile currently when running eval.
