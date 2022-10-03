# **Fine-tuning**

  - TPU or GPU ? For fine-tuning it doesn\'92t really make sense o jump into TPUs because the models are fine-tuned pretty quickly on GPU, usually (1-2 days), so the actual cost that's saved per training run is negligible. Also is much easier to experiment/debug in PyTorch then with FLAX/JAX (PyTorch has more documentation and has better support for GPU)
  - Optimizations:
    -  fp16 or bfloat16? For many models that come from Google. Facebook, Microsoft, fine-tuning can be done in fp16 because these models have all been pretrained in PyTorch usually and also usually in fp16 and reduces model size.
    -  gradient_checkpointing can save a lot of memory during training, especially when the model has many layers. If the model still doesn’t fit into memory, first replace torch's native Adam optimizer with the 8bit adam optimizer which did in some cases save a lot of memory for me. If this also doesn’t work, make use of gradient_accumulation_steps.

# **Pre-training**

  - TPU or GPU ? Here is recommended using TPU because it can lead to significant speed-ups and cost savings: see results
  - PyTorch/XLA on TPU can work very well, but doesn't have great suport. FLAX on TPU is more robust and faster than PyTorch/XLA, but requires to know JAX/FLAX.
  - Optimizations:
    - for pre-training the same optimization tips apply then the ones written above

