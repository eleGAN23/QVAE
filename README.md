# QVAE
Official Pytorch implementation of quaternion-valued variational autoencoder (QVAE).

To run QVAE training type on command line:

```
python train_midq.py
```

Once trained the model, to generate new samples and reconstructions from the test set, type:

```
python generation.py
```
specify `--QVAE=True` to generate from QVAE and not from VAE.
