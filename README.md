# QVAE
Official Pytorch implementation of quaternion-valued variational autoencoder (QVAE).

QVAE is evaluated on [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

To run QVAE training, download CelebA dataset and type:

```
python train_midq.py
```

Once trained the model, to generate new samples and reconstructions from the test set, type:

```
python generation.py
```
specify `--QVAE=True` to generate from QVAE and not from VAE.
