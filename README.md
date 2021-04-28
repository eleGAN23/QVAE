# QVAE
Official Pytorch implementation of [A Quaternion-Valued Variational Autoencoder](https://arxiv.org/abs/2010.11647) (QVAE) accepted as conference paper at ICASSP 2021.

Eleonora Grassucci, Danilo Comminiello, and Aurelio Uncini.

### Abstract
Deep probabilistic generative models have achieved incredible success in many fields of application. Among such models, variational autoencoders (VAEs) have proved their ability in modeling a generative process by learning a latent representation of the input. In this paper, we propose a novel VAE defined in the quaternion domain, which exploits the properties of quaternion algebra to improve performance while significantly reducing the number of parameters required by the network. The success of the proposed quaternion VAE with respect to traditional VAEs relies on the ability to leverage the internal relations between quaternion-valued input features and on the properties of second-order statistics which allow to define the latent variables in the augmented quaternion domain. In order to show the advantages due to such properties, we define a plain convolutional VAE in the quaternion domain and we evaluate it in comparison with its real-valued counterpart on the CelebA face dataset.


### Training

To run QVAE training, download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), install `requirements.txt` and type:
```
python train_qvae.py
```
Once trained the model, to generate new samples and reconstructions from the test set, type:
```
python generation.py
```
specify `--QVAE=True` to generate from QVAE.

Quaternion convolutions are borrowed from [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) by Titouan Parcollet.


### Cite

Plese cite our work if you found it useful:

Eleonora Grassucci, Danilo Comminiello, and Aurelio Uncini, " A Quaternion-Valued Variational Autoencoder", in <i>IEEE Int. Conf. on Acoust., Speech and Signal Process. (ICASSP)</i>, Toronto, Canada, Jun. 6-11, 2021.

```
@Conference{GrassucciICASSP2021,
  author =    {Grassucci, E. Comminiello, D. and Uncini, A.},
  title =     {A Quaternion-Valued Variational Autoencoder},
  booktitle = {IEEE Int. Conf. on Acoust., Speech and Signal Process. (ICASSP)},
  address = {Toronto, Canada},
  month = jun,
  year =      {2021},
}
```

#### Interested in Quaternion Generative Models?

Check also the Quaternion Generative Adversarial Network [[Paper](https://arxiv.org/pdf/2104.09630.pdf)] [[GitHub](https://github.com/eleGAN23/QGAN)].

