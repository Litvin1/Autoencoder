# Autoencoder
We have trained an Autoencoder on the full data which contains parasitized and uninfected cell images.<br />
We used the encoder-network on 200 (100 of each class) samples and collect the embedded vectors, and used t-SNE algorithm to project thesevectors to a 2dim space.<br />
Then,  our goal was to build a FC-NN classifier that is capable of discriminating between the two types of cells. 
