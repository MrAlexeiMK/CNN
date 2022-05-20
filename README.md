# CNN
 Convolutional Neural Network Library (<b><a href="https://github.com/MrAlexeiMK/CNN/tree/main/src/main/java/ru/mralexeimk/cnn">Sources</a></b>)
 
<b>Use case:</b>  
Layers configuration declaration:
```
private static final List<Layer> convLayers = new ArrayList<>(List.of(
        new InputLayer(28, 28, 1, ActivationFunType.SIGMOID), //Input Layer with 28x28x1 sizes
        new FilterLayer(24, 24, 8), //Filter Layer (result of kernel convolutional on InputLayer)
        new PullingLayer(12, 12, 8, PullingType.AVERAGE, ActivationFunType.SIGMOID), //Image reduction by 2 times
        new FilterLayer(8, 8, 16),
        new PullingLayer(4, 4, 16, PullingType.AVERAGE),
        new NeuronsLayer(256, ActivationFunType.SIGMOID), //NeuronsLayer with 256 cells
        new OutputLayer(10)) //OutputLayer with 10 outputs
);
```  
  
Creating Neural Network and test on 'mnist dataset':
```
NeuralNetwork nn = new NeuralNetwork("mnist", convLayers, 0.05); //'mnist'.w - file that saves the weights, convLayer - Layers Configuration, 0.05 - Learning Rate
nn.train(DataExtractor.extractFromFile("/train/mnist_train_100.csv", 28, 28, 1, 10), 10, true); //train Neural Network
nn.printWeights(); //Print Weights
nn.test(DataExtractor.extractFromFile("/test/mnist_test.csv", 28, 28, 1, 10), true); //test Neural Network
```
