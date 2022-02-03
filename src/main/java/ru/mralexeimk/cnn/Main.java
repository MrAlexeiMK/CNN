package ru.mralexeimk.cnn;

import ru.mralexeimk.cnn.layers.*;
import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.models.Matrix;
import ru.mralexeimk.cnn.models.Matrix3D;
import ru.mralexeimk.cnn.models.NeuralNetwork;
import ru.mralexeimk.cnn.other.ActivationFunType;
import ru.mralexeimk.cnn.other.PullingType;

import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final List<Layer> layers = new ArrayList<>(List.of(
            new InputLayer(28, 28, 1, ActivationFunType.SIGMOID),
            new FilterLayer(24, 24, 8),
            new PullingLayer(12, 12, 8, PullingType.AVERAGE, ActivationFunType.SIGMOID),
            new FilterLayer( 8, 8, 16),
            new PullingLayer(4, 4, 16, PullingType.AVERAGE),
            new NeuronsLayer(256, ActivationFunType.SIGMOID),
            new OutputLayer(10)
    ));
    public static void main(String[] args) {
        mnistTest();
    }

    public static void mnistTest() {
        NeuralNetwork nn = new NeuralNetwork("mnist", layers, 0.01);
        nn.trainFromFile("/train/mnist_train_100.csv", 28, 28, 1, 100);
        nn.printData();
        nn.testFromFile("/test/mnist_test_10.csv", 28, 28, 1);
        nn.save();
    }
}
