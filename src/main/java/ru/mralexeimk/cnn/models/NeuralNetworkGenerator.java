package ru.mralexeimk.cnn.models;

import ru.mralexeimk.cnn.other.ExtractedData;
import ru.mralexeimk.cnn.other.NeuralNetworkParameters;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetworkGenerator {
    public static NeuralNetworkParameters genTheBestNeuralNetwork(ExtractedData trainData,
                                                                  ExtractedData testData,
                                                                  List<List<Layer>> layers,
                                                                  double lrFrom, double lrTo,
                                                                  int lrCount,
                                                                  int epochsFrom, int epochsTo) {
        NeuralNetworkParameters parameters = new NeuralNetworkParameters();
        System.out.println("Starting...");
        try(PrintWriter writer = new PrintWriter("log.txt", StandardCharsets.UTF_8)) {
            for (int i = 0; i < layers.size(); ++i) {
                for (int epochs = epochsFrom; epochs <= epochsTo; ++epochs) {
                    for (int j = 0; j < lrCount; ++j) {
                        double lr = lrFrom + new Random().nextDouble() * (lrTo - lrFrom);
                        NeuralNetwork nn = new NeuralNetwork("mnist", layers.get(i), lr);
                        nn.train(trainData, epochs, false);
                        double nextRes = nn.test(testData, false);
                        parameters.add(i, lr, epochs, nextRes);
                        String log = i + " - lr: " + lr + ", epochs: " + epochs + ", res: " + nextRes + "%";
                        System.out.println(log);
                        writer.println(log);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("===================================");
        return parameters;
    }
}
