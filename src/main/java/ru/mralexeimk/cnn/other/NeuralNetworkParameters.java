package ru.mralexeimk.cnn.other;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Data
public class NeuralNetworkParameters implements Serializable {
    private List<Parameters> parametersList;

    public NeuralNetworkParameters() {
        parametersList = new ArrayList<>();
    }

    public void add(int index, double lr, int epochs, double res) {
        parametersList.add(new Parameters(index, lr, epochs, res));
    }

    @Data
    @AllArgsConstructor
    public static class Parameters implements Serializable {
        private int index;
        private double learningRate;
        private int epochs;
        private double res;

        public String toString() {
            return index+" - lr: "+learningRate+", epochs: "+epochs+", res: "+res+"%";
        }
    }

    public List<Parameters> getTop(int top) {
        parametersList.sort((p1, p2) -> Double.compare(p2.getRes(), p1.getRes()));
        return parametersList.stream().limit(top).collect(Collectors.toList());
    }

    public void printTop(int top) {
        List<Parameters> parameters = getTop(top);
        for(int i = 0; i < parameters.size(); ++i) {
            System.out.println(parameters.get(i).toString());
        }
    }
}
