package ru.mralexeimk.cnn.models;

import lombok.Data;
import ru.mralexeimk.cnn.layers.*;
import ru.mralexeimk.cnn.enums.ActivationFunType;
import ru.mralexeimk.cnn.enums.Direction;
import ru.mralexeimk.cnn.other.ExtractedData;
import ru.mralexeimk.cnn.enums.PullingType;

import java.io.*;
import java.util.*;

@Data
public class NeuralNetwork implements Serializable {
    private String id;
    private List<Layer> layers;
    private double learningRate;

    private final String defId = "default";
    private final List<Layer> defLayers = new ArrayList<>(List.of(
            new InputLayer(28, 28, 1, ActivationFunType.SIGMOID),
            new FilterLayer(24, 24, 8),
            new PullingLayer(12, 12, 8, PullingType.AVERAGE, ActivationFunType.SIGMOID),
            new FilterLayer( 8, 8, 16),
            new PullingLayer(4, 4, 16, PullingType.AVERAGE),
            new NeuronsLayer(256, ActivationFunType.SIGMOID),
            new OutputLayer(10)
    ));
    private final double defLearningRate = 0.05;

    public NeuralNetwork() {
        toDefault();
    }

    public NeuralNetwork(String id) {
        loadWeights(id);
    }

    public NeuralNetwork(String id, double learningRate) {
        loadWeights(id);
    }

    public NeuralNetwork(String id, List<Layer> layers, double learningRate) {
        load(id, layers, learningRate);
    }

    public boolean check() {
        for(Layer layer : layers) {
            Layer next = layer.getNextLayer();
            if(next != null) {
                if(layer instanceof FilterLayer && next instanceof PullingLayer) {
                    if(layer.getSizeX()/next.getSizeX() != layer.getSizeY()/next.getSizeY()) return false;
                }
                if(layer instanceof PullingLayer && (next instanceof NeuronsLayer || next instanceof OutputLayer)) {
                    if(next.getSizeY() != layer.getSizeX()*layer.getSizeY()*layer.getSizeD()) return false;
                }
            }
        }
        return true;
    }

    public void load(String id, List<Layer> layers, double learningRate) {
        this.id = id;
        this.layers = layers;
        this.learningRate = learningRate;
        init();
    }

    public InputLayer getInputLayer() {
        return (InputLayer)layers.get(0);
    }

    public OutputLayer getOutputLayer() {
        return (OutputLayer)layers.get(layers.size()-1);
    }

    public void setInputLayerData(Matrix3D m) {
        getInputLayer().setData(m);
    }

    public void setInputLayerData(List<Matrix> m) {
        setInputLayerData(new Matrix3D(m));
    }

    public Matrix getOutputLayerData() {
        return getOutputLayer().getData().getMatrix();
    }

    public void evaluate() {
        for(Layer layer : layers) {
            layer.doStep();
        }
    }

    public String getConfiguration() {
        String res = "";
        for(Layer layer : layers) {
            res += layer.toString()+"\n";
        }
        return res;
    }

    public void printData() {
        System.out.println("Data:");
        for(Layer layer : layers) {
            System.out.println(layer.toString());
            layer.getData().print();
        }
    }

    public void printWeights() {
        System.out.println("Weights:");
        for(Layer layer : layers) {
            System.out.println(layer.toString());
            if(layer.getW() != null) {
                layer.getW().print();
                if(!layer.getBiases().isEmpty()) {
                    System.out.println("  Biases:");
                    for (Double d : layer.getBiases()) {
                        System.out.print(d + "|");
                    }
                    System.out.println();
                }
            }
        }
    }

    public void printDataShapes(boolean d3) {
        for(Layer layer : layers) {
            if(d3) {
                layer.getData().printShapes();
            }
            else layer.getData().get(0).printShapes();
        }
    }

    public void printWeightShapes() {
        for(Layer layer : layers) {
            if(layer.getW() != null) {
                layer.getW().printShapes();
            }
        }
    }

    public void printShapes(boolean d3) {
        System.out.println("DATA:");
        printDataShapes(d3);
        System.out.println("WEIGHTS:");
        printWeightShapes();
    }

    public void init() {
        for(int i = 0; i < layers.size(); ++i) {
            if(i >= 1) layers.get(i).setPrevLayer(layers.get(i-1));
            if(i < layers.size()-1) layers.get(i).setNextLayer(layers.get(i+1));
            if(!layers.get(i).toDefault()) {
                throw new RuntimeException("Incorrect NeuralNetwork structure! " +
                        "Problem with: "+layers.get(i).getLayerType().toString());
            }
        }
        if(!(layers.get(0) instanceof InputLayer)) throw new RuntimeException("InputLayer not found!");
        if(!(layers.get(layers.size()-1) instanceof OutputLayer)) throw new RuntimeException("OutputLayer not found!");
        if(!check()) throw new RuntimeException("Incorrect NeuralNetwork structure");
    }

    public synchronized void train(Matrix3D input, Matrix target) {
        setInputLayerData(input);
        evaluate();
        Matrix outputs = getOutputLayerData().clone();
        Matrix3D errors = new Matrix3D(target.diff(outputs));
        for(int i = layers.size()-2; i >= 0; --i) {
            Layer layer = layers.get(i);
            Layer next = layer.getNextLayer();
            if(layer instanceof NeuronsLayer nl) {
                Matrix I = nl.getData().getMatrix();
                Matrix O = next.getData().getMatrix();
                Matrix error = errors.getMatrix();
                Matrix dif = MatrixExtractor.getMultiply(error, O)
                        .multiply(O.getNegative().sum(1))
                        .multiply(I.getTransposed())
                        .multiply(learningRate);
                double biasError = dif.getSum();
                nl.setBias(0, nl.getBias(0)+biasError);
                nl.getW().getMatrix().sum(dif);
                errors = new Matrix3D(nl.getW().getMatrix().getTransposed().multiply(error));
            }
            else if(layer instanceof InputLayer il) {
                if(next instanceof NeuronsLayer nl) {
                    Matrix I = il.getData().getMatrix();
                    Matrix O = nl.getData().getMatrix();
                    Matrix error = errors.getMatrix();
                    Matrix dif = error
                            .multiply(O)
                            .multiply(O.getNegative().sum(1))
                            .multiply(I.getTransposed())
                            .multiply(learningRate);
                    double biasError = dif.getSum();
                    il.setBias(0, nl.getBias(0)+biasError);
                    il.getW().getMatrix().sum(dif);
                }
                else if(next instanceof FilterLayer fl) {
                    Matrix3D I = il.getData();
                    Matrix3D O = fl.getData();
                    int width = il.getW().getN(), height = il.getW().getM();
                    for(int z = 0; z < O.getD(); ++z) {
                        double biasError = 0;
                        for (int y = 0; y < O.getM(); ++y) {
                            for (int x = 0; x < O.getN(); ++x) {
                                double o = O.get(x, y, z);
                                double e = errors.get(x, y, z);
                                Matrix I_sum = I.get(0).clone();
                                for(int k = 1; k < I.getD(); ++k) I_sum.sum(I.get(k));
                                Matrix Ik = I_sum.getSubMatrix(x, y, width, height);
                                Matrix dif = Ik.multiply(e * o * (1 - o) * learningRate);
                                biasError += dif.getSum();
                                il.getW().get(z).sum(dif);
                            }
                        }
                        biasError /= O.getN()*O.getM();
                        il.setBias(z, il.getBias(z) + biasError);
                    }
                }
            }
            else if(layer instanceof PullingLayer pl) {
                if(next instanceof FilterLayer fl) {
                    Matrix3D I = pl.getData();
                    Matrix3D O = fl.getData();
                    int width = pl.getW().getN(), height = pl.getW().getM();
                    Matrix3D nextErrors = new Matrix3D(I.getN(), I.getM(), I.getD());
                    for(int z = 0; z < O.getD(); ++z) {
                        double biasError = 0;
                        int index = z/pl.getDiv();
                        for(int y = 0; y < O.getM(); ++y) {
                            for(int x = 0; x < O.getN(); ++x) {
                                double o = O.get(x, y, z);
                                double e = errors.get(x, y, z);
                                Matrix Ik = I.get(index).getSubMatrix(x, y, width, height);
                                Matrix difE = nextErrors.get(index).getSubMatrix(x, y, width, height).sum(e);
                                nextErrors.get(index).replace(x, y, difE);

                                Matrix dif = Ik.multiply(e*o*(1-o)*learningRate);
                                biasError += dif.getSum();
                                pl.getW().get(z).sum(dif);
                            }
                        }
                        for(int y = 0; y < I.getM(); ++y) {
                            for(int x = 0; x < I.getN(); ++x) {
                                int w = Math.min(x+1, width);
                                int h = Math.min(y+1, height);
                                nextErrors.get(index).set(x, y, nextErrors.get(index).get(x, y)/(w*h));
                            }
                        }
                        biasError /= O.getN()*O.getM();
                        pl.setBias(z, pl.getBias(z)+biasError);
                    }
                    errors = nextErrors;
                }
                else if(next instanceof NeuronsLayer) {
                    errors.expandLine(pl.getSizeX(), pl.getSizeY(), pl.getSizeD());
                }
            }
            else if(layer instanceof FilterLayer fl) {
                if(next instanceof PullingLayer) {
                    errors.increase(fl.getDiv(), fl.getDiv());
                }
            }
        }
    }

    public Matrix query(Matrix3D input) {
        setInputLayerData(input);
        evaluate();
        return getOutputLayerData();
    }

    public int queryMax(Matrix3D input) {
        Matrix matrix = query(input);
        int res = 0;
        double max = 0;
        for(int i = 0; i < matrix.getM(); ++i) {
            if(matrix.get(0, i) > max) {
                max = matrix.get(0, i);
                res = i;
            }
        }
        return res;
    }

    public List<Double> query(List<Double> m, Direction direction) {
        return query(new Matrix3D(m, direction)).toList();
    }

    public void toDefault() {
        load(defId, defLayers, defLearningRate);
    }

    public void train(ExtractedData data, int epochs, boolean debug) {
        if(debug) System.out.println("Starting training...");
        int size = data.getLen()*epochs;
        for(int i = 0; i < epochs; ++i) {
            for(int j = 0; j < data.getLen(); ++j) {
                train(data.getInputs().get(j), data.getOutputs().get(j));
                if(debug) System.out.println(String.format("%.3f", 100*(double)(i*data.getLen()+j)/size) + "%");
            }
        }
    }

    public double test(ExtractedData data, boolean debug) {
        int countCorrect = 0;
        for(int i = 0; i < data.getLen(); ++i) {
            int label = queryMax(data.getInputs().get(i));
            int correct = data.getOutputs().get(i).getMaxIndex();
            if(correct == label) ++countCorrect;
            if(debug) System.out.println("Correct: " + correct + ", Output: " + label);
        }
        double res = (double)100*countCorrect/data.getLen();
        if(debug) System.out.println(res + "%");
        return res;
    }

    public void save() {
        saveWeights(getId());
    }

    public synchronized void saveWeights(String id) {
        this.id = id;
        File file = new File("weights/"+id+".w");
        file.getParentFile().mkdirs();
        try {
            file.createNewFile();
        } catch (Exception ignored) {}

        try (FileOutputStream f = new FileOutputStream(file.getPath()); ObjectOutputStream o = new ObjectOutputStream(f)) {
            o.writeObject(this);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadWeights(String id) {
        this.id = id;
        File file = new File("weights/"+id+".w");
        try(FileInputStream fi = new FileInputStream(file.getPath()); ObjectInputStream oi = new ObjectInputStream(fi)) {
            NeuralNetwork nn = (NeuralNetwork) oi.readObject();
            load(nn.getId(), nn.getLayers(), nn.getLearningRate());
        } catch (Exception e) {
            toDefault();
        }
    }
}
