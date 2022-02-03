package ru.mralexeimk.cnn.models;

import lombok.Data;
import ru.mralexeimk.cnn.layers.*;
import ru.mralexeimk.cnn.other.ActivationFunType;
import ru.mralexeimk.cnn.other.Direction;
import ru.mralexeimk.cnn.other.PullingType;

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
        Matrix3D errors = new Matrix3D(target.getMinus(outputs));
        for(int i = layers.size()-2; i >= 0; --i) {
            Layer layer = layers.get(i);
            Layer next = layer.getNextLayer();
            if(layer instanceof NeuronsLayer nl) {
                Matrix I = nl.getData().getMatrix();
                Matrix O = next.getData().getMatrix();
                Matrix error = errors.getMatrix();
                Matrix dif = error
                        .getMultiply(O)
                        .multiply(O.getNegative().sum(1))
                        .multiply(I.getTranspose())
                        .multiply(learningRate);
                double biasError = dif.getSum();
                nl.setBias(0, nl.getBias(0)+biasError);
                nl.getW().getMatrix().sum(dif);
                errors = new Matrix3D(nl.getW().getMatrix().getTranspose().multiply(error));
            }
            else if(layer instanceof InputLayer il) {
                if(next instanceof NeuronsLayer nl) {
                    Matrix I = il.getData().getMatrix();
                    Matrix O = nl.getData().getMatrix();
                    Matrix error = errors.getMatrix();
                    Matrix dif = error
                            .multiply(O)
                            .multiply(O.getNegative().sum(1))
                            .multiply(I.getTranspose())
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
                                Matrix Ek = nextErrors.get(index).getSubMatrix(x, y, width, height);
                                Matrix difE = Ik.getTranspose().multiply(e);
                                nextErrors.get(index).replace(x, y, Ek.sum(difE));

                                Matrix dif = Ik.multiply(e*o*(1-o)*learningRate);
                                biasError += dif.getSum();
                                pl.getW().get(z).sum(dif);
                            }
                        }
                        biasError /= O.getN()*O.getM();
                        pl.setBias(z, pl.getBias(z)+biasError);
                    }
                    errors = nextErrors;
                }
                else if(next instanceof NeuronsLayer) {
                    errors.resize(pl.getSizeX(), pl.getSizeY(), pl.getSizeD());
                }
            }
            else if(layer instanceof FilterLayer fl) {
                if(next instanceof PullingLayer) {
                    errors.saveExpand(fl.getDiv(), fl.getDiv());
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

    public void trainFromFile(String path, int inputX, int inputY, int inputZ, int epoch) {
        System.out.println("Starting training...");
        try {
            File file = new File(getClass().getResource(path).toURI());
            BufferedReader reader = new BufferedReader(new FileReader(file));
            int size = 0;
            while(reader.readLine() != null) ++size;
            reader.close();
            size *= epoch;
            String line;
            int j = 0;
            for (int i = 0; i < epoch; ++i) {
                Scanner sc = new Scanner(file);
                while (sc.hasNextLine()) {
                    line = sc.nextLine();
                    String[] spl = line.split(",");
                    int target = Integer.parseInt(spl[0]);
                    List<Double> listInputs = new ArrayList<>();
                    for (int k = 1; k < spl.length; ++k) {
                        double t = Double.parseDouble(spl[k]);
                        t = (t / 255.0) * 0.99 + 0.01;
                        listInputs.add(t);
                    }
                    Matrix3D inputs = new Matrix3D(inputX, inputY, inputZ, listInputs);
                    Matrix targets = new Matrix(1, getOutputLayer().getUnits(), 0.01);
                    targets.set(0, target, 0.99);
                    train(inputs, targets);
                    ++j;
                    System.out.println(String.format("%.3f", 100*(double)j/size) + "%");
                }
                sc.close();
            }
        } catch(Exception e){
            e.printStackTrace();
        }
    }

    public void testFromFile(String path, int inputX, int inputY, int inputZ) {
        try {
            File file = new File(getClass().getResource(path).toURI());
            Scanner sc = new Scanner(file);
            String line;
            int count = 0;
            int count_correct = 0;
            while(sc.hasNextLine()) {
                line = sc.nextLine();
                ++count;
                String[] spl = line.split(",");
                int correct = Integer.parseInt(spl[0]);
                List<Double> listInputs = new ArrayList<>();
                for(int i = 1; i < spl.length; ++i) {
                    double t = Double.parseDouble(spl[i]);
                    t = (t/255.0)*0.99 + 0.01;
                    listInputs.add(t);
                }
                Matrix3D inputs = new Matrix3D(inputX, inputY, inputZ, listInputs);
                int target = queryMax(inputs);
                System.out.println("Correct: " + correct + ", Output: " + target);
                if(correct == target) ++count_correct;
            }
            System.out.println((double)100*count_correct/count + "%");
            sc.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
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
