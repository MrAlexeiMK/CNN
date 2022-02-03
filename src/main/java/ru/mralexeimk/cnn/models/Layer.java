package ru.mralexeimk.cnn.models;

import lombok.Data;
import ru.mralexeimk.cnn.layers.NeuronsLayer;
import ru.mralexeimk.cnn.other.ActivationFunType;
import ru.mralexeimk.cnn.other.LayerType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Data
public abstract class Layer implements Serializable {
    protected Matrix3D data;
    protected Matrix3D W;
    protected List<Double> biases;
    protected Layer nextLayer, prevLayer;
    protected LayerType layerType;
    protected ActivationFunType activationFunType;

    public Layer(int sizeX, int sizeY, int sizeD) {
        data = new Matrix3D(sizeX, sizeY, sizeD);
        biases = new ArrayList<>();
        activationFunType = ActivationFunType.NONE;
    }

    public Layer(int sizeX, int sizeY, int sizeD, ActivationFunType activationFunType) {
        this(sizeX, sizeY, sizeD);
        this.activationFunType = activationFunType;
    }

    public double activationFun(double x) {
        return activationFunType.getActivationFunInterface().activationFun(x);
    }

    public Matrix activationFun(Matrix m) {
        Matrix res = m.clone();
        double sum = 0;
        if(activationFunType == ActivationFunType.SOFT_MAX) {
            for(int x = 0; x < m.getN(); ++x) {
                for(int y = 0; y < m.getM(); ++y) {
                    sum += Math.exp(m.get(x, y));
                }
            }
        }
        for(int x = 0; x < m.getN(); ++x) {
            for(int y = 0; y < m.getM(); ++y) {
                res.set(x, y, activationFun(m.get(x, y)));
                if (activationFunType == ActivationFunType.SOFT_MAX) {
                    if(sum != 0) {
                        res.set(x, y, res.get(x, y) / sum);
                    }
                    else {
                        throw new ArithmeticException("SOFT_MAX sum cannot be 0!");
                    }
                }
            }
        }
        return res;
    }
    public Matrix3D activationFun(Matrix3D m) {
        Matrix3D res = m.clone();
        for(int i = 0; i < m.getD(); ++i) {
            res.set(i, activationFun(m.get(i)));
        }
        return res;
    }

    public void setData(Matrix3D data) {
        if(this.data.getShapes().equals(data.getShapes()) &&
                this.data.get(0).getShapes().equals(data.get(0).getShapes())) {
            this.data = data;
        }
        else {
            throw new RuntimeException("Incorrect setData() in "+ layerType.toString()+"\n"+
                    data.getShapes()+" != "+this.data.getShapes()+" (Matrix3D)\n"+
                    "or\n"+
                    data.get(0).getShapes()+" != "+this.data.get(0).getShapes()+" (Matrix)");
        }
    }

    public int getSizeD() {
        return data.getD();
    }

    public int getSizeX() {
        return data.getN();
    }

    public int getSizeY() {
        return data.getM();
    }

    public String toString() {
        return layerType.toString()+": "+getSizeX()+"x"+getSizeY()+"x"+getSizeD();
    }

    public double getBias(int pos) {
        return biases.get(pos);
    }

    public void setBias(int pos, double val) {
        biases.set(pos, val);
    }

    public boolean toDefault() {
        return false;
    }

    public void doStep() {}
}
