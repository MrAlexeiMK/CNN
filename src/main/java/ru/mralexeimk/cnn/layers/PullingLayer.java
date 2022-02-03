package ru.mralexeimk.cnn.layers;

import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.models.Matrix;
import ru.mralexeimk.cnn.models.Matrix3D;
import ru.mralexeimk.cnn.other.ActivationFunType;
import ru.mralexeimk.cnn.other.LayerType;
import ru.mralexeimk.cnn.other.PullingType;

import java.io.Serializable;
import java.util.ArrayList;

public class PullingLayer extends Layer implements Serializable {
    private PullingType pullingType;
    public PullingLayer(int sizeX, int sizeY, int sizeD, PullingType pullingType) {
        super(sizeX, sizeY, sizeD);
        this.pullingType = pullingType;
        setLayerType(LayerType.PULLING);
    }

    public PullingLayer(int sizeX, int sizeY, int sizeD, PullingType pullingType, ActivationFunType activationFunType) {
        this(sizeX, sizeY, sizeD, pullingType);
        setActivationFunType(activationFunType);
    }

    public PullingType getPullingType() {
        return pullingType;
    }

    public void setPullingType(PullingType pullingType) {
        this.pullingType = pullingType;
    }

    @Override
    public boolean toDefault() {
        if(nextLayer instanceof FilterLayer) {
            int kX = getSizeX()+1-nextLayer.getSizeX();
            int kY = getSizeY()+1-nextLayer.getSizeY();
            biases = new ArrayList<>();
            W = new Matrix3D(kX, kY, 0);
            for(int i = 0; i < nextLayer.getSizeD(); ++i) {
                W.add(new Matrix(kX, kY,
                        -1 / Math.sqrt(nextLayer.getSizeD()),
                        1 / Math.sqrt(nextLayer.getSizeD())));
                biases.add(0.0);
            }
            return true;
        }
        if(nextLayer instanceof NeuronsLayer || nextLayer instanceof OutputLayer) {
            return true;
        }
        return false;
    }

    public int getDiv() {
        return nextLayer.getSizeD() / getSizeD();
    }

    @Override
    public void doStep() {
        if(nextLayer instanceof FilterLayer) {
            nextLayer.setData(activationFun(data.getConvertByKernel(W, 1).sum(biases)));
        }
        else if(nextLayer instanceof NeuronsLayer || nextLayer instanceof OutputLayer) {
            nextLayer.setData(data.getConvertToLine());
        }
    }
}
