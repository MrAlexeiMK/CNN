package ru.mralexeimk.cnn.layers;

import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.models.Matrix;
import ru.mralexeimk.cnn.models.Matrix3D;
import ru.mralexeimk.cnn.enums.ActivationFunType;
import ru.mralexeimk.cnn.enums.LayerType;

import java.io.Serializable;
import java.util.ArrayList;

public class InputLayer extends Layer implements Serializable {
    public InputLayer(int sizeX, int sizeY, int sizeD) {
        super(sizeX, sizeY, sizeD);
        setLayerType(LayerType.INPUT);
    }

    public InputLayer(int sizeX, int sizeY, int sizeD, ActivationFunType activationFunType) {
        this(sizeX, sizeY, sizeD);
        setActivationFunType(activationFunType);
    }

    @Override
    public boolean toDefault() {
        if(nextLayer instanceof NeuronsLayer) {
            biases = new ArrayList<>();
            biases.add(0.0);
            W = new Matrix3D(new Matrix(getSizeX()*getSizeY()*getSizeD(), nextLayer.getSizeY(),
                    -1 / Math.sqrt(nextLayer.getSizeY()),
                       1 / Math.sqrt(nextLayer.getSizeY())
            ));
            return true;
        }
        if(nextLayer instanceof FilterLayer) {
            int kX = getSizeX()-nextLayer.getSizeX()+1;
            int kY = getSizeY()-nextLayer.getSizeY()+1;
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
        return false;
    }

    @Override
    public Matrix3D getData() {
        if(nextLayer instanceof NeuronsLayer) {
            return data.getConvertToLine();
        }
        return data;
    }

    @Override
    public void doStep() {
        if(nextLayer instanceof NeuronsLayer) {
            nextLayer.setData(activationFun(W.getConvertByMultiply(data.getConvertToLine()).sum(biases)));
        }
        else if(nextLayer instanceof FilterLayer) {
            nextLayer.setData(activationFun(data.getConvertByMergeKernel(W).sum(biases)));
        }
    }
}
