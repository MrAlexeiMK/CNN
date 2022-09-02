package ru.mralexeimk.cnn.layers;

import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.models.Matrix;
import ru.mralexeimk.cnn.models.Matrix3D;
import ru.mralexeimk.cnn.enums.ActivationFunType;
import ru.mralexeimk.cnn.enums.LayerType;

import java.io.Serializable;
import java.util.ArrayList;

public class NeuronsLayer extends Layer implements Serializable {
    public NeuronsLayer(int units) {
        super(1, units, 1);
        setLayerType(LayerType.NEURONS);
    }

    public NeuronsLayer(int units, ActivationFunType activationFunType) {
        this(units);
        setActivationFunType(activationFunType);
    }

    public int getUnits() {
        return getSizeY();
    }

    @Override
    public boolean toDefault() {
        if(nextLayer instanceof NeuronsLayer || nextLayer instanceof OutputLayer) {
            biases = new ArrayList<>();
            biases.add(0.0);
            W = new Matrix3D(new Matrix(getSizeY(), nextLayer.getSizeY(),
                    -1 / Math.sqrt(nextLayer.getSizeY()),
                    1 / Math.sqrt(nextLayer.getSizeY())
            ));
            return true;
        }
        return false;
    }

    @Override
    public void doStep() {
        if(nextLayer instanceof NeuronsLayer || nextLayer instanceof OutputLayer) {
            nextLayer.setData(activationFun(W.getConvertByMultiply(data).sum(biases)));
        }
    }
}
