package ru.mralexeimk.cnn.layers;

import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.other.ActivationFunType;
import ru.mralexeimk.cnn.other.LayerType;

import java.io.Serializable;

public class OutputLayer extends Layer implements Serializable {
    public OutputLayer(int units) {
        super(1, units, 1);
        setLayerType(LayerType.OUTPUT);

    }

    public int getUnits() {
        return getSizeY();
    }

    @Override
    public boolean toDefault() {
        return true;
    }
}
