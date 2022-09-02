package ru.mralexeimk.cnn.layers;

import ru.mralexeimk.cnn.models.Layer;
import ru.mralexeimk.cnn.enums.LayerType;
import ru.mralexeimk.cnn.enums.PullingType;

import java.io.Serializable;

public class FilterLayer extends Layer implements Serializable {
    public FilterLayer(int sizeX, int sizeY, int sizeD) {
        super(sizeX, sizeY, sizeD);
        setLayerType(LayerType.FILTER);
    }

    @Override
    public boolean toDefault() {
        if(nextLayer instanceof PullingLayer) {
            return true;
        }
        return false;
    }

    public int getDiv() {
        return getSizeX() / nextLayer.getSizeX();
    }

    @Override
    public void doStep() {
        if(nextLayer instanceof PullingLayer pl) {
            if(pl.getPullingType() == PullingType.AVERAGE) {
                nextLayer.setData(data.getConvertByAveragePulling(getDiv()));
            }
            else if(pl.getPullingType() == PullingType.MAX) {
                nextLayer.setData(data.getConvertByMaxPulling(getDiv()));
            }
            else if(pl.getPullingType() == PullingType.MIN) {
                nextLayer.setData(data.getConvertByMinPulling(getDiv()));
            }
        }
    }
}
