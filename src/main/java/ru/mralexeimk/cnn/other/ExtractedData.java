package ru.mralexeimk.cnn.other;


import lombok.Data;
import ru.mralexeimk.cnn.models.Matrix;
import ru.mralexeimk.cnn.models.Matrix3D;

import java.util.ArrayList;
import java.util.List;

@Data
public class ExtractedData {
    private List<Matrix3D> inputs;
    private List<Matrix> outputs;

    public ExtractedData() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
    }

    public int getLen() {
        return inputs.size();
    }
}
