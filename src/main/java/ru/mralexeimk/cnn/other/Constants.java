package ru.mralexeimk.cnn.other;

import ru.mralexeimk.cnn.models.Matrix;

public class Constants {
    public static final double EPS = 1e-3;
    public static final Matrix IDENTITY_2D = new Matrix("1,0|0,1");
    public static final Matrix IDENTITY_3D = new Matrix("1,0,0|0,1,0|0,0,1");
}
