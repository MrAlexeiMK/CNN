package ru.mralexeimk.cnn.enums;

import java.io.Serializable;

public enum ActivationFunType implements Serializable {
    SIGMOID(x -> 1.0/(1 + Math.exp(-x))),
    TANH(x -> (Math.exp(x) - Math.exp(-x))/(Math.exp(x) + Math.exp(-x))),
    RELU(x -> Math.max(0, x)),
    L_RELU(x -> Math.max(0.001*x, x)),
    SOFT_PLUS(x-> Math.log(1 + Math.exp(x))),
    SOFT_MAX(x -> Math.exp(x)),
    NONE(x -> x);

    private ActivationFunInterface activationFunInterface;

    ActivationFunType(ActivationFunInterface activationFunInterface) {
        this.activationFunInterface = activationFunInterface;
    }

    public ActivationFunInterface getActivationFunInterface() {
        return activationFunInterface;
    }

    public void setActivationFunInterface(ActivationFunInterface activationFunInterface) {
        this.activationFunInterface = activationFunInterface;
    }
}
