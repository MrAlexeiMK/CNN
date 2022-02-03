package ru.mralexeimk.cnn.other;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@Data
@AllArgsConstructor
public class Pair<T, V> implements Serializable {
    private T first;
    private V second;

    public String toString() {
        return first+", "+second;
    }
}
