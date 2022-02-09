package ru.mralexeimk.cnn.models;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.mralexeimk.cnn.other.ExtractedData;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataExtractor {
    public static ExtractedData extractFromFile(String path, int inputX, int inputY, int inputZ, int outputUnits) {
        ExtractedData data = new ExtractedData();
        try {
            File file = new File(DataExtractor.class.getResource(path).toURI());
            String line;
            Scanner sc = new Scanner(file);
            while (sc.hasNextLine()) {
                line = sc.nextLine();
                String[] spl = line.split(",");
                int target = Integer.parseInt(spl[0]);
                List<Double> listInputs = new ArrayList<>();
                for (int k = 1; k < spl.length; ++k) {
                    double t = Double.parseDouble(spl[k]);
                    t = (t / 255.0) * 0.99 + 0.01;
                    listInputs.add(t);
                }
                Matrix3D inputs = new Matrix3D(inputX, inputY, inputZ, listInputs);
                Matrix targets = new Matrix(1, outputUnits, 0.01);
                targets.set(0, target, 0.99);
                data.getInputs().add(inputs);
                data.getOutputs().add(targets);
            }
            sc.close();
        } catch (Exception e) {}
        return data;
    }
}
