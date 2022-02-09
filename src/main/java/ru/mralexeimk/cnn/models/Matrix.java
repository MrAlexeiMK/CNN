package ru.mralexeimk.cnn.models;

import ru.mralexeimk.cnn.other.Pair;

import java.io.Serializable;
import java.util.*;

public class Matrix implements Serializable {
    protected double[][] data;
    protected int N, M;

    public Matrix() {
        try (Scanner sc = new Scanner(System.in)) {
            N = sc.nextInt();
            M = sc.nextInt();
            data = new double[M][N];
            for (int y = 0; y < M; ++y) {
                for (int x = 0; x < N; ++x) {
                    data[y][x] = sc.nextDouble();
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public Matrix(int N, int M, double value) {
        this.N = N;
        this.M = M;
        data = new double[M][N];
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                data[y][x] = value;
            }
        }
    }

    public Matrix(int N, int M) {
        this(N, M, 0);
    }

    public Matrix(int N, int M, double from, double to) {
        this.N = N;
        this.M = M;
        data = new double[M][N];
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                data[y][x] = from + new Random().nextDouble() * (to - from);
            }
        }
    }

    public Matrix(Matrix m) {
        assign(m);
    }

    public Matrix(List<Double> vector) {
        N = 1;
        M = vector.size();
        data = new double[M][N];
        for(int y = 0; y < M; ++y) {
            data[y][0] = vector.get(y);
        }
    }

    public Matrix(String pattern, String separatorRows, String separatorColumns) {
        String[] arr = pattern.split(separatorRows);
        M = arr.length;
        N = arr[0].split(separatorColumns).length;
        data = new double[M][N];
        for(int y = 0; y < arr.length; ++y) {
            String[] nums = arr[y].split(separatorColumns);
            for(int x = 0; x < nums.length; ++x) {
                data[y][x] = Double.parseDouble(nums[x]);
            }
        }
    }

    public Matrix(String pattern) {
        this(pattern, "\\|", ",");
    }

    public List<Double> toList() {
        List<Double> res = new ArrayList<>();
        for(int y = 0; y < getM(); ++y) {
            for(int x = 0; x < getN(); ++x) {
                res.add(get(x, y));
            }
        }
        return res;
    }

    public void assign(Matrix m) {
        this.N = m.getN();
        this.M = m.getM();
        data = new double[M][N];
        for(int x = 0; x < m.getN(); ++x) {
            for(int y = 0; y < m.getM(); ++y) {
                data[y][x] = m.get(x, y);
            }
        }
    }

    public Matrix clone() {
        return new Matrix(this);
    }

    public int getN() {
        return N;
    }

    public int getM() {
        return M;
    }

    public double get(int x, int y) {
        return data[y][x];
    }

    public void set(int x, int y, double value) {
        data[y][x] = value;
    }

    public boolean isVector() {
        return N == 1;
    }

    public boolean isTransposeVector() {
        return M == 1;
    }

    public List<Double> getListLine(int y) {
        List<Double> res = new ArrayList<>();
        for(int x = 0; x < N; ++x) res.add(get(x, y));
        return res;
    }

    public void setLine(int y, List<Double> list) {
        for(int x = 0; x < N; ++x) data[y][x] = list.get(x);
    }

    public List<Double> getListColumn(int x) {
        List<Double> res = new ArrayList<>();
        for(int y = 0; y < M; ++y) res.add(get(x, y));
        return res;
    }

    public void setColumn(int x, List<Double> list) {
        for(int y = 0; y < M; ++y) data[y][x] = list.get(y);
    }

    public Matrix getNegative() {
        Matrix res = new Matrix(this);
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res.set(x, y, -get(x, y));
            }
        }
        return res;
    }

    public void replace(int xStart, int yStart, Matrix m) {
        for(int x = xStart; x < xStart+m.getN(); ++x) {
            for(int y = yStart; y < yStart+m.getM(); ++y) {
                set(x, y, m.get(x-xStart, y-yStart));
            }
        }
    }

    public double[][] getData() {
        return Arrays.stream(data).map(double[]::clone).toArray(double[][]::new);
    }

    public void expand(int addN, int addM) {
        N += addN;
        M += addM;
        double[][] temp = getData();
        data = new double[M][N];
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                if(x < N-addN && y < M-addM) {
                    set(x, y, temp[y][x]);
                }
                else {
                    set(x, y, 0);
                }
            }
        }
    }

    public void saveExpand(int kX, int kY) {
        Matrix res = new Matrix(N*kX, M*kY);
        for(int x = 0; x < res.getN(); ++x) {
            for(int y = 0; y < res.getM(); ++y) {
                res.set(x, y, get(x/kX, y/kY));
            }
        }
        assign(res);
    }

    public void erase(int removeN, int removeM) {
        N -= removeN;
        M -= removeM;
        double[][] temp = getData();
        data = new double[M][N];
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, temp[y][x]);
            }
        }
    }

    public void removeRows(int start, int end) {
        int lenRemove = (end-start+1);
        Matrix res = new Matrix(N, M-lenRemove);
        for(int y = 0; y < start; ++y) {
            for(int x = 0; x < N; ++x) {
                res.set(x, y, get(x, y));
            }
        }
        for(int y = end+1; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                res.set(x, y-lenRemove, get(x, y));
            }
        }
        assign(res);
    }

    public void removeColumns(int start, int end) {
        int lenRemove = (end-start+1);
        Matrix res = new Matrix(N-lenRemove, M);
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < start; ++x) {
                res.set(x, y, get(x, y));
            }
        }
        for(int y = 0; y < M; ++y) {
            for(int x = end+1; x < N; ++x) {
                res.set(x-lenRemove, y, get(x, y));
            }
        }
        assign(res);
    }

    public void removeRow(int index) {
        removeRows(index, index);
    }

    public void removeColumn(int index) {
        removeColumns(index, index);
    }


    public void removeLastRow() {
        removeRow(M-1);
    }

    public void removeLastColumn() {
        removeColumn(N-1);
    }

    public Matrix getSubMatrix(int xStart, int yStart, int width, int height) {
        Matrix res = new Matrix(width, height);
        for(int x = xStart; x < xStart+width; ++x) {
            for(int y = yStart; y < yStart+height; ++y) {
                res.set(x-xStart, y-yStart, get(x, y));
            }
        }
        return res;
    }

    public void toIdentity() {
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < M; ++y) {
                if (x != y) {
                    set(x, y, 0);
                } else set(x, y, 1);
            }
        }
    }

    public void setTrack(double value) {
        int size = Math.min(N, M);
        for(int i = 0; i < size; ++i) {
            set(i, i, value);
        }
    }

    public void swapLines(int l1, int l2) {
        List<Double> temp = new ArrayList<>(getListLine(l1));
        setLine(l1, getListLine(l2));
        setLine(l2, temp);
    }

    public void swapColumns(int c1, int c2) {
        List<Double> temp = new ArrayList<>(getListColumn(c1));
        setColumn(c1, getListColumn(c2));
        setColumn(c2, temp);
    }

    public void transpose() {
        Matrix res = new Matrix(M, N);
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < M; ++y) {
                res.set(y, x, get(x, y));
            }
        }
        assign(res);
    }


    public Matrix getTranspose() {
        Matrix res = new Matrix(this);
        res.transpose();
        return res;
    }

    public double getAverage() {
        double res = 0;
        for(int y = 0; y < getM(); ++y) {
            for(int x = 0; x < getN(); ++x) {
                res += get(x, y);
            }
        }
        res /= getN()*getM();
        return res;
    }

    public int toTriangularDown() {
        int swaps = 0;
        int size = Math.min(M, N);
        for (int L1 = 0; L1 < size; ++L1) {
            for (int L2 = L1 + 1; L2 < size; ++L2) {
                double l1 = get(L1, L1);
                double l2 = get(L1, L2);

                if (l2 == 0) continue;
                else if (l1 == 0) {
                    swapLines(L1, L2);
                    ++swaps;
                    l1 = get(L1, L1);
                    l2 = get(L1, L2);
                }
                double val = -(l2 / l1);
                set(L1, L2, 0);
                for (int x = L1 + 1; x < N; ++x) {
                    set(x, L2, get(x, L2) + get(x, L1) * val);
                }
            }
        }
        return swaps;
    }

    public int toTriangularUp() {
        int swaps = 0;
        int size = Math.min(M, N);
        for (int L1 = 1; L1 < size; ++L1) {
            for (int L2 = L1 - 1; L2 >= 0; --L2) {
                double l1 = get(L1, L1);
                double l2 = get(L1, L2);

                if (l2 == 0) continue;
                else if (l1 == 0) {
                    swapLines(L1, L2);
                    ++swaps;
                    l1 = get(L1, L1);
                    l2 = get(L1, L2);
                }
                double val = -(l2 / l1);
                set(L1, L2, 0);
                for (int x = L1 + 1; x < N; ++x) {
                    set(x, L2, get(x, L2) + get(x, L1) * val);
                }
            }
        }
        return swaps;
    }

    public double getTrack() {
        double ans = get(0, 0);
        for(int i = 1; i < Math.min(N, M); ++i) {
            ans *= get(i, i);
        }
        return ans;
    }

    public double getDeterminant() {
        Matrix A = new Matrix(this);
        int swaps = A.toTriangularDown();
        double ans = A.getTrack();
        return (swaps%2 == 0) ? ans : -ans;
    }

    public void toUnit() {
        toTriangularDown();
        toTriangularUp();
        int size = Math.min(N, M);
        for(int i = 0; i < size; ++i) {
            double el = get(i, i);
            if(el != 0 && el != 1) {
                for(int x = 0; x < N; ++x) {
                    set(x, i, get(x, i)/el);
                }
            }
        }
    }

    public void joinRight(Matrix m) {
        if(M == m.getM()) {
            expand(m.getN(), 0);
            replace(N-m.getN(), 0, m);
        }
    }

    public void joinBottom(Matrix m) {
        if(N == m.getN()) {
            expand(0, m.getM());
            replace(0, M-m.getM(), m);
        }
    }

    public Matrix getInverse() {
        if(N != M) throw new IndexOutOfBoundsException("Matrix must be square");
        Matrix E = new Matrix(N, M);
        E.setTrack(1);
        Matrix A = new Matrix(this);
        A.joinRight(E);
        A.toUnit();
        for(int i = 0; i < A.getM(); ++i) {
            if (A.get(i, i) == 0) throw new RuntimeException("Determinant mustn't be 0");
        }
        return A.getSubMatrix(N, 0, N, M);
    }

    public Matrix convertByKernel(Matrix K, int step) {
        if(K.getN() <= getN() && K.getM() <= getM()) {
            int sizeX = (int)Math.ceil((getN() - K.getN() + 1)/(float)step);
            int sizeY = (int)Math.ceil((getM() - K.getM() + 1)/(float)step);
            Matrix res = new Matrix(sizeX, sizeY);
            for(int y = 0; y < res.getM(); ++y) {
                for(int x = 0; x < res.getN(); ++x) {
                    double val = 0;
                    for(int x1 = x*step; x1 < x*step+K.getN(); ++x1) {
                        for(int y1 = y*step; y1 < y*step + K.getM(); ++y1) {
                            val += get(x1, y1)*K.get(x1-x, y1-y);
                        }
                    }
                    res.set(x, y, val);
                }
            }
            assign(res);
        }
        return this;
    }

    public Matrix convertByMaxPulling(int size) {
        if(size <= getN() && size <= getM()) {
            Matrix res = new Matrix(getN()/size, getM()/size);
            for(int y = 0; y < res.getM(); ++y) {
                for(int x = 0; x < res.getN(); ++x) {
                    double val = 0;
                    for(int x1 = size*x; x1 < size*(x+1); x1++) {
                        for(int y1 = size*y; y1 < size*(y+1); y1++) {
                            val = Math.max(val, get(x1, y1));
                        }
                    }
                    res.set(x, y, val);
                }
            }
            assign(res);
        }
        return this;
    }

    public Matrix convertByMinPulling(int size) {
        if(size <= getN() && size <= getM()) {
            Matrix res = new Matrix(getN()/size, getM()/size);
            for(int y = 0; y < res.getM(); ++y) {
                for(int x = 0; x < res.getN(); ++x) {
                    double val = Double.MAX_VALUE;
                    for(int x1 = size*x; x1 < size*(x+1); x1++) {
                        for(int y1 = size*y; y1 < size*(y+1); y1++) {
                            val = Math.min(val, get(x1, y1));
                        }
                    }
                    res.set(x, y, val);
                }
            }
            assign(res);
        }
        return this;
    }

    public Matrix convertByAveragePulling(int size) {
        if(size <= getN() && size <= getM()) {
            Matrix res = new Matrix(getN()/size, getM()/size);
            for(int y = 0; y < res.getM(); ++y) {
                for(int x = 0; x < res.getN(); ++x) {
                    double val = 0;
                    for(int x1 = size*x; x1 < size*(x+1); x1++) {
                        for(int y1 = size*y; y1 < size*(y+1); y1++) {
                            val += get(x1, y1);
                        }
                    }
                    val /= size*size;
                    res.set(x, y, val);
                }
            }
            assign(res);
        }
        return this;
    }

    public List<Double> getSquareValues(int size, int index) {
        List<Double> res = new ArrayList<>();
        for(Pair<Integer, Integer> pair : getSquare(size, index)) {
            res.add(get(pair.getFirst(), pair.getSecond()));
        }
        return res;
    }

    public List<Pair<Integer, Integer>> getSquare(int size, int index) {
        List<Pair<Integer, Integer>> coords = new ArrayList<>();
        int width = N/size;
        int x = (index%width) * size;
        int y = (index/width) * size;
        for(int x1 = x; x1 < x + size; ++x1) {
            for(int y1 = y; y1 < y + size; ++y1) {
                coords.add(new Pair<>(x1, y1));
            }
        }
        return coords;
    }

    public void sumIntoSquare(int size, int index, double error) {
        int x = (index%N) * size;
        int y = (index/N) * size;
        for(int x1 = x; x1 < x + size; ++x1) {
            for(int y1 = y; y1 < y + size; ++y1) {
                set(x1, y1, get(x1, y1) + error);
            }
        }
    }

    public void print() {
        printShapes();
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                System.out.print(Math.round(get(x, y)*100.0)/100.0+"|");
            }
            System.out.println();
        }
    }

    public String getShapes() {
        return "("+N+";"+M+")";
    }

    public void printShapes() {
        System.out.println(getShapes());
    }

    public String toString() {
        String res = N+" " + M + "\n";
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                res += get(x, y)+" ";
            }
            res += "\n";
        }
        return res;
    }

    public Matrix getSum(double val) {
        Matrix res = new Matrix(N, M);
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res.set(x, y, get(x, y) + val);
            }
        }
        return res;
    }

    public Matrix sum(double val) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) + val);
            }
        }
        return this;
    }

    public double getSum() {
        double res = 0;
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res += get(x, y);
            }
        }
        return res;
    }

    public Matrix getSum(Matrix m) {
        Matrix res = new Matrix(N, M);
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res.set(x, y, get(x, y) + m.get(x, y));
            }
        }
        return res;
    }

    public Matrix sum(Matrix m) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) + m.get(x, y));
            }
        }
        return this;
    }
    public Matrix getMinus(double val) {
        return getSum(-val);
    }

    public Matrix getMinus(Matrix m) {
        return getSum(m.getNegative());
    }

    public Matrix minus(double val) {
        return sum(-val);
    }

    public Matrix minus(Matrix m) {
        return sum(m.getNegative());
    }

    public Matrix getMultiply(double val) {
        Matrix res = new Matrix(N, M);
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res.set(x, y, get(x, y) * val);
            }
        }
        return res;
    }

    public Matrix multiply(double val) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) * val);
            }
        }
        return this;
    }

    public Matrix getMultiply(Matrix m) {
        if(N == m.getM()) {
            Matrix res = new Matrix(m.getN(), M);
            for (int y = 0; y < M; ++y) {
                for (int x = 0; x < m.getN(); ++x) {
                    for(int i = 0; i < N; ++i) {
                        res.set(x, y, res.get(x, y) + get(i, y)*m.get(x, i));
                    }
                }
            }
            return res;
        }
        else if(N == m.getN() && M == m.getM()) {
            Matrix res = new Matrix(N, M);
            for(int y = 0; y < M; ++y) {
                for(int x = 0; x < N; ++x) {
                    res.set(x, y, get(x, y) * m.get(x, y));
                }
            }
            return res;
        }
        return null;
    }

    public Matrix multiply(Matrix m) {
        assign(getMultiply(m));
        return this;
    }

    public int getMaxIndex() {
        int index = 0;
        double max = Double.MIN_VALUE;
        for(int y = 0; y < M; ++y) {
            if(get(0, y) > max) {
                max = get(0, y);
                index = y;
            }
        }
        return index;
    }

    public int getMinIndex() {
        int index = 0;
        double max = Double.MAX_VALUE;
        for(int y = 0; y < M; ++y) {
            if(get(0, y) < max) {
                max = get(0, y);
                index = y;
            }
        }
        return index;
    }
}
