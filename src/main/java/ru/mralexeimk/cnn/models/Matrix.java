package ru.mralexeimk.cnn.models;

import ru.mralexeimk.cnn.enums.PaddingFill;
import ru.mralexeimk.cnn.other.Constants;
import ru.mralexeimk.cnn.other.Pair;

import java.io.Serializable;
import java.util.*;

public class Matrix implements Serializable {
    public double[][] data;
    public int N, M;

    /**
     * Define matrix by keyboard input
     */
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

    /**
     * Define matrix (N,M) with value
     * @param N Columns
     * @param M Rows
     * @param value value
     */
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

    /**
     * Define zero-matrix (N,M)
     * @param N Columns
     * @param M Rows
     */
    public Matrix(int N, int M) {
        this(N, M, 0);
    }

    /**
     * Define matrix (N,M) with random values in range [from; to]
     * @param N Columns
     * @param M Rows
     * @param from Random begin
     * @param to Random end
     */
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

    /**
     * Define matrix by another matrix
     * @param m Matrix
     */
    public Matrix(Matrix m) {
        assign(m);
    }

    /**
     * Define vector (with matrix implementation) by List of values
     */
    public Matrix(List<Double> vector) {
        N = 1;
        M = vector.size();
        data = new double[M][N];
        for(int y = 0; y < M; ++y) {
            data[y][0] = vector.get(y);
        }
    }

    /**
     * Define matrix by pattern string
     * @param pattern Pattern of string
     * @param separatorRows Rows separator
     * @param separatorColumns Columns separator
     */
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

    /**
     * Define matrix by string pattern with '|' separator
     * @param pattern Pattern of string
     */
    public Matrix(String pattern) {
        this(pattern, "\\|", ",");
    }

    /**
     * @return List of matrix values
     */
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

    /**
     * @return Count of columns (width)
     */
    public int getN() {
        return N;
    }

    /**
     * @return Count of rows (height)
     */
    public int getM() {
        return M;
    }

    /**
     * @return Matrix value on (x,y)
     */
    public double get(int x, int y) {
        return data[y][x];
    }

    public void set(int x, int y, double value) {
        data[y][x] = value;
    }

    /**
     * Check if matrix is vector (count of columns equals 1)
     */
    public boolean isVector() {
        return N == 1;
    }

    /**
     * Check if matrix is transposed vector (count of rows equals 1)
     */
    public boolean isTransposedVector() {
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

    /**
     * A - matrix
     * @return -A
     */
    public Matrix getNegative() {
        Matrix res = new Matrix(this);
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                res.set(x, y, -get(x, y));
            }
        }
        return res;
    }

    /**
     * Replace matrix values with matrix 'm' starting at (xStart,yStart)
     */
    public void replace(int xStart, int yStart, Matrix m) {
        if(xStart+m.getN() > N || yStart+m.getM() > M) throw new RuntimeException("Matrix 'm' is too large");
        for(int x = xStart; x < xStart+m.getN(); ++x) {
            for(int y = yStart; y < yStart+m.getM(); ++y) {
                set(x, y, m.get(x-xStart, y-yStart));
            }
        }
    }

    public double[][] getData() {
        return Arrays.stream(data).map(double[]::clone).toArray(double[][]::new);
    }

    /**
     * Expand matrix on right side and bottom side with zero values
     */
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

    /**
     * Increase matrix by (kX,kY) times
     */
    public void increase(int kX, int kY) {
        Matrix res = new Matrix(N*kX, M*kY);
        for(int x = 0; x < res.getN(); ++x) {
            for(int y = 0; y < res.getM(); ++y) {
                res.set(x, y, get(x/kX, y/kY));
            }
        }
        assign(res);
    }

    /**
     * Erase last 'removeN' columns and last 'removeM' rows
     */
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

    /**
     * @return SubMatrix of matrix
     */
    public Matrix getSubMatrix(int xStart, int yStart, int width, int height) {
        Matrix res = new Matrix(width, height);
        for(int x = xStart; x < xStart+width; ++x) {
            for(int y = yStart; y < yStart+height; ++y) {
                res.set(x-xStart, y-yStart, get(x, y));
            }
        }
        return res;
    }

    public double getSum() {
        double res = 0;
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < M; ++j) {
                res += get(i, j);
            }
        }
        return res;
    }

    /**
     * Update diagonal elements with value
     */
    public void setDiagonal(double value) {
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

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (obj.getClass() != this.getClass()) {
            return false;
        }
        Matrix A = (Matrix)obj;

        if(N != A.N || M != A.M) return false;
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < M; ++j) {
                if(Math.abs(get(i, j) - A.get(i, j)) > Constants.EPS) return false;
            }
        }
        return true;
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

    /**
     * Convert matrix to triangular down
     * @return count of swaps
     */
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

    /**
     * Convert matrix to triangular up
     * @return count of swaps
     */
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

    /**
     * @return product of diagonal elements
     */
    public double getTrack() {
        double ans = get(0, 0);
        for(int i = 1; i < Math.min(N, M); ++i) {
            ans *= get(i, i);
        }
        return ans;
    }

    /**
     * @return Determinant of matrix
     */
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

    /**
     * Matrix transpose
     */
    public Matrix transpose() {
        Matrix res = new Matrix(M, N);
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < M; ++y) {
                res.set(y, x, get(x, y));
            }
        }
        assign(res);
        return this;
    }

    public Matrix getTransposed() {
        Matrix res = new Matrix(M, N);
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < M; ++y) {
                res.set(y, x, get(x, y));
            }
        }
        return res;
    }

    /**
     * @return Inverse matrix
     */
    public Matrix inverse() {
        assign(MatrixExtractor.getInverse(this));
        return this;
    }

    public Matrix getInverse() {
        return MatrixExtractor.getInverse(this);
    }

    /**
     * Resize matrix with saving important information
     */
    public Matrix resize(int width, int height) {
        assign(MatrixExtractor.getResized(this, width, height));
        return this;
    }

    public Matrix convertByKernel(Matrix K, int paddingSizeX, int paddingSizeY,
                                  PaddingFill paddingFill, int stridingSizeX, int stridingSizeY) {
        return MatrixExtractor.getConvertByKernel(this, K, paddingSizeX, paddingSizeY,
                paddingFill, stridingSizeX, stridingSizeY);
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

    /**
     * Update matrix value on (xCenter,yCenter) with average of elements in window with K-radius
     */
    public void updateByAverageWindow(int xCenter, int yCenter, int K)  {
        double sum = 0.0;
        int count = 0;
        for (int x = Math.max(0, xCenter-K/2); x <= Math.min(M-1, xCenter+K/2); ++x)  {
            for (int y = Math.max(0, yCenter-K/2); y <= Math.min(N-1, yCenter+K/2); ++y) {
                double val = get(x, y);
                if (val > 0) {
                    sum += val;
                    ++count;
                }
            }
        }
        if (count > 0) {
            set(xCenter, yCenter, sum / count);
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
        StringBuilder res = new StringBuilder(N + " " + M + "\n");
        for(int y = 0; y < M; ++y) {
            for(int x = 0; x < N; ++x) {
                res.append(get(x, y)).append(" ");
            }
            res.append("\n");
        }
        return res.toString();
    }

    public Matrix sum(double val) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) + val);
            }
        }
        return this;
    }

    public Matrix sum(Matrix m) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) + m.get(x, y));
            }
        }
        return this;
    }

    public Matrix diff(double val) {
        return sum(-val);
    }

    public Matrix diff(Matrix m) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) - m.get(x, y));
            }
        }
        return this;
    }

    public Matrix multiply(double val) {
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                set(x, y, get(x, y) * val);
            }
        }
        return this;
    }

    public Matrix multiply(Matrix m) {
        if(N == m.getM()) {
            Matrix res = new Matrix(m.getN(), M);
            for (int y = 0; y < M; ++y) {
                for (int x = 0; x < m.getN(); ++x) {
                    for(int i = 0; i < N; ++i) {
                        res.set(x, y, res.get(x, y) + get(i, y)*m.get(x, i));
                    }
                }
            }
            assign(res);
            return this;
        }
        else if(N == m.getN() && M == m.getM()) {
            Matrix res = new Matrix(N, M);
            for(int y = 0; y < M; ++y) {
                for(int x = 0; x < N; ++x) {
                    res.set(x, y, get(x, y) * m.get(x, y));
                }
            }
            assign(res);
            return this;
        }
        throw new RuntimeException("Can't multiply matrices");
    }

    /**
     * @return Index of maximum element on first column
     */
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

    /**
     * @return Index of minimum element on first column
     */
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
